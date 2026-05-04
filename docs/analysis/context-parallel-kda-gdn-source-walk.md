# FLA 源码走读：Context Parallel support for KDA and GDN 实现解析

在长序列训练里，把序列维度切到多张卡上听起来很自然：每个 rank 只算一段 token，显存压力就能按 Context Parallel size 摊薄。但对 KDA（Kimi Delta Attention）和 GDN（Gated DeltaNet / Gated Delta Rule）这类 delta-rule recurrent 模型来说，事情没有这么简单。它们不是无状态的逐 token MLP，也不是普通 softmax attention 的块状矩阵乘法；后半段 token 的输出依赖前半段 token 累积出来的矩阵状态。

FLA 在 README 中把这个能力描述为：为 KDA 和 GDN 增加 Context Parallel 支持，使训练可以沿 sequence dimension 分布式执行（`README.md:36`）。本文不展开 KDA/GDN 的论文推导，也不讲通用 DDP/FSDP 原理，而是顺着源码回答一个更工程化的问题：**FLA 到底如何让被切开的 rank 重新获得正确的 recurrent state？这个能力是模型层开关，还是算子层能力？它节省了哪些显存，又新增了哪些通信和维护成本？**

# 前言

## 业务 / 工程背景

FLA 的 CP 文档开头直接给出定位：Context Parallel of Linear Attention，又称 KCP，是为 GDN、KDA 这类 delta-rule recurrent model 设计的序列维并行，目标是让每个 rank 处理本地 token chunk，同时自动同步跨 rank state（`fla/ops/cp/README.md:1-4`）。

这里的场景明确是**训练侧长序列扩展**，不是参数分片，也不是 checkpoint 保存优化。普通数据并行复制完整序列，解决不了单卡长序列激活显存；tensor parallel 切 head 或 hidden，也不天然处理“前一段序列的状态如何传给后一段”。CP 正是切 sequence dimension 时补齐因果状态的机制。

## 核心矛盾

这个特性的核心矛盾可以用三句话概括：

1. **序列切分能省激活显存，但 recurrent state 不能被简单切断。** rank1 的第一个 token 需要 rank0 末尾累积出的状态。
2. **直接 all-gather 全部 token 激活太贵。** CP 需要通信，但应通信“状态转移摘要”，而不是把完整 `q/k/v/o` 拼回来。
3. **FLA 当前把 CP 放在 op/runtime context 边界，而不是 HuggingFace model config 边界。** 这让算子能力很清晰，但也带来用户容易误解的模型层接入缺口。

## 本文主线

本文按机制展开：

1. 用户如何真正开启 CP，以及为什么不是配置字段；
2. `FLACPContext` 如何把全局 `cu_seqlens` 变成 rank-local 元数据；
3. KDA/GDN 前向和反向如何注入 state pre-process；
4. all-gather + merge 如何替代全序列 gather；
5. 模型层、短卷积、benchmark patch、backend dispatch 哪些在主路径，哪些不是；
6. 一次真实调用中的 shape、rank、状态和通信如何变化；
7. 显存收益、通信开销、测试覆盖和局限性。

## 不展开的内容

本文不讲 FSDP/DDP/DeviceMesh 基础，不讲 KDA/GDN 论文完整数学推导，不讲 Triton kernel 教程，也不讲 HuggingFace `PreTrainedModel` 通用保存/加载机制。对于源码未确认的意图，本文会标注为“基于源码行为的推断”。

## 核心文件表

| 文件 | 职责 |
|---|---|
| `fla/ops/cp/context.py` | `FLACPContext` 与 `build_cp_context`：构造 rank-local CP 元数据 |
| `fla/ops/cp/comm.py` | CP 通信封装：`all_gather_into_tensor`、conv send/recv alias |
| `fla/ops/cp/chunk_delta_h.py` | CP 核心：前向/反向 state summary、all-gather、merge、`compress_h0` |
| `fla/ops/gated_delta_rule/chunk.py` | GDN 公开 op、autograd Function、CP 参数约束与前后向接入 |
| `fla/ops/kda/chunk.py` | KDA 公开 op、dispatch、autograd Function、CP 参数约束 |
| `fla/ops/kda/chunk_fwd.py` | KDA forward 编排：gate、WY、CP pre-process、state scan、output |
| `fla/ops/kda/chunk_bwd.py` | KDA backward 编排：重算、CP backward pre-process、梯度传播 |
| `fla/layers/kda.py` | KDA 模型层；当前不透传 `cp_context` |
| `fla/layers/gated_deltanet.py` | GDN 模型层；当前不透传 `cp_context` |
| `tests/context_parallel/test_cp_kda.py` / `test_cp_gdn.py` | op 级 CP 正确性测试 |

💡 小结

- CP 的问题背景是长序列训练的序列维切分，而不是参数或 optimizer state 分片。
- 对 KDA/GDN 来说，切序列必须同步 recurrent state，否则数学语义错误。
- FLA 的主实现是 op 级显式 `cp_context`，这决定了后文所有边界和坑点。

# 一、显式算子入口：CP 不是配置开关，而是 `cp_context` 改变执行路径

## 1.1 设计哲学与核心问题

如果这是一个完整训练框架，用户可能期待配置里写 `context_parallel_size=4`，模型 forward 自动切数据、建 group、传 context。但 FLA 是算子/模型库，它没有自己的 Trainer、Engine 或全局 DeviceMesh 管理器。源码里真正的设计是：**调用者先准备分布式环境和局部输入，再把 `FLACPContext` 显式传给低层 op。**

这一层解决的是“如何让一个普通 chunk op 切换到 CP 语义”的问题。它不是通过环境变量，也不是通过 monkey patch，而是通过函数参数 `cp_context`。

## 1.2 源码入口与关键对象

```text
fla/ops/cp/README.md
  - Quick Start：build_cp_context(...) 后直接调用 chunk_kda(..., cp_context=...)

fla/ops/kda/chunk.py
  - chunk_kda：KDA CP 的公开算子入口
  - ChunkKDAFunction：保存 cp_context 并传到 backward

fla/ops/gated_delta_rule/chunk.py
  - chunk_gated_delta_rule：GDN CP 的公开算子入口
  - chunk_gdn：chunk_gated_delta_rule 的 alias
```

文档 Quick Start 展示的入口非常直接：先 `from fla.ops.cp import build_cp_context`，再 `chunk_kda(..., cp_context=cp_context)`（`fla/ops/cp/README.md:7-24`, `fla/ops/cp/README.md:46-59`）。注意它没有展示 `KDAConfig` 或 `KDAForCausalLM` 的配置字段。

KDA 公开 op 的签名包含 `cp_context: FLACPContext = None`（`fla/ops/kda/chunk.py:160-182`），文档字符串也明确说：提供 CP context 时，`initial_state` 和 `output_final_state` 不支持，`cu_seqlens` 会被 context 覆盖（`fla/ops/kda/chunk.py:250-253`）。GDN 的公开 op 同样有 `cp_context` 参数（`fla/ops/gated_delta_rule/chunk.py:353-367`），并给出同样约束（`fla/ops/gated_delta_rule/chunk.py:399-402`）。

## 1.3 主流程拆解

最小 CP 调用不是模型 forward，而是类似下面的 op 调用：

```text
caller / test / benchmark
  -> dist.init_process_group(...)
  -> build_cp_context(global_cu_seqlens, group)
  -> local slice: q[:, rank_start:rank_end], k, v, g, beta
  -> chunk_kda(..., cp_context=context)
     或 chunk_gated_delta_rule(..., cp_context=context)
```

第一个“真正改变行为”的位置在公开 op wrapper：

```python
# KDA: fla/ops/kda/chunk.py:332-340
if cp_context is not None:
    assert initial_state is None
    assert output_final_state is False
    assert cp_context.cu_seqlens is not None
    cu_seqlens = cp_context.cu_seqlens
    if cp_context.cu_seqlens_cpu is not None:
        cu_seqlens_cpu = cp_context.cu_seqlens_cpu
```

GDN 是同样逻辑（`fla/ops/gated_delta_rule/chunk.py:470-476`）。这一段做了三件事：

1. **禁止外部 recurrent cache**：因为跨 rank 初始状态要由 CP pre-process 算出来；
2. **禁止输出 final state**：训练 CP 路径不负责返回可缓存的全局最终状态；
3. **覆盖 `cu_seqlens`**：调用者手动传入的 `cu_seqlens` 不再可信，rank-local 边界以 `cp_context` 为准。

接下来，KDA 的 autograd Function 把 `cp_context` 传入 forward helper（`fla/ops/kda/chunk.py:67-88`），并存入 `ctx.cp_context` 供 backward 使用（`fla/ops/kda/chunk.py:95-109`、`126-148`）。GDN 的 autograd Function 也在 forward 调用中传入 `cp_context`（`fla/ops/gated_delta_rule/chunk.py:280-298`），并在 backward 取回（`fla/ops/gated_delta_rule/chunk.py:299-341`）。

## 1.4 关键细节与误区澄清

这里有一个最容易误解的点：**传 `cu_seqlens` 不等于开启 CP。** `cu_seqlens` 只是 varlen metadata；真正打开 CP 分支的是 `cp_context is not None`。KDA/GDN wrapper 都只有在 `cp_context` 非空时才执行 CP 约束和覆盖（`fla/ops/kda/chunk.py:332-340`, `fla/ops/gated_delta_rule/chunk.py:470-476`）。

第二个误区：**README 的 “support for KDA and GDN” 不是指模型配置里有 CP 开关。** `KDAConfig` 的字段列表没有 CP size、process group 或 context 字段（`fla/models/kda/configuration_kda.py:15-96`），`GatedDeltaNetConfig` 也没有（`fla/models/gated_deltanet/configuration_gated_deltanet.py:17-107`）。源码证据指向 op-level API。

第三个误区：**CP 不支持训练时传入 `initial_state` 或 `output_final_state=True`。** 这不是文档建议，而是 wrapper 的 assert（KDA: `fla/ops/kda/chunk.py:332-334`；GDN: `fla/ops/gated_delta_rule/chunk.py:470-472`）。这直接影响 cache、generation 和 resume 语义，后文会展开。

## 1.5 本章小结

💡 小结

- 用户开启 CP 的最小路径是 `build_cp_context(...)` + 低层 op 的 `cp_context` 参数。
- 第一个行为切换点是 KDA/GDN wrapper 对 `cp_context` 的约束和 `cu_seqlens` 覆盖。
- 当前没有 HF config 级 CP 开关；这是 op 级能力，不是模型层默认能力。

# 二、`FLACPContext` 初始化：把全局序列边界变成 rank-local 状态

## 2.1 设计哲学与核心问题

CP 要解决的第一个工程问题是：每个 rank 只拿到局部 token，但仍要知道自己处理的是哪些全局序列片段。尤其是 varlen batch：一个 rank 可能同时包含上一条 sequence 的尾巴和下一条 sequence 的开头。

如果没有这层元数据，forward 不知道自己是否需要接收前序 state，backward 不知道是否要把梯度传给前一个 rank，短卷积也不知道应该从前一个 rank 取几个 tail token。

## 2.2 源码入口与关键对象

```text
fla/ops/cp/context.py
  - FLACPContext：保存 group、rank-local cu_seqlens、first/last/pre/post rank 信息
  - get_cp_cu_seqlens：核心计算逻辑，带 tensor_cache
  - build_cp_context：用户入口，薄封装 get_cp_cu_seqlens

fla/utils.py
  - tensor_cache：单项 identity cache，会缓存最近一次 context 结果
```

`FLACPContext` 是 dataclass，字段包括 `group`、`cu_seqlens`、`cu_seqlens_cpu`、`is_last_rank`、`pre_num_ranks`、`is_first_rank`、`post_num_ranks`、`conv1d_kernel_size`、`pre_num_conv_tokens`（`fla/ops/cp/context.py:22-33`）。它的 `is_cp_enabled` 只检查 `group is not None`（`fla/ops/cp/context.py:54-57`）。

`build_cp_context` 本身只是把参数转给 `get_cp_cu_seqlens`（`fla/ops/cp/context.py:155-172`），真正计算都在后者。

## 2.3 主流程拆解

`get_cp_cu_seqlens` 的核心逻辑可以简化为：

```text
输入：global cu_seqlens = [0, s1, s1+s2, ..., total]
     group = CP process group

1. world_size = dist.get_world_size(group)
   rank = dist.get_rank(group)

2. total_tokens = cu_seqlens[-1]
   part_len = total_tokens // world_size
   rank_start = part_len * rank
   rank_end = rank_start + part_len

3. 找出与 [rank_start, rank_end) 重叠的序列区间

4. 把全局 cu_seqlens clamp 到本 rank 区间，并减 rank_start
   得到 rank-local cu_seqlens

5. 计算：
   is_first_rank / pre_num_ranks
   is_last_rank / post_num_ranks
   pre_num_conv_tokens
```

源码中 `part_len` 和 rank 区间计算在 `fla/ops/cp/context.py:80-86`；重叠序列用 `torch.searchsorted` 找到（`context.py:87-101`）；local `cu_seqlens` 通过 clamp、subtract、`unique_consecutive` 得到（`context.py:103-114`）。随后计算 first/last rank 元数据（`context.py:116-140`），最后返回 `FLACPContext`（`context.py:142-152`）。

举一个 `world_size=4`、`cu_seqlens=[0,700,1024]` 的例子：

```text
total_tokens = 1024
part_len = 256

rank0: [0,256)     -> seq0 的前段，is_first_rank=True,  post_num_ranks>0
rank1: [256,512)   -> seq0 中段，pre_num_ranks=1, post_num_ranks>0
rank2: [512,768)   -> seq0 尾 + seq1 头，既要接前序 state，也要在反向给后续传梯度
rank3: [768,1024)  -> seq1 尾
```

这也是 GDN/conv 测试注释中反复覆盖的场景：CP4 complex 中第一条序列跨多个 rank，另一条序列也可能被切开（`tests/context_parallel/test_cp_conv.py:304-315`，GDN 测试场景见 `tests/context_parallel/test_cp_gdn.py:342-353`）。

## 2.4 关键细节与误区澄清

最关键的坑：**源码默认 `total_tokens` 能被 `world_size` 整除。** `part_len = total_tokens // world_size` 没有 remainder 处理（`fla/ops/cp/context.py:80-85`）。测试在 worker 开头显式 assert `T % world_size == 0`（KDA: `tests/context_parallel/test_cp_kda.py:126`；GDN: `tests/context_parallel/test_cp_gdn.py:120`），benchmark 也把 `T_total % cp_size != 0` 作为错误（`benchmarks/cp/benchmark_kda_cp8_vs_cp2tp.py:202-207`）。

如果用户传 `total_tokens=10, world_size=3`，`part_len=3`，rank2 只覆盖 `[6,9)`，最后一个 token 没有任何 rank 负责。更糟的是 `last_rank_of_last_seq = (last_seq_global_end - 1) // part_len` 可能得到等于 `world_size` 的 rank id（`fla/ops/cp/context.py:134-140`），后续 merge kernel 会按不存在的 rank 索引 all-gather buffer（`fla/ops/cp/chunk_delta_h.py:463-470`）。这是源码行为直接推出的硬约束。

第二个误区：**`FLACPContext` 不是全局 context manager。** 它没有 `__enter__` / `__exit__`，也没有 thread-local registry。谁创建，谁显式传参；后续函数从参数读取。`copy_for_backward()` 存在（`fla/ops/cp/context.py:35-47`），但仓库搜索未发现调用点，不能把它理解成主流程的一部分。

第三个误区：**`get_cp_cu_seqlens` 有缓存，但不是分布式状态管理。** 它被 `@tensor_cache` 装饰（`fla/ops/cp/context.py:60`），`tensor_cache` 只按参数对象 identity 缓存最近一次结果（`fla/utils.py:114-153`）。这能减少重复构造，但也意味着最近的 `FLACPContext` 会在闭包里持有 process group，直到下一次不同调用覆盖。它不是生命周期管理器。

## 2.5 本章小结

💡 小结

- `build_cp_context` 把全局 `cu_seqlens` 变成 rank-local `cu_seqlens` 和 first/last/pre/post rank metadata。
- 当前源码实际要求 `total_tokens % cp_world_size == 0`；测试和 benchmark 都遵守这一点，但 context builder 本身未显式报错。
- CP 状态是显式对象参数，不是全局开关，也不是 context manager。

# 三、KDA/GDN 前向接入：为什么要在 state scan 之前做 pre-process

## 3.1 设计哲学与核心问题

KDA/GDN 的训练主路径都是 chunk recurrence。每个 rank 本地可以完成 gate、WY 表示、chunk 内计算，但跨 rank 的初始状态不能凭空得到。CP 的前向注入点必须满足两个条件：

1. 它要发生在本地 recurrent state scan 之前；
2. 它要使用和主 state scan 完全一致的张量语义。

这就是为什么 KDA/GDN 都在 `chunk_gated_delta_rule_fwd_h(...)` 之前调用 CP pre-process。

## 3.2 源码入口与关键对象

```text
fla/ops/gated_delta_rule/chunk.py
  - chunk_gated_delta_rule_fwd：GDN forward 主编排，CP 在 lines 78-89 注入

fla/ops/kda/chunk_fwd.py
  - chunk_kda_fwd：KDA forward 主编排，CP 在 lines 81-92 注入

fla/ops/common/chunk_delta_h.py
  - chunk_gated_delta_rule_fwd_h：共享 state scan kernel
```

GDN forward 的 CP 分支在 `fla/ops/gated_delta_rule/chunk.py:78-89`。KDA forward 的 CP 分支在 `fla/ops/kda/chunk_fwd.py:81-92`。两者随后都进入共享 state scan：`chunk_gated_delta_rule_fwd_h`。

## 3.3 主流程拆解

GDN forward 可以简化为：

```text
chunk_gated_delta_rule_fwd(q, k, v, g, beta, cp_context)
  -> g = chunk_local_cumsum(...) 或 gdn_gate_chunk_cumsum(...)
  -> w, u, A = chunk_gated_delta_rule_fwd_intra(k, v, beta, g)
  -> if cp_context:
       initial_state = chunk_gated_delta_rule_fwd_h_pre_process(k=k, w=w, u=u, g=g, ...)
  -> h, v_new, final_state = chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=g, initial_state=...)
  -> if cp_context: initial_state = compress_h0(initial_state)
  -> o = chunk_fwd_o(q, k, v_new, h, g)
```

源码对应关系：gate 和 WY 在 `fla/ops/gated_delta_rule/chunk.py:47-76`，CP pre-process 在 `78-89`，state scan 在 `91-102`，`compress_h0` 在 `104-105`，输出在 `107-118`。

KDA forward 类似，但 KDA 多了一个关键差异：per-dim gate 会先生成 pre-gated `kg/qg`。源码里 `chunk_kda_fwd_intra` 返回 `w, u, qg, kg, Aqk, Akk`（`fla/ops/kda/chunk_fwd.py:66-79`），随后 CP pre-process 使用的是 `k=kg`、`gk=g`、`use_exp2=True`（`chunk_fwd.py:81-92`），主 state scan 同样用 `kg/gk`（`chunk_fwd.py:94-106`）。

这与 CP README 的说明一致：GDN 用原始 `k/q + scalar g`，KDA 用 pre-gated `kg/qg + per-dim gk`，并强调 pre-process 和主 kernel 必须收到相同语义的张量（`fla/ops/cp/README.md:138-167`, `339-350`）。

## 3.4 关键细节与误区澄清

一个常见误区是：**KDA 和 GDN 只是同一个 CP 模板换个函数名。** 通信框架确实共享，但张量语义不同。GDN 的 `g` 是 `[B,T,HV]` 的 scalar per-head log gate（`fla/ops/gated_delta_rule/chunk.py:378-384`），KDA 的 `g` 是 `[B,T,HV,K]` 的 per-dim gate（`fla/ops/kda/chunk.py:193-197`）。KDA pre-process 如果错用原始 `k` 而不是 `kg`，跨 rank 状态的 transition matrix 就和主 state scan 不一致。

第二个误区：**CP pre-process 不是替代 `chunk_gated_delta_rule_fwd_h`。** 它只生成本 rank 的正确 `initial_state`。真正按 chunk 扫描局部 token 的仍是共享 state scan kernel `chunk_gated_delta_rule_fwd_h`（`fla/ops/common/chunk_delta_h.py:662-721`）。

第三个误区：**`output_final_state` 在 CP 中不是“暂时没有用”，而是被禁止。** 因为 public wrapper 已 assert `output_final_state is False`。所以 CP 训练路径不会返回用于 generation cache 的 final recurrent state。

## 3.5 本章小结

💡 小结

- CP 前向插在 WY/intra-chunk 之后、state scan 之前，目标是补齐 rank-local `initial_state`。
- GDN 用原始 `k/q + scalar g`；KDA 用 pre-gated `kg/qg + per-dim gk`，两者不能混淆。
- CP pre-process 不是主计算本身，只是让后续本地 state scan 的初值正确。

# 四、反向链路：为什么前向 all-gather 还不够

## 4.1 设计哲学与核心问题

前向里，后一个 rank 依赖前一个 rank 的 state；反向里方向反过来：前一个 rank 的 tail token 梯度还要接收后一个 rank 因使用它的状态而产生的梯度贡献。

如果只做前向 state 同步，输出可能对得上，但梯度会错。CP 的反向 pre-process 因此要构造“从后续 rank 传回来的 state gradient”，再交给本地 `bwd_dhu` kernel。

## 4.2 源码入口与关键对象

```text
fla/ops/gated_delta_rule/chunk.py
  - chunk_gated_delta_rule_bwd：GDN backward，CP 在 lines 154-197 处理

fla/ops/kda/chunk_bwd.py
  - chunk_kda_bwd：KDA backward，CP 在 lines 470-524 处理

fla/ops/cp/chunk_delta_h.py
  - chunk_gated_delta_rule_bwd_dhu_pre_process：all-gather backward summary 并 merge post ranks
```

## 4.3 主流程拆解

GDN backward 的结构是：

```text
chunk_gated_delta_rule_bwd(..., cp_context)
  -> recompute w, u
  -> if cp_context: initial_state = expand_h0(initial_state)
  -> recompute h, v_new
  -> dv = chunk_bwd_dv_local(...)
  -> if cp_context:
       dht, initial_state = chunk_gated_delta_rule_bwd_dhu_pre_process(...)
  -> dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(...)
  -> dq, dk, dw, dg = chunk_bwd_dqkwg(...)
  -> prepare_wy_repr_bwd(...)
```

源码中 `expand_h0` 在 `fla/ops/gated_delta_rule/chunk.py:154-155`，CP backward pre-process 在 `180-197`。

KDA backward 多了 KDA 专属的 `dAqk/dv` 和 WY/Gate 梯度路径：

```text
chunk_kda_bwd(..., cp_context)
  -> if not disable_recompute:
       recompute g, w, u, qg, kg
       if cp_context: expand_h0(...)
       recompute h, v_new
     else:
       use saved w/u/qg/kg/v_new/h
       if cp_context: expand_h0(...)
  -> dAqk, dv = chunk_kda_bwd_dAv(...)
  -> if cp_context:
       dht, initial_state = chunk_gated_delta_rule_bwd_dhu_pre_process(q=qg, k=kg, gk=g, ...)
  -> dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(q=qg, k=kg, gk=g, ...)
  -> KDA-specific dq/dk/dv/db/dg
```

对应源码：重算路径与 `expand_h0` 在 `fla/ops/kda/chunk_bwd.py:448-491`，`dAqk/dv` 在 `493-505`，CP backward pre-process 在 `507-524`，共享 `bwd_dhu` 在 `526-540`。

## 4.4 关键细节与误区澄清

第一个误区：**反向 CP 不是自动由 PyTorch/NCCL 推出来的。** `ChunkKDAFunction` 和 `ChunkGatedDeltaRuleFunction` 必须显式把 `cp_context` 存在 autograd ctx 上（KDA: `fla/ops/kda/chunk.py:108`；GDN: `fla/ops/gated_delta_rule/chunk.py:306`），backward 才知道要走 CP pre-process。

第二个误区：**`compress_h0` / `expand_h0` 是小优化，不影响正确性？** 恰恰相反，测试目录里的 debug 文档把“compressed initial_state 丢失”列为常见精度失败：如果 forward helper 没有返回更新后的 `initial_state` 并让 autograd 保存它，rank1+ 的 backward 会默默丢失 merged state（`tests/context_parallel/debug.md:69-85`）。源码现在 KDA/GDN 都返回并保存更新后的 state（KDA: `fla/ops/kda/chunk_fwd.py:108-135`, `fla/ops/kda/chunk.py:95-99`；GDN: `fla/ops/gated_delta_rule/chunk.py:104-119`, `299-303`）。

第三个误区：**前向和反向通信完全对称。** 方向对称，但取的本地子序列并不总是同一个。debug 文档指出 forward pre-process 用 `cu_seqlens[-2:]`，即本 rank 最后一段局部序列；backward pre-process 用 `cu_seqlens[:2]`，即本 rank 第一段局部序列（`tests/context_parallel/debug.md:113-125`）。一个 rank 同时拥有某条序列尾和下一条序列头时，两者处理的对象不同。

## 4.5 本章小结

💡 小结

- CP backward 的核心是把后续 rank 的 state gradient 合并成本地 `dht`。
- `cp_context` 必须被 autograd Function 保存，否则 backward 不会知道跨 rank 边界。
- `compress_h0/expand_h0` 是显存优化，也是正确性边界；保存错了会导致 rank1+ 梯度偏差。

# 五、通信与 merge：用 compact state summary 代替全序列 gather

## 5.1 设计哲学与核心问题

序列维并行一定要通信，但通信什么决定了这个特性是否值得。最直观的做法是 all-gather 所有 rank 的 token 激活，算完再切回去；这会把 CP 的显存收益抵消掉。FLA 的 CP 选择通信每个 rank 对 state 的“摘要”：一个外部累积 state 加一个 transition matrix。

基于源码行为和 README 描述，可以把它理解成：每个 rank 本地计算 `(S_ext, M)`，all-gather 后，rank r 把前面若干 rank 的 `(S_ext, M)` 串起来，得到自己第一段 token 的初始状态。

## 5.2 源码入口与关键对象

```text
fla/ops/cp/comm.py
  - all_gather_into_tensor：底层 dist.all_gather_into_tensor 封装

fla/ops/cp/chunk_delta_h.py
  - pre_process_fwd_kernel_merged：生成 hm = [S_ext, M]
  - pre_process_bwd_kernel_merged：生成 dhm = [dS_ext, dM]
  - merge_fwd_bwd_kernel：按 rank 顺序链式 merge
```

`all_gather_into_tensor` 会分配 `[world_size, *inp.shape]`，再调用 `dist.all_gather_into_tensor`（`fla/ops/cp/comm.py:19-41`）。

## 5.3 主流程拆解

前向 CP pre-process 的关键代码在 `fla/ops/cp/chunk_delta_h.py:797-880`：

```text
if context is None or context.group is None:
    return initial_state
assert initial_state is None

hm = zeros([HV, K, V + K], fp32)
initial_state = zeros([N, HV, K, V], fp32)

if not context.is_last_rank:
    pre_process_fwd_kernel_merged(..., hm=hm, cu_seqlens=cu_seqlens[-2:])

ag_hm = all_gather_into_tensor(hm, group=context.group)

if not context.is_first_rank:
    merge_fwd_bwd_kernel(
        h=initial_state[0],
        ag_hm=ag_hm,
        pre_or_post_num_ranks=context.pre_num_ranks,
        rank=rank,
        FORWARD=True,
    )

return initial_state
```

通信对象 `hm` 的 shape 是 `[HV, K, V+K]`（`chunk_delta_h.py:828`），all-gather 后变成 `[world_size, HV, K, V+K]`。它不是完整 token 序列。

反向类似，只是 `dhm` 被 all-gather，非 last rank merge 后续 ranks：`chunk_gated_delta_rule_bwd_dhu_pre_process` 在 `fla/ops/cp/chunk_delta_h.py:883-973`；`dhm` 分配在 `916`，all-gather 在 `948`，merge 在 `950-969`。

merge kernel 的 CP 分支在 `fla/ops/cp/chunk_delta_h.py:452-480`。如果 `FORWARD=True`，它从 `rank - num_ranks` 到 `rank-1` 链式合并；如果 `FORWARD=False`，它从 `rank + num_ranks` 反向合并（`chunk_delta_h.py:463-475`）。这和 README 的 all-gather + merge 公式一致（`fla/ops/cp/README.md:170-185`, `204-241`）。

## 5.4 关键细节与误区澄清

第一个误区：**CP 主路径用了 all-to-all 或 reduce-scatter。** 源码中 KDA/GDN CP state 同步的核心通信是 `all_gather_into_tensor`（forward: `fla/ops/cp/chunk_delta_h.py:859`；backward: `948`）。`all_to_all_single` 出现在 benchmark 的 TP/CP 对比和 patched forward 里（`benchmarks/cp/test_gdn_with_cp.py:87-126`），不是 KDA/GDN op-level CP 主路径。

第二个误区：**all-gather 的是完整输出，生产路径每步都会恢复全序列。** 不对。测试为了验证正确性会把各 rank 的 output/grad all-gather 后和 reference 比较（KDA: `tests/context_parallel/test_cp_kda.py:349-372`；GDN: `tests/context_parallel/test_cp_gdn.py:229-252`），但生产 op 返回的是 local output。CP 自身 all-gather 的是 `hm/dhm` compact summary。

第三个误区：**communication wrapper 里的 `send_recv_fwd/bwd` 是 KDA/GDN state 主路径。** `send_recv_fwd/bwd` 确实通过 all-gather 模拟邻接 send/recv（`fla/ops/cp/comm.py:64-137`），但它们主要服务 conv/token-shift CP alias（`conv_cp_send_recv_fwd/bwd` 在 `comm.py:142-169`）。KDA/GDN recurrent state 主路径使用的是 `chunk_delta_h.py` 里的 `hm/dhm` all-gather。

## 5.5 本章小结

💡 小结

- KDA/GDN CP 的核心通信是 compact state summary 的 all-gather，不是全 token 激活 gather。
- forward merge 串前序 rank，backward merge 串后续 rank。
- 测试里的 output/grad gather 是验证手段，不是生产 CP 计算语义。

# 六、模型层、短卷积与 patch：为什么“算子支持”不等于“模型默认支持”

## 6.1 设计哲学与核心问题

到这里，op-level CP 已经成立。但真实用户往往使用 `KDAForCausalLM` 或 `GatedDeltaNetForCausalLM`，而不是手写 `q/k/v/g/beta`。这就出现了第二个工程矛盾：**op 能 CP，不代表 layer/model 已经把数据切分、context、短卷积边界和 cache 约束都串好了。**

源码里这一层的答案很明确：生产模型层没有接通 `cp_context`；benchmark 里存在 patch/实验路径，但它们不是库的标准主流程。

## 6.2 源码入口与关键对象

```text
fla/layers/kda.py
  - KimiDeltaAttention.forward：读取 cu_seqlens，但不读取/传递 cp_context

fla/layers/gated_deltanet.py
  - GatedDeltaNet.forward：读取 cu_seqlens，但不读取/传递 cp_context

fla/modules/conv/causal_conv1d.py
  - causal_conv1d：支持 cp_context 的短卷积入口

benchmarks/cp/test_gdn_with_cp.py
  - gdn_forward / kda_forward / short_conv_forward：benchmark patch/实验路径
```

KDA layer 在 forward 中只从 kwargs 读取 `cu_seqlens`（`fla/layers/kda.py:218`），调用 `chunk_kda` 时传了 `cu_seqlens`，但没有传 `cp_context`（`fla/layers/kda.py:263-278`）。GDN layer 同样只读取 `cu_seqlens`（`fla/layers/gated_deltanet.py:225`），调用 `chunk_gated_delta_rule` 时也没有 `cp_context`（`fla/layers/gated_deltanet.py:266-279`）。

HF model block 虽然把 `**kwargs` 传给 layer（KDA: `fla/models/kda/modeling_kda.py:85-103`；GDN: `fla/models/gated_deltanet/modeling_gated_deltanet.py:85-103`），但 layer 没消费 `cp_context`，所以传了也不会生效。

## 6.3 主流程拆解

标准 KDA model forward 是：

```text
KDAForCausalLM.forward
  -> KDAModel.forward
    -> for layer in layers:
       -> KDABlock.forward(..., **kwargs)
         -> KimiDeltaAttention.forward(..., **kwargs)
           -> cu_seqlens = kwargs.get("cu_seqlens")
           -> q/k/v short conv
           -> chunk_kda(..., cu_seqlens=cu_seqlens)  # no cp_context
```

标准 GDN model forward 同理：

```text
GatedDeltaNetForCausalLM.forward
  -> GatedDeltaNetModel.forward
    -> GatedDeltaNetBlock.forward(..., **kwargs)
      -> GatedDeltaNet.forward
        -> cu_seqlens = kwargs.get('cu_seqlens')
        -> short conv
        -> chunk_gated_delta_rule(..., cu_seqlens=cu_seqlens)  # no cp_context
```

短卷积其实有 CP 实现：`causal_conv1d` 如果收到 `cp_context`，会 assert 不支持 `initial_state/output_final_state`，然后走 `causal_conv1d_cp`（`fla/modules/conv/causal_conv1d.py:73-84`）。更底层 `causal_conv1d_cp` 要求 `cp_context.conv1d_kernel_size` 和 `cp_context.cu_seqlens`，并只支持 Triton backend（`fla/modules/conv/cp/ops.py:218-258`）。但 KDA/GDN layer 调 `q_conv1d/k_conv1d/v_conv1d` 时也没有传 `cp_context`（KDA: `fla/layers/kda.py:223-244`；GDN: `fla/layers/gated_deltanet.py:230-251`）。

benchmark 里确实有 patch-like forward：`gdn_forward` / `kda_forward` 用 `qkvo_all2ll` 包装 all-to-all（`benchmarks/cp/test_gdn_with_cp.py:351-500`, `503-641`），`test_layer` 里会用 `partial(...)` 替换 `cp_layer.forward` 和短卷积 forward（`benchmarks/cp/test_gdn_with_cp.py:811-825`）。但这些函数在 `benchmarks/` 下，不是 `fla/` 生产路径；而且该 patched path 的重点是 all-to-all/head-parallel layout 实验，并不是低层 `cp_context` state pre-process 的标准接入。

## 6.4 关键细节与误区澄清

第一个误区：**`KDAForCausalLM(...).forward(cp_context=...)` 会启用 CP。** 源码不支持这个结论。kwargs 可以一路传到 layer，但 layer 只取 `cu_seqlens`，不取 `cp_context`。因此模型层默认 forward 不触发 CP state all-gather。

第二个误区：**短卷积 CP 已经自动接入 KDA/GDN layer。** `causal_conv1d` 有 CP 分支，但 `ShortConvolution.forward` 只是把 `**kwargs` 透传给 `causal_conv1d`（`fla/modules/conv/short_conv.py:187-199`）；KDA/GDN layer 调短卷积时没有把 `cp_context` 放进 kwargs。除非外部 patch/wrapper 显式传入，否则短卷积边界不会处理跨 rank tail。

第三个误区：**benchmark patch 是主流程。** 不是。`benchmarks/cp/test_gdn_with_cp.py` 展示了实验性 all-to-all 和 forward replacement，但库主路径没有 monkey patch 注册、没有全局替换，也没有恢复逻辑。文章分析主流程时必须把它归为 benchmark/实验路径。

## 6.5 本章小结

💡 小结

- 当前 CP 是 KDA/GDN op 级能力，不是 HF model/layer 默认能力。
- 短卷积也有 CP 实现，但模型层没有把 `cp_context` 传进去。
- benchmark patch 能帮助理解可能的集成方向，但不能当成生产调用链。

# 七、完整主路径串联：一次真实 CP 调用到底发生了什么

## 7.1 设计哲学与核心问题

前面拆开了 context、forward、backward、通信和模型边界。现在需要把它们串成一次真实调用，区分主流程、初始化路径、测试路径、兼容路径和非主路径。

## 7.2 完整调用栈

以 KDA op-level CP 训练测试为例，主路径是：

```text
User / torchrun / pytest spawned worker
  │
  ├─ Step 1: 初始化 distributed
  │     └─ dist.init_process_group("nccl"), torch.cuda.set_device(rank)
  │        源码：tests/context_parallel/test_cp_kda.py:82-95
  │
  ├─ Step 2: 准备全局数据与 global cu_seqlens
  │     └─ rank0 生成 q/k/v/g/beta/do，broadcast 到其他 rank
  │        源码：tests/context_parallel/test_cp_kda.py:137-205
  │
  ├─ Step 3: 构造 CP context
  │     └─ context = build_cp_context(cu_seqlens_global, group=WORLD)
  │        源码：tests/context_parallel/test_cp_kda.py:305-310
  │
  ├─ Step 4: 按 sequence dimension 切 local tensors
  │     └─ start_idx = rank * (T // world_size)
  │        end_idx = (rank + 1) * (T // world_size)
  │        源码：tests/context_parallel/test_cp_kda.py:311-321
  │
  ├─ Step 5: 调用低层 op
  │     └─ chunk_kda(..., cp_context=context)
  │        源码：tests/context_parallel/test_cp_kda.py:328-344
  │
  ├─ Step 6: wrapper 进入 CP 分支
  │     └─ 禁止 initial_state/output_final_state，覆盖 cu_seqlens
  │        源码：fla/ops/kda/chunk.py:332-340
  │
  ├─ Step 7: forward CP pre-process
  │     └─ local hm -> all_gather -> merge prior ranks -> initial_state
  │        源码：fla/ops/kda/chunk_fwd.py:81-92; fla/ops/cp/chunk_delta_h.py:797-880
  │
  ├─ Step 8: local recurrent state scan + output
  │     └─ chunk_gated_delta_rule_fwd_h + chunk_gla_fwd_o_gk
  │        源码：fla/ops/kda/chunk_fwd.py:94-127
  │
  ├─ Step 9: backward CP pre-process
  │     └─ local dhm -> all_gather -> merge later ranks -> dht
  │        源码：fla/ops/kda/chunk_bwd.py:507-524; fla/ops/cp/chunk_delta_h.py:883-973
  │
  └─ Step 10: 测试收集 output/grads 验证
        └─ dist.all_gather local output/grads，仅测试使用
           源码：tests/context_parallel/test_cp_kda.py:349-395
```

GDN 路径把 Step 5 换成 `chunk_gated_delta_rule(..., cp_context=context)`（`tests/context_parallel/test_cp_gdn.py:215-224`），wrapper 换成 `fla/ops/gated_delta_rule/chunk.py:470-476`，forward/backward helper 换成 `chunk_gated_delta_rule_fwd/bwd` 中的 CP 分支（`fla/ops/gated_delta_rule/chunk.py:78-89`, `180-197`）。

## 7.3 每一层做了什么

| 层 | 输入 | 输出 / 状态 | 通信 | 显存影响 | 执行频率 |
|---|---|---|---|---|---|
| distributed init | rank/world/env | NCCL group | 初始化通信 | 无直接激活收益 | 进程启动 |
| `build_cp_context` | global `cu_seqlens`, group | rank-local `cu_seqlens`, first/last metadata | `get_world_size/get_rank` | 小张量元数据 | 通常每个 batch/context 构造 |
| local slice | global tensor | local `[1,T_local,...]` | 无 | 激活随 `T_local` 缩小 | 每 step |
| op wrapper | local tensors + context | 覆盖 `cu_seqlens`，禁止 cache | 无 | 避免 final state cache | 每 op 调用 |
| forward pre-process | local `k/w/u/g` 或 `kg/w/u/gk` | `initial_state` | all-gather compact `hm` | 额外 `hm/ag_hm` buffer | 每 CP op forward |
| local state scan | local tensors + initial state | `h/v_new/o` | 无 | 只保留 local sequence 激活 | 每 CP op forward |
| backward pre-process | local grad/tensors | `dht` | all-gather compact `dhm` | 额外 `dhm/ag_dhm` buffer | 每 CP op backward |
| test gather | local output/grads | global output/grads | all-gather full tensors | 测试额外内存 | 测试专用 |

## 7.4 哪些逻辑不在主路径

1. **HF 模型层自动 CP**：不在主路径。`KimiDeltaAttention.forward` 和 `GatedDeltaNet.forward` 没传 `cp_context`。
2. **benchmark all-to-all patch**：不在 op-level CP 主路径。它在 `benchmarks/` 下，使用 `all_to_all_single` 做布局实验。
3. **FlashKDA backend**：CP 下会被 verifier 拒绝，原因是 “FlashKDA does not support context parallel”（`fla/ops/kda/backends/flashkda.py:82-83`）。
4. **Intra-card CP backend**：这是 inference/varlen 下的单卡内部拆分，受 `FLA_INTRACARD_CP` 控制（`fla/ops/common/backends/intracard.py:8-14`, `29-67`），不是 `torch.distributed` process-group CP。
5. **save/load hook**：未在 KDA/GDN model class 中看到 CP-specific save/load。`FLACPContext` 是 runtime dataclass，不是 module parameter/buffer。

## 7.5 本章小结

💡 小结

- 真实 CP 主路径从 distributed caller 直接进入 KDA/GDN op，而不是从 HF config 自动触发。
- 每个 CP op forward/backward 都会执行一次 compact all-gather + merge。
- benchmark、FlashKDA、intra-card CP、模型层 kwargs 都要和主路径分开理解。

# 八、关键数据流 / 状态流 / shape 流程

## 8.1 设计哲学与核心问题

CP 的收益和风险都藏在 shape 里：哪些张量按 `T/world_size` 缩小，哪些张量因为 all-gather 变成 `[world_size,...]`，哪些状态只保存第一份，决定了显存和通信的真实边界。

## 8.2 Tensor shape 变化

以 KDA 为例，op 文档给出的基础 shape 是：`q/k: [B,T,H,K]`，`v: [B,T,HV,V]`，`g: [B,T,HV,K]`，`beta: [B,T,HV]`（`fla/ops/kda/chunk.py:186-199`）。CP/varlen 要求 `B==1`，否则 wrapper 在 `cu_seqlens` 路径会报错（`fla/ops/kda/chunk.py:341-346`）。

```text
全局输入（逻辑上）:
  q_global:    [1, T, H,  K]
  k_global:    [1, T, H,  K]
  v_global:    [1, T, HV, V]
  g_global:    [1, T, HV, K]   # KDA
  beta_global: [1, T, HV]

按 rank 切分后:
  q_local:     [1, T / cp_size, H,  K]
  k_local:     [1, T / cp_size, H,  K]
  v_local:     [1, T / cp_size, HV, V]
  g_local:     [1, T / cp_size, HV, K]

KDA WY / gate 后:
  kg/qg:       [1, T_local, HV, K]
  w/u:         [1, T_local, HV, K or V-related]

CP forward summary:
  hm:          [HV, K, V + K]          fp32
  ag_hm:       [cp_size, HV, K, V + K] fp32

本地 state scan:
  h:           [1, NT_local, HV, K, V]
  v_new/o:     [1, T_local, HV, V]

输出:
  o_local:     [1, T_local, HV, V]
```

GDN 的 `g/beta` 是 `[B,T,HV]`，而不是 KDA 的 `[B,T,HV,K]`。GDN op 文档说明 `q/k: [B,T,H,K]`，`v: [B,T,HV,V]`，`g/beta: [B,T,HV]`（`fla/ops/gated_delta_rule/chunk.py:371-384`）。

显存收益主要来自 `T` 相关激活变成 `T_local`：`q/k/v/g/beta/o/v_new/h` 都在每 rank 本地计算。但 `hm/ag_hm` 是新增 buffer，和 `T` 不线性相关，而和 `world_size * HV * K * (V+K)` 相关。

例如 `cp_size=8, HV=64, K=V=128`：

```text
hm 元素数    = 64 * 128 * (128 + 128) = 2,097,152 fp32 ≈ 8 MB
ag_hm 元素数 = 8 * hm = 16,777,216 fp32 ≈ 64 MB
```

forward 有 `ag_hm`，backward 有 `ag_dhm`。这解释了 CP 为什么适合超长 T：当 T 很大时，局部激活收益能覆盖 compact state all-gather 的固定/准固定开销；当 T 不够长或 `HV/K/V/world_size` 很大时，通信 buffer 反而可能显眼。

## 8.3 Rank / Process Group 变化

纯 CP 下，测试直接使用 `dist.group.WORLD`：

```text
world_size = cp_size = 4
rank0 -> local tokens [0, T/4)
rank1 -> local tokens [T/4, T/2)
rank2 -> local tokens [T/2, 3T/4)
rank3 -> local tokens [3T/4, T)
CP group = WORLD
```

CP+TP benchmark 则显式构造两个 group：CP group 是相同 TP rank 的进程，TP group 是相同 CP rank 的进程（`benchmarks/cp/benchmark_kda_cp8_vs_cp2tp.py:169-193`）。示例：`world_size=8, cp_size=4, tp_size=2`：

```text
rank = cp_rank * tp_size + tp_rank

CP group for tp_rank=0: [0, 2, 4, 6]
CP group for tp_rank=1: [1, 3, 5, 7]
TP group for cp_rank=0: [0, 1]
TP group for cp_rank=1: [2, 3]
TP group for cp_rank=2: [4, 5]
TP group for cp_rank=3: [6, 7]
```

这说明 `dist.get_rank(group)` 得到的是 group 内 rank，而不是全局 rank。`build_cp_context` 使用 group 内 rank 计算 `rank_start/rank_end`，所以非连续 global ranks 也能表示 CP 逻辑顺序。

## 8.4 状态切换

CP 没有全局状态切换，但有显式对象状态流：

```text
进入 CP:
  context = build_cp_context(global_cu_seqlens, group)
  写入: group, local cu_seqlens, is_first_rank, pre_num_ranks, is_last_rank, post_num_ranks

执行中:
  chunk_kda / chunk_gated_delta_rule 读取 context.cu_seqlens 覆盖 cu_seqlens
  fwd pre-process 读取 context.group / is_first_rank / pre_num_ranks
  bwd pre-process 读取 context.group / is_last_rank / post_num_ranks

退出:
  没有 __exit__，没有自动恢复
  process group lifecycle 由调用者负责
```

这在多线程意义上不是 thread-local 安全设计；但分布式训练一般是多进程单 GPU，每个进程持有自己的 Python 对象。更值得注意的是 `tensor_cache` 会保留最近一次 `FLACPContext`，可能延长 group 生命周期（`fla/utils.py:133-153`）。

## 8.5 本章小结

💡 小结

- CP 让大多数 `T` 相关激活按 `T/cp_size` 缩小，但新增 `[cp_size, HV, K, V+K]` fp32 summary buffer。
- CP group 的 rank 顺序由 `ProcessGroup` 决定；CP+TP benchmark 说明 CP group 可以不是全局连续 rank。
- CP 状态通过显式 `FLACPContext` 流动，没有自动进入/退出，也没有模型级注册表。

# 九、核心机制深挖

## 9.1 机制一：All-gather + merge 为什么比直接 gather token 更合理

### 设计哲学与核心问题

KDA/GDN 的状态更新可以抽象成“incoming state 经过本地 chunk 的 transition 后，加上本地 token 贡献”。因此每个 rank 不必把全部 token 发给别人，只要发出这段 chunk 对 state 的变换摘要。

### 源码实现

README 把该机制写成三步：本地计算 `S_ext` 和 `M`，all-gather，rank r 链接 rank `<r` 的摘要（`fla/ops/cp/README.md:170-185`）。源码里这个摘要就是 `hm = zeros(HV,K,V+K)`（`fla/ops/cp/chunk_delta_h.py:828`），前 `V` 列可以理解为 state contribution，后 `K` 列保存 transition matrix。merge kernel 在 CP 分支循环 `num_ranks` 次，按 `FORWARD` 选择前序或后序 rank，并执行 `b_h = M @ b_h + he`（非 transpose layout）（`fla/ops/cp/chunk_delta_h.py:463-475`）。

### 隐藏假设与副作用

隐藏假设是 rank 间 token 分段连续、总 token 可均分、所有 rank 都参与同样 collectives。副作用是每个 rank 都拿到所有 rank 的 compact summary，而不只是邻居；实现简单可靠，但通信量是 `O(cp_size * HV * K * (V+K))`。

### 误区澄清

不要把它理解成 pipeline send/recv。虽然语义上像 rank0 把 state 传给 rank1，但实现用 all-gather 保证所有 rank 参与（`fla/ops/cp/comm.py:19-41`）。conv/token-shift 的 send/recv alias 也用 all-gather 实现（`fla/ops/cp/comm.py:64-137`），不是 NCCL P2P。

💡 小结

- CP 通信的是 state transition summary，不是全序列 token。
- merge kernel 用 fp32 链式合并前序/后序 rank。
- 该方案简单但通信 buffer 随 `cp_size` 和 state 维度增长。

## 9.2 机制二：KDA/GDN gate 语义一致性为什么是正确性关键

### 设计哲学与核心问题

KDA 和 GDN 都叫 delta-rule，但 gate 的维度不同。CP pre-process 必须和主 state scan 使用同样的 gate 语义，否则初始 state 是按一种 recurrence 算的，本地 scan 却按另一种 recurrence 继续。

### 源码实现

GDN forward：`g` 经过 cumsum 后，`chunk_gated_delta_rule_fwd_h_pre_process` 收到 `k=k, g=g`（`fla/ops/gated_delta_rule/chunk.py:47-89`）。

KDA forward：先 `kda_gate_chunk_cumsum` 或 `chunk_local_cumsum` 得到 per-dim gate，再 `chunk_kda_fwd_intra` 产出 `kg/qg`，CP pre-process 收到 `k=kg, gk=g, use_exp2=True`（`fla/ops/kda/chunk_fwd.py:43-92`）。KDA backward 同样用 `q=qg, k=kg, gk=g`（`fla/ops/kda/chunk_bwd.py:507-524`）。

README 明确把这列成 input tensor summary，并写出 “Pre-process and main kernel must always receive the same tensors”（`fla/ops/cp/README.md:339-350`）。

### 隐藏假设与维护风险

维护风险在于 CP helper 名字来自 `gated_delta_rule`，KDA 也复用它。读者容易以为它只适配 scalar gate；实际上它靠 `g` / `gk` 两个参数区分 GDN/KDA。如果未来改 KDA gate 表示，必须同步检查 CP pre-process 和主 state scan。

### 误区澄清

`chunk_gated_delta_rule_fwd_h_pre_process` 虽然名字含 GDN，但它是共享 CP state helper，不是只服务 GDN。KDA 传入 `gk` 时走 per-dim gate 路径。

💡 小结

- GDN 和 KDA 共用 CP 通信/merge 框架，但输入张量语义不同。
- KDA 必须使用 pre-gated `kg/qg`；GDN 使用原始 `k/q` 和 scalar gate。
- 共享 helper 名字容易误导，真正语义要看传入的是 `g` 还是 `gk`。

## 9.3 机制三：Backend dispatch 与 CP 的关系

### 设计哲学与核心问题

FLA 有 backend dispatch：同一个 op 可能在特定环境下走外部优化 backend。但 CP 是带 `ProcessGroup` 和 autograd 通信语义的训练路径，不能随便被 inference backend 接管。

### 源码实现

`chunk_kda` 被 `@dispatch('kda')` 装饰（`fla/ops/kda/chunk.py:160-162`）。dispatch wrapper 会懒加载 backend，按优先级验证并调用 backend，否则回退默认实现（`fla/ops/backends/__init__.py:139-195`）。FlashKDA backend 的 verifier 明确拒绝 `cp_context is not None`（`fla/ops/kda/backends/flashkda.py:41-88`，尤其 `82-83`）。

GDN/KDA 共享的 `chunk_gated_delta_rule_fwd_h` 也有 `@dispatch('common')`（`fla/ops/common/chunk_delta_h.py:662-663`）。common 的 `IntraCardCPBackend` 只在 inference mode 且 varlen 下启用（`fla/ops/common/backends/intracard.py:8-14`, `59-67`），受 `FLA_INTRACARD_CP` 控制（`intracard.py:32-35`）。这不是 distributed CP。

### 隐藏假设与副作用

dispatch 带来的副作用是：读源码时不能只看一个函数。KDA 在 CP 下 FlashKDA 会拒绝，因此回到默认 Triton/autograd path；但如果不带 CP，在 inference mode 下可能被 FlashKDA 接管。性能分析必须区分这两条路径。

### 误区澄清

不要把 `FLA_INTRACARD_CP=1` 当成本文的 process-group CP 开关。它是单卡 inference prefill 优化，不构造 `FLACPContext`，也不使用 `torch.distributed` group。

💡 小结

- KDA CP 经过 dispatch，但 FlashKDA 明确拒绝 `cp_context`，所以 CP 训练回到默认实现。
- common intra-card backend 是另一种 inference/varlen 优化，不等价于分布式 CP。
- backend dispatch 是兼容路径，分析主 CP 时必须看 verifier。

## 9.4 机制四：为什么当前没有生产 monkey patch

### 设计哲学与核心问题

用户要求关注 patch。源码里最重要的结论反而是：**生产 CP op 路径不依赖 monkey patch；需要 patch 的是“想把模型层强行接成 CP/TP 实验”的 benchmark。**

### 源码实现

低层 CP 是函数参数接入：`chunk_kda(..., cp_context=...)`、`chunk_gated_delta_rule(..., cp_context=...)`。没有全局替换模块命名空间。

benchmark 中则有显式替换：`test_layer` 对 `cp_layer.forward` 使用 `partial(gdn_forward/kda_forward, ...)`，还替换 `q_conv1d/k_conv1d/v_conv1d.forward`（`benchmarks/cp/test_gdn_with_cp.py:811-825`）。`gdn_forward/kda_forward` 里使用 `qkvo_all2ll`，后者基于自定义 autograd Function 调 `dist.all_to_all_single`（`benchmarks/cp/test_gdn_with_cp.py:87-126`）。

### 隐藏假设与维护风险

这种 patch 没有版本保护、没有自动恢复、没有 save/load 语义，且写在 benchmark 里。它适合做性能和 layout 实验，不适合作为用户训练入口。

### 误区澄清

“CP 必须 patch 才能用”不准确。直接 op CP 不需要 patch；但“HF 模型层想自动 CP”目前确实需要外部 wrapper/patch，因为生产 layer 没透传 `cp_context`。

💡 小结

- op-level CP 是零 monkey patch 的显式参数设计。
- benchmark patch 是实验路径，不是库主流程。
- 模型层自动 CP 尚未产品化，因此外部训练框架若要接模型层，需要自定义集成。

# 十、显存、性能与通信分析

## 10.1 设计哲学与核心问题

CP 本质是用通信和额外 state summary 换长序列激活显存。不能只说“切序列省显存”，必须看哪些内存真的减少，哪些新增，哪些仍然不变。

## 10.2 显存收益范围

| 内容 | 是否节省 | 原因 |
|---|---:|---|
| 参数 | ❌ | CP 不 shard 参数；KDA/GDN layer 参数仍在各 rank 本地完整或由外部 TP/FSDP 管 |
| optimizer state | ❌ | op-level CP 不处理 optimizer；Adam state 等不变 |
| `q/k/v/g/beta` 激活 | ✅ | 每 rank 只持有 `[1,T_local,...]` |
| recurrent intermediate `h/v_new/o` | ✅ | state scan 只覆盖 local tokens/chunks |
| attention logits | 不适用 / ✅ | KDA/GDN 本身不构造 softmax attention 的 `[T,T]` logits |
| `hm/ag_hm`、`dhm/ag_dhm` | ❌ 新增 | CP 为跨 rank state 新增 fp32 compact summary 和 all-gather buffer |
| final recurrent cache | ❌ / 禁用 | CP wrapper 禁止 `output_final_state=True` |
| 输入 batch 存储 | 取决于调用者 | 测试为了 reference 会保留 global tensors；生产可只准备 local slice |
| 保存/加载内存 | ❌ | CP context 不参与权重保存；没有 rank0-only load 优化 |

真正的显存收益来自所有与 `T` 成正比的激活和中间状态按 `T_local=T/cp_size` 缩小。新增成本来自 compact summary：`hm/dhm: [HV,K,V+K] fp32`，all-gather 后 `[cp_size,HV,K,V+K] fp32`。

`compress_h0` 是一个细粒度显存优化：如果 local context 有多个 sequence，只保存第一份可能跨 rank 的 initial state（`fla/ops/cp/chunk_delta_h.py:976-980`），backward 再扩展回 `[N, ...]`（`983-989`）。

## 10.3 通信开销

KDA/GDN op-level CP 每个 op 调用的主要通信：

| 阶段 | 通信 | 源码 | group | 频率 |
|---|---|---|---|---|
| forward pre-process | `dist.all_gather_into_tensor(hm)` | `fla/ops/cp/chunk_delta_h.py:859` | `cp_context.group` | 每个 CP op forward |
| backward pre-process | `dist.all_gather_into_tensor(dhm)` | `fla/ops/cp/chunk_delta_h.py:948` | `cp_context.group` | 每个 CP op backward |
| conv CP forward | all-gather tails | `fla/ops/cp/comm.py:86`, `fla/modules/conv/cp/ops.py:59-71` | `cp_context.group` | 仅 conv CP 被显式传入时 |
| conv CP backward | all-gather d_initial_state | `fla/ops/cp/comm.py:124`, `fla/modules/conv/cp/ops.py:116-118` | `cp_context.group` | 仅 conv CP 被显式传入时 |
| 测试验证 | all-gather output/grads | `tests/context_parallel/test_cp_kda.py:349-372` | WORLD | 测试专用 |
| CP+TP benchmark output | all-reduce | `benchmarks/cp/benchmark_kda_cp8_vs_cp2tp.py:300-303` | TP group | TP 配置专用 |

没有在 KDA/GDN op-level CP 主路径看到 `reduce_scatter` 或 `broadcast`。测试里的 `broadcast` 用于让所有 rank 拿到相同 reference 输入（KDA: `tests/context_parallel/test_cp_kda.py:165-201`；GDN: `tests/context_parallel/test_cp_gdn.py:151-157`），不是生产 op 内部逻辑。

## 10.4 性能取舍

CP 牺牲了三类成本：

1. **每层 forward/backward 的 all-gather 延迟。** 如果每个 KDA/GDN layer 都接 CP，那么每层 forward 一次 all-gather，backward 一次 all-gather。
2. **merge kernel 串行链长。** merge kernel 对 `pre_num_ranks/post_num_ranks` 做循环（`fla/ops/cp/chunk_delta_h.py:463-475`），单条超长序列跨越越多 rank，链越长。
3. **额外 fp32 buffer。** 为了数值稳定，summary 和 merge 使用 fp32；README 也强调 M chain multiply 需要 fp32，避免 bf16 累积误差（`fla/ops/cp/README.md:374-375`）。

换来的收益是 `T` 相关训练激活显著下降，并避免全序列 token all-gather。对于 128K/256K 这类长序列，benchmark 脚本也围绕 CP8、CP2TP 进行比较（`benchmarks/cp/benchmark_kda_cp8_vs_cp2tp.py:8-25`），说明该特性定位在极长序列场景。

## 10.5 本章小结

💡 小结

- CP 省的是序列长度相关激活，不省参数和 optimizer state。
- 新增通信是每个 CP op 的 forward/backward compact all-gather。
- 序列越长，CP 的激活收益越明显；state 维度和 cp size 越大，summary 通信越显眼。

# 十一、配置项、边界条件、测试与局限性

## 11.1 配置如何改变源码路径

| 配置 / 参数 | 影响源码路径 | 行为变化 | 风险 / 坑点 |
|---|---|---|---|
| `cp_context` | `chunk_kda` / `chunk_gated_delta_rule` wrapper | 开启 CP，覆盖 `cu_seqlens`，禁止 initial/final state | 模型层不透传；只能 op/direct/wrapper 使用 |
| `cu_seqlens` | varlen path | `B` 必须为 1，准备 chunk indices | 只传它不启用 CP；CP 下会被 context 覆盖 |
| `total_tokens % world_size` | `get_cp_cu_seqlens` | 源码未校验但实际需整除 | 非整除会丢 token 或产生越界 rank metadata |
| `initial_state` | CP wrapper | CP 下 assert 禁止 | 与 cache/resume state 不兼容 |
| `output_final_state=True` / `use_cache=True` | CP wrapper | CP 下 assert 禁止 | 模型 generation cache 不能直接用于 CP |
| `conv1d_kernel_size` | `causal_conv1d_cp` | conv CP 要求该字段 | KDA/GDN state CP 不需要；短卷积 CP 未被 layer 默认接通 |
| `safe_gate/lower_bound` | KDA gate path | safe gate 要求 lower_bound | KDA CP tests 主要覆盖 safe gate + gate in kernel |
| `use_gate_in_kernel` | KDA/GDN gate path | raw gate 在 kernel 内激活 | KDA CP tests 默认覆盖 True；其他组合覆盖不足 |
| `transpose_state_layout` | state layout | `[K,V]` vs `[V,K]` | KDA/GDN tests 有 transpose 场景 |
| `FLA_FLASH_KDA` | KDA dispatch | 控制 FlashKDA backend | FlashKDA 明确拒绝 CP |
| `FLA_INTRACARD_CP` | common backend dispatch | inference varlen 的 intra-card CP | 不是 distributed CP |

## 11.2 硬约束

1. **总 token 数必须能被 CP size 整除。** 源码用整除计算 `part_len`，测试/benchmark 都显式要求整除。
2. **varlen CP 输入要求 `B==1`。** KDA/GDN wrapper 在 `cu_seqlens` 路径检查 batch size（KDA: `fla/ops/kda/chunk.py:341-346`；GDN: `fla/ops/gated_delta_rule/chunk.py:478-483`）。
3. **K <= 256。** KDA wrapper assert `K <= 256`（`fla/ops/kda/chunk.py:366-369`），共享 state scan 也 assert `K <= 256`（`fla/ops/common/chunk_delta_h.py:689`, `744`）。
4. **GVA/head 约束。** KDA 要求 `HV % H == 0`（`fla/ops/kda/chunk.py:370-375`），GDN 也检查 `HV % H`（`fla/ops/gated_delta_rule/chunk.py:452-463`）。
5. **CP 不支持 user initial/final state。** 这影响 cache、generation 和 resume。

## 11.3 测试证明了什么

KDA 测试：

- 初始化 NCCL 多进程（`tests/context_parallel/test_cp_kda.py:82-100`）；
- 生成全局数据并 broadcast（`137-205`）；
- reference 用 naive recurrent KDA 分序列计算（`207-303`）；
- CP path 构造 context、切 sequence、调用 `chunk_kda(cp_context=context)`（`305-344`）；
- all-gather output/grads 并和 reference 比较（`349-395`）；
- 场景覆盖 CP2/CP4/CP8、sequence cut、boundary aligned、many short sequences、disable recompute、transpose layout（`448-587`）。

GDN 测试：

- reference 用单卡 `chunk_gated_delta_rule` varlen（`tests/context_parallel/test_cp_gdn.py:163-184`）；
- CP path 构造 context、切分并调用 `chunk_gated_delta_rule(cp_context=context)`（`193-224`）；
- all-gather output/grads 验证（`229-269`）；
- 覆盖 CP2/CP4/CP8、many short sequences、GQA、transpose state（`314-457`）。

conv CP 也有单独测试，覆盖短卷积跨 rank tail 和 backward dx 修正（`tests/context_parallel/test_cp_conv.py:8-49`, `161-185`, `190-212`），但这不等于 KDA/GDN layer 已经自动使用 conv CP。

## 11.4 覆盖缺口

| 风险点 | 当前是否有测试 | 可能后果 |
|---|---:|---|
| 非整除 `T % cp_size != 0` | ❌，测试反而 assert 整除 | 丢 token、越界 rank metadata、错误梯度 |
| HF model/layer 透传 `cp_context` | ❌ | 用户以为模型 CP 生效，但实际 no-op |
| 短卷积 CP 与 KDA/GDN layer 联动 | ❌ | sequence 边界卷积状态不正确 |
| KDA `use_gate_in_kernel=False` CP | 覆盖不足 / 主场景不覆盖 | gate 预处理路径 bug 逃逸 |
| KDA GVA `HV > H` CP | 覆盖不足 | value-head 分组梯度风险 |
| `use_beta_sigmoid_in_kernel=True` | ❌ | beta raw/sigmoid 语义风险 |
| CP + save/load/resume | ❌ | runtime group/context/cache 重建不明确 |
| 多机非 WORLD group | benchmark 有 group 构造，测试主要 WORLD | 真实训练 group 生命周期/hang 风险 |
| 性能/显存断言 | ❌ | 只能证明数值，不证明收益 |
| KDA `db` 梯度严格性 | ⚠️ warning-only | beta grad 误差可能被弱化 |

KDA 测试里 `db` 比较设置 `warning=True`，注释说 beta grad 误差更高（`tests/context_parallel/test_cp_kda.py:378-395`）。这不是失败，但说明数值容忍度上有已知敏感点。

## 11.5 已知优化点与维护成本

1. **在 `build_cp_context` 显式校验整除。** 目前调用方校验，库函数不校验。把硬约束前移能避免 silent corruption。
2. **模型层集成 contract。** 如果要让 `KDAForCausalLM`/`GatedDeltaNetForCausalLM` 支持 CP，需要定义 `cp_context` 如何进入 layer、如何传给 short conv、如何禁止 `use_cache`。
3. **减少 all-gather 成本。** 目前所有 rank 收到所有 summary；更复杂的前缀扫描、分层 merge 或异步 overlap 可能降低成本，但源码中未实现。
4. **process group 生命周期文档。** `FLACPContext` 持有 raw group，`tensor_cache` 也会延长最近 context 生命周期；真实训练 resume/destroy group 需要外部纪律。
5. **benchmark patch 产品化风险。** 如果把 `benchmarks/cp/test_gdn_with_cp.py` 的 patch 思路迁入生产，需要版本保护、恢复逻辑、save/load 与测试。

## 11.6 本章小结

💡 小结

- CP 的最小配置不是 config field，而是显式 `cp_context`；`cu_seqlens` 只是 varlen metadata。
- 测试很好地覆盖了 op-level 多 rank 数值主路径，但没有覆盖模型层自动集成和非整除边界。
- 主要维护成本来自 process group 生命周期、模型层缺口、summary all-gather 和 benchmark patch 的非产品化状态。

# 小结与展望

FLA 的 KDA/GDN Context Parallel 实现可以用几个关键词概括。

**关键词一：显式 runtime context。**  
CP 不是 `KDAConfig` / `GatedDeltaNetConfig` 的字段，而是 `FLACPContext` 这个 runtime 对象。它记录 group、rank-local `cu_seqlens`、first/last rank 和前后依赖数量。这个设计让 op 级集成很清晰，也把 distributed lifecycle 留给调用者。

**关键词二：compact all-gather + merge。**  
CP 没有 gather 全部 token，而是 gather `[HV,K,V+K]` 的 state summary。forward merge 前序 rank，backward merge 后序 rank。这是它能在长序列训练中用通信换显存的核心。

**关键词三：KDA/GDN 共享框架但不共享张量语义。**  
GDN 的 CP pre-process 使用原始 `k/q` 和 scalar gate；KDA 必须使用 pre-gated `kg/qg` 和 per-dim `gk`。这是读源码时最容易漏掉、也最影响正确性的细节。

**关键词四：op 级能力，不是模型层默认能力。**  
`chunk_kda` 和 `chunk_gated_delta_rule` 支持 CP；`KimiDeltaAttention.forward` 和 `GatedDeltaNet.forward` 当前不透传 `cp_context`。短卷积虽然有 CP 实现，也没有被模型层自动接上。想在完整模型训练中使用 CP，需要外部训练框架或 wrapper 明确接入。

**关键词五：通信换显存，但边界很硬。**  
CP 适合超长序列、训练态、调用者能控制 rank-local 输入和 process group 的场景。不适合直接拿 HF model forward 当黑盒、需要 generation cache、sequence length 不能整除 CP size、或无法管理 distributed group 生命周期的场景。

与替代方案相比，FLA 这套实现的取舍很鲜明：相比全序列 all-gather，它通信更小；相比完整训练框架级 CP，它更轻量、更靠近算子，但用户集成成本更高；相比 tensor parallel，它直接减少 sequence activation，却不处理参数/optimizer state。后续值得继续走读的方向，是把 op-level CP 产品化到模型层：包括 `cp_context` 透传、short conv CP 接入、cache 禁用策略、非整除 padding/校验、多机 group 测试，以及更低通信成本的 prefix-scan/overlap 实现。
