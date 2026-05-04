# Flash Linear Attention 源码走读：GPT-OSS-style attention sink support 实现解析

在标准 causal attention 里，每个 query 的概率质量最终都会分配给某个历史 key/value；而 GPT-OSS-style attention sink 想增加一个“可学习但不产出 value 的去处”：模型可以把一部分 softmax 概率吸收到一个 per-head 标量上，从而降低对真实 token 的强制分配。对 `flash-linear-attention`（下文简称 FLA）来说，这个特性并不是一次完整的 GPT-OSS 模型接入，而是一次更底层的 kernel 能力扩展：让现有 `parallel_attn` 和 decode kernel 支持一个名为 `sink_bias` 的参数。

本文不展开 GPT-OSS 的模型论文或 attention sink 的外部理论，而是顺着源码回答一个工程问题：FLA 如何在不新增 K/V sink token、不改模型配置、不引入额外通信的前提下，把一个“分母里的 learnable no-op logit”塞进已有 attention kernel，并让 forward/backward、滑窗、varlen、gating 和 TileLang backend 尽量保持一致。

# 前言

## 业务 / 工程背景

README 只用一行宣布了这个功能：`[2026-04] Add GPT-OSS-style attention sink support to fla's attention kernels`（`README.md:35`）。真正的入口并不在 CLI、Trainer 或 HuggingFace model config，而是在低层算子 API：

- `parallel_attn(..., sink_bias=...)`：训练 / prefill 风格的 causal attention 主路径（`fla/ops/attn/parallel.py:769-835`）。
- `attn_decoding_one_step(..., sink_bias=...)`：packed varlen 单步 decode 路径（`fla/ops/attn/decoding.py:119-209`）。
- `naive_parallel_attn(..., sink_bias=...)` / `naive_attn_decoding(..., sink_bias=...)`：测试用 PyTorch reference（`fla/ops/attn/naive.py:12-176`）。

这意味着，用户如果想开启该特性，最小调用不是写配置文件，而是显式传入一个 shape 为 `[HQ]` 的 `sink_bias` 张量：

```python
from fla.ops.attn import parallel_attn

o = parallel_attn(q, k, v, sink_bias=sink_bias)
```

其中 `HQ` 是 query head 数；源码在 `parallel_attn` 中用 `assert sink_bias.shape == (q.shape[2],)` 检查（`fla/ops/attn/parallel.py:831-832`），decode 路径也做了同样检查（`fla/ops/attn/decoding.py:166-167`）。

## 核心矛盾

这个特性的核心矛盾可以概括为三句话：

1. GPT-OSS-style sink 需要参与 softmax 归一化，但它没有对应的 K/V，因此不能简单当作普通 token 拼进 value matmul。
2. FLA 的 `parallel_attn` 已经支持 GQA、sliding window、varlen、gating 和 TileLang dispatch；新增 sink 不能破坏这些路径。
3. 如果 sink 是 learnable 参数，训练时还必须给出正确梯度；但最好不要为它单独改写整套 backward kernel。

FLA 的答案很克制：只把 sink 当作一个 per-query-head bias logit，加到 softmax denominator；forward 不增加 value 项，backward 用已保存的 `lse` 和 `delta` 在 Python 侧补出 `dsink_bias`。

## 本文主线

本文按机制而不是按文件展开：

1. 先看入口：`sink_bias` 如何进入低层 API，为什么不是 model config。
2. 再看 forward：sink 如何只影响分母，不贡献 value。
3. 再看 backward：为什么梯度可以不改主 backward kernel。
4. 再看 decode / varlen / gating / TileLang：同一语义如何穿过多条执行路径。
5. 最后分析 shape、显存、通信、配置、测试覆盖、局限与优化点。

## 不展开的内容

本文不讲 GPT-OSS 模型结构，不讲 FlashAttention 原理，不讲 TileLang 语言本身，也不讲 FLA 其他线性注意力模型。所有结论以当前源码为准；如果源码和“GPT-OSS-style”这个名称带来的直觉不一致，以源码行为为准。

## 核心文件表

| 文件 | 职责 |
|---|---|
| `fla/ops/attn/parallel.py` | `parallel_attn` 主 API、Triton forward/backward、自定义 autograd、`sink_bias` 主路径 |
| `fla/ops/attn/decoding.py` | 单步 packed varlen decode kernel，支持 forward-only sink |
| `fla/ops/attn/naive.py` | PyTorch reference，定义 GPT-OSS-style denominator-only 语义 |
| `fla/ops/backends/__init__.py` | 通用 backend dispatch 框架 |
| `fla/ops/attn/backends/__init__.py` | 注册 attention 的 TileLang backend |
| `fla/ops/common/backends/tilelang/__init__.py` | TileLangBackend 对 `parallel_attn_fwd/bwd` 的接管入口 |
| `fla/ops/common/backends/tilelang/parallel_attn_fwd.py` | TileLang forward 的 sink 分母实现 |
| `fla/ops/common/backends/tilelang/parallel_attn_bwd.py` | TileLang backward 后处理 `dsink_bias` |
| `tests/ops/test_attn_sink.py` | 专门验证 GPT-OSS eager 等价、full/SWA/varlen/gating/decode/empty-row |
| `tests/ops/test_attn.py` | 原 attention 测试中的广义 sink 回归用例 |

# 一、低层 API 接入：它解决的是“如何让模型显式传一个 no-op logit”

## 1.1 设计哲学与核心问题

FLA 没有在某个 `Config` 里加 `use_attention_sink=True`，也没有在某个 Transformer layer 里自动创建 `nn.Parameter`。它做的是更底层的事情：让 attention op 接受一个可选 `sink_bias`，由调用方决定这个张量来自哪里、是否可训练、如何保存。

这层设计解决的是 **API 边界问题**：kernel 只负责“给定 q/k/v/g/window/varlen/sink_bias，算出正确 attention 输出和梯度”；模型层是否引入一个 learnable parameter，不属于当前 patch 的职责。

如果没有这一层，用户只能自己把 sink 当作额外 token 拼到 logits 里，再在 value matmul 前丢掉最后一列；这会破坏 fused kernel 的在线 softmax 结构，也会把 GPT-OSS 的 eager 语义留在 Python 侧，性能和显存都不合适。

## 1.2 源码入口与关键对象

```text
fla/ops/__init__.py
  - parallel_attn：包级导出，用户可 from fla.ops import parallel_attn

fla/ops/attn/__init__.py
  - parallel_attn / naive_parallel_attn：attention 子包导出

fla/ops/attn/parallel.py
  - parallel_attn：公开低层 API，接收 sink_bias
  - ParallelAttentionFunction：autograd 包装，负责 forward/backward 串联
  - parallel_attn_fwd / parallel_attn_bwd：可被 backend dispatch 接管的实现点

fla/ops/attn/decoding.py
  - attn_decoding_one_step：packed varlen decode API，接收 sink_bias
```

包级导出很直接：`fla/ops/__init__.py:8-9` 从 `.attn` 导入 `parallel_attn`，`fla/ops/attn/__init__.py:8-13` 再导出 `naive_parallel_attn` 与 `parallel_attn`。真正改变行为的是 `parallel_attn` 新增的 keyword-only 参数：

```python
# fla/ops/attn/parallel.py:769-835，简化

def parallel_attn(q, k, v, g=None, scale=None, window_size=None,
                  cu_seqlens=None, chunk_indices=None, *, sink_bias=None, **kwargs):
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(...)
    if sink_bias is not None:
        assert sink_bias.shape == (q.shape[2],), "sink_bias must have shape [HQ]"

    return ParallelAttentionFunction.apply(
        q, k, v, g, sink_bias, scale, window_size, cu_seqlens, chunk_indices
    )
```

这段代码说明三件事：

- `sink_bias` 不参与 positional embedding、cache、模型初始化等上游逻辑；它只在 op 调用时出现。
- shape 是 `[HQ]`，不是 `[H]`、不是 `[B, HQ]`，也不是 token 维度。
- varlen 模式仍要求 `q.shape[0] == 1`，sink 并没有改变 FLA 原本 packed varlen 输入约束。

## 1.3 主流程拆解

一次标准训练 / prefill 风格调用的主路径是：

```text
User code
  -> parallel_attn(q, k, v, g=None/..., window_size=..., cu_seqlens=..., sink_bias=s)
    -> ParallelAttentionFunction.forward(...)
      -> sink_bias *= RCP_LN2
      -> parallel_attn_fwd(q, k, v, g_cumsum, sink_bias, ...)
        -> Triton kernel 或 TileLang backend
      -> save_for_backward(q, k, v, o, g_cumsum, lse, sink_bias)
    -> ParallelAttentionFunction.backward(do)
      -> parallel_attn_bwd(..., sink_bias=sink_bias, lse=lse)
      -> return dsink_bias
```

这里有一个容易忽略的状态变化：`sink_bias` 在 `ParallelAttentionFunction.forward` 中乘上 `RCP_LN2`（`fla/ops/attn/parallel.py:726-728`），因为 Triton 主 forward 用 `exp2/log2` 做在线 softmax；原始用户传入的是自然指数语义下的 logit，而 kernel 内部用 base-2 表示。

```python
# fla/ops/attn/parallel.py:726-733，简化

g_cumsum = chunk_global_cumsum(g, scale=RCP_LN2) if g is not None else None
sink_bias = sink_bias * RCP_LN2 if sink_bias is not None else None
o, lse = parallel_attn_fwd(..., sink_bias=sink_bias, ...)
```

## 1.4 关键细节与误区澄清

**误区一：README 说支持 GPT-OSS attention sink，就等于 FLA 模型支持 GPT-OSS attention layer。**

正确结论：当前源码确认的是 **op-level support**。`sink_bias` 在 `fla/models` 中没有命中，在 `fla/layers` 中也没有作为参数被创建或传递。普通 dense `Attention` 仍然调用 `flash_attn_func` / `flash_attn_varlen_func`（`fla/layers/attn.py:146-174`），不是 `fla.ops.attn.parallel_attn`。因此保存、加载、初始化一个 learnable `sink_bias` 需要调用方或未来模型层自己完成。

**误区二：`sink_bias` 是配置项。**

正确结论：不是。`TransformerConfig` 展示了常规 attention 相关字段，如 `num_heads`、`num_kv_heads`、`window_size`、`qk_norm` 等（`fla/models/transformer/configuration_transformer.py:18-91`），没有 `sink_bias` / `attention_sink` 字段。该特性没有 config schema，也没有 validation 层；只有 op 内 shape assert。

## 1.5 本章小结

💡 小结

- FLA 的 attention sink 是低层算子能力，不是完整模型功能。
- 用户开启方式是显式传入 `sink_bias=[HQ]`，而不是改 YAML、CLI 或 HF config。
- 第一个真正改变行为的函数是 `parallel_attn` / `attn_decoding_one_step` 的 `sink_bias` 参数检查与传递。
- 这也解释了为什么初始化、保存、resume 没有 sink 专属代码：当前仓库没有接管这个参数的所有权。

# 二、Forward 语义：sink 只吃概率质量，不产出 value

## 2.1 设计哲学与核心问题

如果把 attention 看成：

```text
o_i = softmax(q_i k_j) @ v_j
```

那么 sink 的难点是：它要进入 softmax denominator，但没有 `v_sink`。最直观的 PyTorch eager 写法是给 logits 拼一列 `sink_bias`，softmax 后再把最后一列丢掉，只用真实 token 的概率去乘 `V`。测试文件里的 GPT-OSS eager reference 正是这么写的：

```python
# tests/ops/test_attn_sink.py:58-63，简化
sink_bias_logits = sink_bias.view(1, HQ, 1, 1).expand(B, HQ, T, 1)
combined_logits = torch.cat((attn_logits, sink_bias_logits), dim=-1)
probs = F.softmax(combined_logits, dim=-1)
attn_probs = probs[..., :-1]
output = torch.matmul(attn_probs, value_states)
```

FLA kernel 不能真的拼这一列，因为 online softmax 是边读 K/V block 边更新最大值和归一化因子。它的工程选择是：等所有真实 key block 处理完以后，把 `exp(sink_bias)` 加到累积 denominator，再统一除输出累积量。

## 2.2 源码入口与关键对象

```text
fla/ops/attn/parallel.py
  - parallel_attn_fwd_kernel：Triton causal attention forward；sink 进入 online softmax normalizer
  - parallel_attn_fwd：分配 o/lse，计算 grid，调用 kernel

fla/ops/attn/naive.py
  - naive_parallel_attn：reference，显式构造 denominator-only sink

tests/ops/test_attn_sink.py
  - _gpt_oss_eager_sink_reference：拼 logits 再 drop sink column 的对照实现
```

## 2.3 主流程拆解

在 Triton forward kernel 里，`USE_SINK_BIAS` 是一个 heuristic constexpr：当 `sink_bias is not None` 时开启专门分支（`fla/ops/attn/parallel.py:21-25`）。每个 program 处理一个 value tile、一个 query block、一个 `(batch, query_head)` 组合：

```text
program_id -> i_v, i_t, i_bh
  i_hq = i_bh % HQ
  i_h  = i_hq // G        # GQA: 多个 query head 共享一个 KV head
  b_q  = load Q block
  b_m  = -inf             # row-wise online max
  b_acc = 0               # row-wise denominator accumulator
  b_o   = 0               # output numerator accumulator

for key block:
  score = q @ k * scale
  apply gate / causal / window mask
  update online max b_m
  update b_acc += exp(score - b_m)
  update b_o += exp(score - b_m) @ v

if sink_bias:
  b_acc += exp(sink_bias[hq] - b_m)

output = b_o / b_acc
lse = b_m + log(b_acc)
```

源码中对应的核心几行在 `fla/ops/attn/parallel.py:167-179`：

```python
if USE_SINK_BIAS:
    b_m = tl.where(b_m == float('-inf'), 0., b_m)
    b_acc += exp2(b_sink_bias - b_m)

b_o = b_o / b_acc[:, None]
b_m += log2(b_acc)
```

这段代码没有修改 `b_o`，只修改 `b_acc`。这正是 denominator-only 语义：sink 分走概率质量，真实 token 的 value 加权和被更大的 denominator 缩小。

reference 也验证了同一个公式：`naive_parallel_attn` 在 `sink_bias is not None` 时不调用普通 `F.softmax(scores)`，而是手工算 `probs_unnorm`、`sink_bias_unnorm` 和 `denom`（`fla/ops/attn/naive.py:85-90`）。

## 2.4 关键细节与误区澄清

**误区三：attention sink 等于“额外的 K/V sink token”。**

正确结论：当前实现没有额外 K/V，没有 cache entry，也没有 value contribution。`parallel_attn` docstring 明确写道，`sink_bias` “Augments the softmax denominator ... without adding a corresponding key/value entry”（`fla/ops/attn/parallel.py:803-814`）。同一段还保留了 `sink_tokens_*` 这个未来名字，说明 Xiao-style K/V sink tokens 尚未实现。

**误区四：sink 会改变输出 shape。**

不会。输入 `q=[B,T,HQ,K]`、`k=[B,T,H,K]`、`v=[B,T,H,V]`，输出仍是 `[B,T,HQ,V]`（`fla/ops/attn/parallel.py:784-818`）。sink 的 shape 只是 `[HQ]`，它不是序列维的新 token。

**误区五：empty row 下没有 key，输出应该来自 sink value。**

当前没有 sink value，所以 empty row 输出是 zero numerator / sink denominator，即零向量。Triton forward 为避免 `-inf + inf` 导致 NaN，在没有有效 key 时把 `b_m` 临时置为 0 再合入 sink（`fla/ops/attn/parallel.py:167-175`）。测试专门覆盖了 `window_size=0` 的 empty-row 场景（`tests/ops/test_attn_sink.py:121-164`、`243-288`）。

## 2.5 本章小结

💡 小结

- forward 的核心不是“拼 token”，而是“给 denominator 加一个 per-head logit”。
- `sink_bias` 会降低真实 token 的总概率质量，但不会产生自己的 value 输出。
- FLA 用 online softmax 的 `b_acc` 扩展实现这个语义，避免显式构造 `[T, T+1]` logits。
- empty-row 稳定性是该 patch 的关键细节之一：没有有效 key 时也不能让 LSE 变 NaN。

# 三、Backward 语义：不用重写主 kernel，也要给 sink 正确梯度

## 3.1 设计哲学与核心问题

训练时 `sink_bias` 是 learnable scalar 的可能性很高；如果 forward 让它参与 softmax denominator，backward 必须返回 `dsink_bias`。但 sink 不参与 value matmul，只影响归一化项，因此它的梯度可以从两个已存在的中间量恢复：

- `lse`：forward 保存的 log-sum-exp，已经包含 sink mass。
- `delta`：`sum(o * do)`，attention backward 常用的 softmax 梯度辅助量。

工程上，这避免了在 `dq` / `dkv` 两个主 backward Triton kernel 里再塞入 sink 分支。FLA 的做法是在 `parallel_attn_bwd` 的末尾用 PyTorch 表达式补出 `dsink_bias`。

## 3.2 源码入口与关键对象

```text
fla/ops/attn/parallel.py
  - parallel_attn_bwd_preprocess：计算 delta = sum(o * do)
  - parallel_attn_bwd_kernel_dq / dkv：主梯度 kernel，不直接处理 sink
  - parallel_attn_bwd：reduce GQA 的 dk/dv 后计算 dsink_bias
  - ParallelAttentionFunction.backward：把 dsink_bias 返回给 autograd

fla/ops/common/backends/tilelang/parallel_attn_bwd.py
  - parallel_attn_bwd_tilelang：TileLang 主梯度后同样在 Python 侧计算 dsink_bias
```

## 3.3 主流程拆解

主 backward 路径如下：

```text
Autograd receives do
  -> ParallelAttentionFunction.backward
    -> load q,k,v,o,g_cumsum,lse,sink_bias
    -> parallel_attn_bwd(...)
       -> delta = parallel_attn_bwd_preprocess(o, do)
       -> dq kernel
       -> dkv kernel
       -> reduce dk/dv from HQ to H if GQA
       -> if sink_bias:
            p_sink = exp2(sink_bias - lse)
            dsink = -sum(p_sink * delta over B,T)
    -> return dq, dk, dv, dg, dsink_bias, None...
```

源码中关键位置在 `fla/ops/attn/parallel.py:637-714`。其中 `dsink_bias` 的公式是：

```python
# fla/ops/attn/parallel.py:709-713

dsink_bias = None
if sink_bias is not None:
    p_sink_bias = torch.exp2(sink_bias[None, None, :] - lse)
    dsink_bias = -(p_sink_bias * delta).sum((0, 1))
```

为什么是负号？直觉上，sink 的概率质量越大，真实 value 输出越被压小；如果上游梯度希望当前输出变大，sink 往往会得到负向梯度。更形式化地说，sink 没有 value 项，因此它只通过 denominator 影响所有真实 token 概率；`delta=sum(o*do)` 正是 softmax backward 中被减掉的项。

`ParallelAttentionFunction.backward` 将它作为第五个 tensor 梯度返回，对应 forward 的第五个输入 `sink_bias`（`fla/ops/attn/parallel.py:748-766`）。这说明训练路径是有 autograd 支持的。

## 3.4 关键细节与误区澄清

**误区六：sink backward 已经完全融合进 Triton 主 backward kernel。**

正确结论：没有。主 `dq/dkv` kernel 不需要知道 `sink_bias`；`dsink_bias` 在 `parallel_attn_bwd` 末尾用 `lse` 与 `delta` 补算（`fla/ops/attn/parallel.py:709-713`）。commit 信息里也提到过 “reuse backward delta” 与 “fuse dsinks into parallel_attn_bwd”，这里的“fuse”是进入 `parallel_attn_bwd` 函数整体，而不是进入每个 Triton kernel。

**误区七：decode 路径也支持 sink 的训练梯度。**

不成立。`attn_decoding_one_step` 直接启动 Triton kernel 并返回 `o`（`fla/ops/attn/decoding.py:189-209`），没有自定义 `torch.autograd.Function` backward。测试也只比较 decode forward 输出（`tests/ops/test_attn_sink.py:374-481`）。这符合 decode 通常用于推理的定位，但不能把它当作训练时可反传的 sink path。

## 3.5 本章小结

💡 小结

- `parallel_attn` 的 sink 训练梯度是有的，返回给 autograd 的是 `dsink_bias`。
- 这个梯度没有塞进主 `dq/dkv` kernel，而是利用 `lse` 和 `delta` 在 backward 函数末尾补算。
- decode sink 是 forward-only 的工程路径，测试没有覆盖其 autograd。
- 这种设计降低了 kernel 改动面，但也留下了 dtype、device、极端数值和 TileLang parity 的测试压力。

# 四、完整主路径串联

## 4.1 完整调用栈

以用户显式调用 `parallel_attn(q, k, v, sink_bias=s)` 为例，真实主路径可以串成：

```text
User: parallel_attn(q, k, v, sink_bias=s, window_size=W, cu_seqlens=cu)
  │
  ├─ Step 1: 入口检查
  │     └─ fla/ops/attn/parallel.py:769-835
  │        - 默认 scale = 1/sqrt(K)
  │        - cu_seqlens 模式要求 B=1
  │        - sink_bias shape 必须是 [HQ]
  │
  ├─ Step 2: autograd forward 包装
  │     └─ ParallelAttentionFunction.forward, parallel.py:723-743
  │        - g -> g_cumsum
  │        - sink_bias -> base-2 logit
  │        - 调 parallel_attn_fwd
  │        - 保存 q/k/v/o/g_cumsum/lse/sink_bias
  │
  ├─ Step 3: backend dispatch / kernel forward
  │     ├─ 默认 Triton: parallel_attn_fwd_kernel, parallel.py:21-180
  │     └─ TileLang 可选: TileLangBackend, tilelang/__init__.py:100-135
  │        - 计算 online softmax
  │        - sink 只加 denominator
  │        - 输出 o 和 lse
  │
  ├─ Step 4: backward
  │     └─ ParallelAttentionFunction.backward, parallel.py:748-766
  │        - 调 parallel_attn_bwd
  │        - 主 kernel 算 dq/dk/dv/dg
  │        - 用 lse/delta 算 dsink_bias
  │
  └─ Step 5: 返回梯度
        └─ dq, dk, dv, dg, dsink_bias
```

## 4.2 每一层做了什么

| 层级 | 输入 | 输出 / 状态 | 通信 | 显存影响 | 频率 |
|---|---|---|---|---|---|
| API 检查 | `q,k,v,sink_bias` | `scale` 默认值、shape 检查 | 无 | 无 | 每次调用 |
| Autograd forward | `g,sink_bias` | `g_cumsum`、base-2 `sink_bias`、保存 `lse` | 无 | 保存 backward 所需 tensor | 每次 forward |
| Kernel forward | `q,k,v,g_cumsum,sink_bias` | `o,lse` | 无 | 不构造完整 logits；额外 `lse=[B,T,HQ]` 本来就需要 | 每层/每次调用 |
| Kernel backward | `do,o,lse` | `dq,dk,dv,dg,dsink` | 无 | GQA 时临时 `dk/dv` 可能按 HQ 分配再 reduce | 每次 backward |
| TileLang dispatch | 同上 | 同上 | 无 | 可能创建 dummy `sink_kern` / `g_kern` | 每次调用，可被 env 控制 |

这里没有 process group、device mesh、rank mapping，也没有 `all_gather` / `reduce_scatter`。针对 `fla/ops/attn` 与 TileLang attention 目录的通信关键词搜索只命中了注释中的 “barrier” 等局部同步语义，没有分布式通信原语。换言之，attention sink 不是一个分布式训练特性；它只是每个 rank 本地 attention op 的计算扩展。

## 4.3 哪些逻辑不在主路径

- `fla/layers/attn.py` 的 dense `Attention`：它走 `flash_attn_func` / `flash_attn_varlen_func`（`fla/layers/attn.py:146-174`），没有把 `sink_bias` 传给 FLA `parallel_attn`。
- `fla/layers/forgetting_attn.py` / `fla/layers/path_attn.py` 的 decode 调用：它们调用 `attn_decoding_one_step`，但没有传 `sink_bias`（`fla/layers/forgetting_attn.py:127`、`fla/layers/path_attn.py:184`）。
- `parallel_forgetting_attn`：内部调用 `parallel_attn(q,k,v,g,...)` 时也没有暴露 `sink_bias`（`fla/ops/forgetting_attn/parallel.py:61`）。
- `sink_tokens_*`：只在 docstring 中作为未来保留命名出现（`fla/ops/attn/parallel.py:812-814`），当前没有实现。
- 保存 / 加载 / checkpoint：没有 sink 专属逻辑。因为当前没有模型层 `nn.Parameter`，state dict 自然也没有 sink 参数。

## 4.4 本章小结

💡 小结

- 主路径非常短：低层 API → autograd wrapper → forward kernel → backward 函数补 `dsink_bias`。
- 该特性没有 Trainer、CLI、model config、save/load、process group 的初始化路径。
- 看似复用 attention kernel 的上层 layer 并不会自动获得 sink；必须显式传参才生效。
- TileLang 是同一函数名下的可选 backend，不是另一套用户入口。

# 五、关键数据流 / 状态流 / shape 流程

## 5.1 Tensor shape 变化

标准 `parallel_attn` 的 shape 流如下：

```text
输入:
  q:         [B, T, HQ, K]
  k:         [B, T, H,  K]
  v:         [B, T, H,  V]
  g:         [B, T, HQ]      optional
  sink_bias: [HQ]            optional

GQA 映射:
  G = HQ // H
  query head i_hq -> kv head i_h = i_hq // G

Triton forward block:
  b_q:       [BT, BK]
  b_s:       [BT, BS]
  b_o:       [BT, BV]
  b_acc:     [BT]
  b_sink:    scalar for i_hq

输出:
  o:         [B, T, HQ, V]
  lse:       [B, T, HQ]
```

varlen 模式不是把 batch 维保留为 `B>1`，而是要求外部 flatten：`parallel_attn` 在 `cu_seqlens is not None and q.shape[0] != 1` 时直接报错（`fla/ops/attn/parallel.py:826-830`）。之后 `prepare_chunk_indices(cu_seqlens, BT)` 生成 `(sequence_id, chunk_id)` 索引（`fla/ops/utils/index.py:137-148`），kernel 中用 `bos/eos` 切回每条序列范围（`fla/ops/attn/parallel.py:60-66`）。

```text
varlen 输入:
  q/k/v:     [1, total_T, heads, dim]
  cu_seqlens:[N+1]

prepare_chunk_indices:
  chunk_indices: [num_chunks, 2]
    each row = [sequence_id, chunk_id_inside_sequence]

kernel 内:
  bos = cu_seqlens[sequence_id]
  eos = cu_seqlens[sequence_id+1]
  local T = eos - bos
```

sink 不改变这些 shape，只在每个 query head 上提供一个 scalar。

## 5.2 Rank / Mesh / Process Group 变化

这一节反而要明确说“没有”。源码中 `fla/ops/attn` 的 sink 相关文件没有 `torch.distributed`、`all_gather`、`all_to_all`、`reduce_scatter`、`broadcast` 等调用。每个 rank 如果在分布式训练中执行这个 op，它只在本地 GPU 上使用本地 shard / local batch 的 `q/k/v/sink_bias`。

```text
world_size = N 时:
  rank0: parallel_attn(local_q0, local_k0, local_v0, sink_bias0)
  rank1: parallel_attn(local_q1, local_k1, local_v1, sink_bias1)
  ...

attention sink 本身:
  不创建 group
  不切换 mesh
  不广播 sink_bias
  不 reduce dsink_bias
```

如果 `sink_bias` 是模型参数，DDP/FSDP 是否同步它由上层训练框架负责；FLA kernel 只返回梯度。

## 5.3 状态切换：backend dispatch 是唯一“全局状态”相关机制

当前 sink 没有 monkey patch，也没有 context manager。但 `parallel_attn_fwd` 和 `parallel_attn_bwd` 被 `@dispatch('attn')` 装饰（`fla/ops/attn/parallel.py:508`、`594`）。第一次调用时，dispatch 会 lazy import `fla.ops.attn.backends`（`fla/ops/backends/__init__.py:122-136`），后者注册 TileLang backend（`fla/ops/attn/backends/__init__.py:8-11`）。

```text
进入 parallel_attn_fwd:
  dispatch wrapper
    -> BackendRegistry.ensure_initialized('attn')
      -> import fla.ops.attn.backends
      -> register(TileLangBackend)
    -> 遍历 backend
      -> be.can_use()
      -> be.verify(...)
      -> 若可用，调用 TileLangBackend.parallel_attn_fwd
      -> 否则回落到默认 Triton parallel_attn_fwd
```

这个 dispatch 受环境变量影响：

- `FLA_DISABLE_BACKEND_DISPATCH=1`：完全绕过 dispatch（`fla/ops/backends/__init__.py:25-27`、`145-147`）。
- `FLA_TILELANG`：控制 TileLang backend 是否启用；文档说明默认 unset 等价启用，装了 `tilelang` 就可能接管（`ENVs.md:47-56`）。

这不是 monkey patch：函数名没有被替换，dispatch wrapper 是装饰器定义时就存在的，backend 是 registry 选择。

## 5.4 本章小结

💡 小结

- `sink_bias` 的 shape 是 `[HQ]`，本质是每个 query head 一个 scalar。
- 它不增加 token 维，也不改变 `o=[B,T,HQ,V]` 的输出契约。
- varlen 的复杂性来自 `cu_seqlens/chunk_indices`，不是 sink 本身。
- 没有分布式状态切换；唯一需要注意的全局行为是 backend dispatch 与 `FLA_TILELANG` / `FLA_DISABLE_BACKEND_DISPATCH`。

# 六、核心机制深挖

## 6.1 Denominator-only sink：为什么不能简单拼 K/V？

如果要完全模拟“一个额外可注意位置”，最简单做法似乎是拼一个 K/V token。但 GPT-OSS-style sink bias 的当前语义并不是 K/V token，它没有 value；拼 K/V 反而会改变模型表达。FLA 源码明确将其定义为 “no-op target”：`p_i = exp(s_i)/(sum_j exp(s_j)+exp(sink_bias[h]))`，输出仍是 `sum_i p_i*v_i`（`fla/ops/attn/parallel.py:807-811`）。

这带来两个隐含假设：

1. sink 只用于吸收概率质量，不需要被 cache。
2. sink 的位置不依赖序列长度、batch、token，只依赖 query head。

副作用也很清晰：当 `sink_bias` 很大时，真实 token 概率整体变小，输出趋近零；当它很小时，退化回普通 attention。

## 6.2 Backward 后处理：为什么可以只靠 LSE 和 delta？

主 backward kernel 已经需要 `lse` 和 `delta` 来计算 softmax 梯度。sink 的概率可以由：

```text
p_sink[b,t,h] = exp(sink_bias[h] - lse[b,t,h])
```

恢复出来。于是 `dsink_bias = -sum(p_sink * delta)` 可以在 Python/PyTorch 侧完成。这种设计避免在 `parallel_attn_bwd_kernel_dq` 和 `parallel_attn_bwd_kernel_dkv` 两处都加入 sink 逻辑。

维护风险在于：这要求 forward 保存的 `lse` 一定包含 sink mass。Triton forward 在 sink 分支后才 `b_m += log2(b_acc)` 并写回 `lse`（`fla/ops/attn/parallel.py:167-180`），TileLang forward 也在加入 sink 后 finalize LSE（`fla/ops/common/backends/tilelang/parallel_attn_fwd.py:183-198`）。只要未来有人改了 LSE 语义，`dsink_bias` 就会被连带破坏。

## 6.3 TileLang backend：零用户感知，双实现一致性压力

TileLang backend 的接入发生在 registry，而不是用户 API。`TileLangBackend` 定义 `parallel_attn_fwd` / `parallel_attn_bwd` 两个方法（`fla/ops/common/backends/tilelang/__init__.py:116-180`），dispatch 如果选中它，就会调用 TileLang 文件。

TileLang forward 与 Triton 一样只加 denominator：

```python
# fla/ops/common/backends/tilelang/parallel_attn_fwd.py:183-191，简化
if _USE_SINK:
    if m_f[i] == -inf: m_f[i] = 0
    l_f[i] += T.exp2(sink_bias[i_h] - m_f[i])
```

TileLang backward 也不改主 kernel，而是后处理：

```python
# fla/ops/common/backends/tilelang/parallel_attn_bwd.py:289-295
p_sink = torch.exp2(sink_bias.float()[None, None, :] - lse.float())
dsink_bias = -(p_sink * delta.float()).sum((0, 1))
```

隐藏风险是：TileLang 是可选依赖，`pyproject.toml` 把它放在 optional dependency：`tilelang = ["tilelang>=0.1.9"]`（`pyproject.toml:24-28`）。当前环境下 code-reviewer 子任务确认 `tilelang` 未安装，因此运行验证覆盖的是 Triton 默认路径，TileLang 只做了源码审阅。

## 6.4 Decode kernel：推理路径需要同语义，但不需要 backward

`attn_decoding_one_step` 面向 `[1,B,HQ,K]` 的单 query decode，`k/v` 是 packed 的 `[1,total_T,H,D]`，必须提供 `cu_seqlens`（`fla/ops/attn/decoding.py:119-159`）。它也有 `USE_SINK_BIAS` heuristic（`fla/ops/attn/decoding.py:17-20`），kernel 中在处理完所有 KV 后：

```python
# fla/ops/attn/decoding.py:111-116
if USE_SINK_BIAS:
    b_m = tl.where(b_m == float('-inf'), 0., b_m)
    b_acc += exp(b_sink_bias - b_m)
b_o = b_o / b_acc
```

这里使用的是自然指数 `exp`，因为 decode kernel 的 score 已经以自然指数 softmax 的方式处理，不像 `parallel_attn` 主 kernel 那样整体用 `exp2/log2`。这也是为什么 `parallel_attn` 的 `sink_bias` 要乘 `RCP_LN2`，而 decode path 不做这一步。

## 6.5 本章小结

💡 小结

- 核心机制可以用一句话概括：sink 是 softmax 分母里的 per-head no-op logit。
- backward 的关键复用点是 `lse` 和 `delta`，因此不需要大改主梯度 kernel。
- TileLang 通过 dispatch 零感知接入，但也带来双实现一致性测试压力。
- decode 实现保持 forward 语义一致，但不是训练 autograd 主路径。

# 七、显存、性能与通信分析

## 7.1 显存收益范围

| 内容 | 是否节省 | 原因 |
|---|---:|---|
| 参数 | ❌ | 当前 kernel 不创建参数；若上层加 `sink_bias`，只增加 `[HQ]` 标量 |
| 激活值 | ⚠️ 局部节省 | 相比 eager 拼接 `[T,T+1]` logits，kernel 不物化完整 logits；但 FLA 原本 fused attention 就不物化完整 logits |
| logits | ✅ 相对 eager | sink 不通过 `torch.cat` 产生额外 logits 列；只在 register/fragment 中加 denominator |
| logits 输出 | ❌ | `parallel_attn` 仍只输出 `o`，没有额外返回 sink prob |
| optimizer state | ❌ | kernel 不负责参数；若上层注册 sink 参数，会有对应 optimizer state |
| 输入 batch | ❌ | sink 不改变 padding/unpadding 或 varlen flatten 方式 |
| 中间 buffer | ⚠️ 小幅增加 | 需要保存/使用 `sink_bias`，但主额外量是 `[HQ]`；`lse=[B,T,HQ]` 原本 backward 就需要 |
| GQA backward 临时 dk/dv | ❌ | `parallel_attn_bwd` 仍可能分配 `[B,T,HQ,K/V]` 临时再 reduce 到 H（`fla/ops/attn/parallel.py:639-705`） |

真正的显存大头仍是 Q/K/V/O、反向中间量和模型层激活，而不是 `[HQ]` 的 sink。sink 的显存收益主要是“没有退回到 eager 拼 logits + softmax + drop column”的路径；在 fused attention kernel 内，它几乎只是一个 scalar load 和 denominator 累加。

## 7.2 通信开销

该特性本身没有通信开销：

| 阶段 | 通信类型 | 次数 | 说明 |
|---|---|---:|---|
| forward | 无 | 0 | 每个 rank 本地 kernel 计算 |
| backward | 无 | 0 | `dsink_bias` 本地计算；DDP/FSDP 参数同步由上层处理 |
| save/load | 无 | 0 | 仓库没有 sink 参数保存逻辑 |
| TileLang dispatch | 无 | 0 | registry 选择本地 backend，不是分布式通信 |

如果上层模型把 `sink_bias` 注册成参数，那么梯度同步会和普通参数一样发生；但这不是 FLA sink kernel 的源码行为，本文未在源码中确认任何 sink 专属分布式逻辑。

## 7.3 性能取舍

这个实现的性能取舍很明确：

- 用极小的额外计算换表达能力：每个 query row 多一次 `exp(sink_bias - max)` 和 denominator 加法。
- 用后处理梯度换 kernel 简洁：`dsink_bias` 不是完全 fused 到 Triton backward kernel，可能多一个 PyTorch reduction，但规模是 `[B,T,HQ]`，通常远小于 attention matmul。
- 用 backend dispatch 换硬件适配：TileLang 可接管，但如果安装了 TileLang，用户可能在不知情情况下跑另一套实现。
- 用 `assert` 简化校验，牺牲生产健壮性：Python `-O` 会禁用 assert，错误 shape 可能变成更晚的 kernel 失败。

## 7.4 本章小结

💡 小结

- sink 不解决模型激活显存大头，它避免的是 eager attention sink 的 logits 物化方式。
- 没有新增分布式通信，也没有 rank0 聚合或 process group 切换。
- 性能成本主要是每行一个额外 normalizer 项和 backward 的 `[B,T,HQ]` reduction。
- 更大的实际风险不是性能，而是 backend 一致性与上层模型未集成。

# 八、配置项、边界条件与坑点

## 8.1 配置如何改变源码路径

| 配置 / 条件 | 影响源码路径 | 行为变化 | 风险 / 坑点 |
|---|---|---|---|
| `sink_bias=None` | `parallel.py:21-25` heuristic | `USE_SINK_BIAS=False`，普通 attention | 默认不启用 |
| `sink_bias.shape == [HQ]` | `parallel.py:831-832`、`decoding.py:166-167` | 才允许 kernel load per-head bias | 用 `assert`，`python -O` 下可能失效 |
| `cu_seqlens is not None` | `parallel.py:826-830`、`utils/index.py:137-148` | 要求 `B=1` packed varlen，准备 chunk indices | 不是普通 `[B,T]` mask；调用方要先 flatten |
| `window_size` | `parallel.py:96-117`、`151-154` | 限制可见 key 范围，sink 仍总是可见 | `window_size=0` 时只有 sink denominator，输出趋零 |
| `g is not None` | `parallel.py:726`、`110-114` | gate 先转 `g_cumsum`，再加入 score | sink 与 gate 并存，测试覆盖 full/SWA/varlen |
| `FLA_DISABLE_BACKEND_DISPATCH=1` | `backends/__init__.py:25-27`、`145-147` | 强制走默认 Triton | 调试 TileLang 差异时有用 |
| `FLA_TILELANG=0/1` | `ENVs.md:47-56`、`tilelang/__init__.py:21-26` | 控制 TileLang backend 是否可接管 | TileLang 未安装则回落；安装后需额外验证 |
| `head_first` kwarg | `parallel.py:820-823` | 直接抛 `DeprecationWarning` | 老调用方式不能用于 sink 新路径 |

## 8.2 最小配置与默认行为

最小开启条件：

```python
sink_bias = torch.zeros(q.shape[2], device=q.device, dtype=torch.float32)
o = parallel_attn(q, k, v, sink_bias=sink_bias)
```

默认行为是 `sink_bias=None`，完全不启用。没有 deprecated 的 sink 配置项；但 commit body 显示曾将 `sinks` 重命名为 `sink_bias`，并保留 `sink_tokens_*` 给未来实现。当前源码只认 `sink_bias`。

## 8.3 静默失效与不兼容组合

- **模型层静默无效**：给 Transformer config 加自定义 `sink_bias` 字段不会自动生效，因为模型层没有消费它。
- **dense Attention 不走 FLA attention kernel**：`fla/layers/attn.py` 使用 flash-attn，不会读取 `sink_bias`。
- **Forgetting / PaTH wrapper 不透传 sink**：底层 op 可支持，但 wrapper 调用没有传参。
- **TileLang 差异**：安装 TileLang 且未禁用时可能走 TileLang；当前本地验证未运行 TileLang。
- **`V != K` reference 风险**：code-reviewer 子任务指出 `naive_parallel_attn` 在 sink 分支用从 `q` 取到的 `D` reshape 输出（`fla/ops/attn/naive.py:46`、`89-90`），对 `V != K` 的 reference 可能失败；生产 kernel 本身分开处理 `K` 和 `V`。

## 8.4 本章小结

💡 小结

- 真正的“配置”只有函数参数和 backend env var；没有模型级 schema。
- `sink_bias` 的 shape、device、dtype 需要调用方负责，源码只做有限 assert。
- `FLA_TILELANG` 和 `FLA_DISABLE_BACKEND_DISPATCH` 会改变实际实现路径。
- 最大坑点是把 op-level support 误认为 end-to-end model support。

# 九、测试、示例与覆盖缺口

## 9.1 已覆盖路径

`tests/ops/test_attn_sink.py` 是这次特性的核心测试文件，共 481 行。它证明的不是“模型能训练 GPT-OSS”，而是“kernel 语义与 GPT-OSS eager/reference 一致”。

| 测试 | 覆盖的行为 | 说明 |
|---|---|---|
| `test_attn_sink_ref_matches_gpt_oss_eager`（`tests/ops/test_attn_sink.py:74-118`） | reference vs GPT-OSS eager，full/SWA，含梯度 | 验证 denominator-only 语义 |
| `test_attn_sink_empty_row_ref_matches_gpt_oss_eager`（`121-164`） | empty-row reference 等价 | 防 NaN 语义 |
| `test_parallel_attn_sink_matches_reference`（`167-240`） | Triton `parallel_attn` vs reference，full/SWA/varlen，含 `dsink` | 主训练路径 |
| `test_parallel_attn_sink_empty_row_matches_reference`（`243-288`） | `window_size=0` empty-row kernel | 稳定性 |
| `test_parallel_attn_sink_with_g_matches_reference`（`291-371`） | sink + gating `g` + full/SWA/varlen | 与 FoX/forget gate 类机制兼容 |
| `test_attn_decoding_sink_matches_reference`（`374-403`） | decode forward | 推理单步路径 |
| `test_attn_decoding_sink_empty_row_matches_reference`（`406-438`） | decode empty sequence | packed varlen 空样本 |
| `test_attn_decoding_sink_with_g_matches_reference`（`441-481`） | decode sink + gate scale | PaTH/FoX decode 相关语义 |
| `test_parallel_sink`（`tests/ops/test_attn.py:233-264`） | 更广义参数化 sink 回归 | 对原 attention 测试集补充 |

此外，我本地做了轻量验证：`python -m py_compile` 通过 `parallel.py/decoding.py/naive.py/test_attn_sink.py`；`pytest tests/ops/test_attn_sink.py::test_attn_sink_ref_matches_gpt_oss_eager --collect-only -q` 成功收集 2 个参数化用例。code-reviewer 子任务还在 `FLA_DISABLE_BACKEND_DISPATCH=1` 下跑过 sink 测试文件，报告 `14 passed`，以及 `test_parallel_sink` 的 `3 passed`。

## 9.2 未覆盖风险

| 风险点 | 当前是否有测试 | 可能后果 |
|---|---:|---|
| TileLang sink runtime | 未在当前环境运行 | 安装 TileLang 的用户可能走未被本地验证的实现 |
| 模型层初始化 / 保存 / resume | 无 | 用户以为 config 生效但 state dict 没有参数 |
| `V != K` | 不充分 | reference sink 分支可能失败，掩盖 kernel 真实边界 |
| bf16 / fp32 多 dtype | 不充分 | dtype-specific 数值误差未充分暴露 |
| 极大 / 极小 `sink_bias` | 不充分 | 概率饱和、梯度消失或数值稳定性风险 |
| invalid `HQ % H` | 无明确 sink 测试 | GQA 映射 `i_hq // G` 的隐式假设可能出错 |
| 非 contiguous `sink_bias` | 无 | `contiguous` decorator 会处理张量参数，但边界未专测 |
| decode backward | 无 | 若误用于训练 decode，梯度不可依赖 |
| 多机 / distributed | 无 | 但源码也没有 sink 分布式逻辑；应由上层框架覆盖 |

测试中没有 skip/xfail 在 `test_attn_sink.py` 内；`tests/ops/test_attn.py:233-235` 的 `test_parallel_sink` 有 shared memory 条件 skip，用来避免非 Hopper 下某些大维度共享内存不足。

## 9.3 本章小结

💡 小结

- 测试强项是 kernel/reference 语义：full、SWA、varlen、gating、decode、empty-row 都覆盖到了。
- 测试弱项是模型集成、TileLang runtime、dtype/shape 极端边界和保存加载。
- 当前没有示例或文档告诉用户如何在模型里注册 learnable `sink_bias`。
- 因此“op correctness”相对可信，“end-to-end GPT-OSS support”未被测试证明。

# 十、局限性与已知优化点

## 10.1 硬约束

- `sink_bias.shape == [HQ]`，不支持 batch-specific 或 token-specific sink。
- `cu_seqlens` 模式要求 `B=1` packed 输入（`fla/ops/attn/parallel.py:826-830`）。
- `parallel_attn_fwd` 要求 `K` 的 tile 不超过 256，否则断言 `NK == 1`（`fla/ops/attn/parallel.py:539-545`）。
- 当前没有 K/V sink token；`sink_tokens_*` 只是未来保留名。
- decode sink 没有 autograd backward。

## 10.2 维护成本

- 双 backend 维护：Triton 与 TileLang 都实现 sink forward/backward 后处理，任何 LSE 语义变化都要同步。
- dispatch 隐式路径：用户安装 TileLang 后可能自动切 backend；需要 CI 保证两边一致。
- assert 校验不够生产化：`assert` 被优化掉后，错误 shape 可能导致低层 kernel 问题。
- reference 的 `V != K` 风险会影响未来测试可信度。
- 上层 wrapper 未透传，容易造成“底层支持但模型不用”的认知落差。

## 10.3 性能瓶颈

- forward 额外成本小，但每个 query row 都要合入 sink denominator。
- backward 的 `dsink_bias` 是一个 `[B,T,HQ]` 上的 reduction；一般不大，但没有融合进 Triton 主 kernel。
- GQA backward 仍会分配按 HQ 展开的 `dk/dv` 临时再 reduce（`fla/ops/attn/parallel.py:639-705`），这和 sink 无关，但仍是 attention backward 的显存/带宽点。
- TileLang runtime 未验证时，性能收益或退化不能只从 Triton 路径推断。

## 10.4 已知优化点

源码里最明确的未来方向是 `sink_tokens_*`：`parallel_attn` docstring 明确说未来会支持 Xiao 2024-style K/V sink tokens，并且可与 `sink_bias` 组合（`fla/ops/attn/parallel.py:812-814`）。除此之外，基于当前源码行为，可以提出几类工程优化：

1. **模型层集成**：为需要 GPT-OSS-style sink 的 layer 增加 `nn.Parameter(torch.zeros(num_heads))`，配置、初始化、state_dict、load/save 一起补齐。
2. **显式校验**：把 `assert sink_bias.shape` 改为 `ValueError`，并验证 device、dtype、`HQ % H`。
3. **测试扩展**：增加 `V != K`、bf16/fp32、极端 sink、TileLang CI。
4. **wrapper 透传**：让 `parallel_forgetting_attn`、PaTH/FoX decode 等复用路径可选传入 `sink_bias`。
5. **后处理融合评估**：如果 `dsink_bias` reduction 成为瓶颈，可考虑融合到 backward kernel 或提供专门 reduction kernel。

## 10.5 本章小结

💡 小结

- 当前实现的硬约束来自“per-head denominator bias”这个定位，而不是完整 sink token 机制。
- 维护风险集中在双 backend、隐式 dispatch、assert 校验和上层未集成。
- 性能瓶颈不在 sink scalar 本身，而在 attention backward 原有临时 buffer 与可能的 TileLang 差异。
- 最值得继续补的是模型集成和测试边界，而不是再扩展低层 API 名称。

# 小结与展望

FLA 的 GPT-OSS-style attention sink support 可以用几个关键词概括。

**关键词一：低层算子能力。**  
它不是模型配置、不是 Trainer 功能、不是保存加载机制，而是 `parallel_attn` / `attn_decoding_one_step` 接收 `sink_bias=[HQ]` 后改变 kernel softmax 归一化的能力。适合已经能直接调用 FLA attention op、并愿意自己管理 sink 参数的研究/实验代码。

**关键词二：denominator-only no-op logit。**  
源码没有实现 K/V sink token。sink 只把 `exp(sink_bias[h])` 加进 softmax denominator，不向 `b_o` 或 value matmul 贡献任何内容。这让实现非常轻，但也限定了表达形式。

**关键词三：LSE + delta 梯度补偿。**  
训练主路径没有大改 backward kernel，而是用 forward 保存的 `lse` 与 backward preprocess 得到的 `delta` 计算 `dsink_bias`。这是一个简洁的工程复用点，也要求未来维护者不要破坏 LSE “包含 sink mass” 的约定。

**关键词四：dispatch 下的双实现。**  
Triton 是默认源码主实现，TileLang 通过 backend registry 可选接管。这个设计对用户透明，但对测试提出更高要求：装了 TileLang 的用户可能跑的是另一套 sink kernel。

**关键词五：通信无关，集成未完成。**  
没有 process group、rank mapping、all-gather、save/load、model config，也没有 monkey patch。它适合单卡或每 rank 本地 attention kernel 的能力扩展；不适合被宣传成完整 GPT-OSS 模型适配。

与替代方案相比，FLA 没有选择 eager 拼接 logits，也没有引入额外 K/V sink token，而是在 fused attention 的在线 softmax 中做最小侵入修改。这带来了很小的额外计算和清晰的 backward 公式，但代价是：上层模型需要自己拥有参数，TileLang parity 需要持续验证，未来如果要支持真正的 K/V sink tokens，还要再设计 cache、shape、mask 和 backward 语义。

后续最值得继续走读的方向有三个：第一，`parallel_attn` 的 TileLang backend 如何与 Triton 路径保持数值一致；第二，FLA dense `Attention` 为什么仍走 flash-attn，而不是统一到 `parallel_attn`；第三，如果未来实现 `sink_tokens_*`，它将如何穿过 cache、varlen、sliding window 与 GQA 映射。
