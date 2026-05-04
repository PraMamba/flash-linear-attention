# FLA 源码走读：Kimi Delta Attention (KDA) 实现解析

在 `flash-linear-attention`（下文简称 FLA）里，Kimi Delta Attention 很容易被误读成“又一个 attention kernel”。但顺着源码看，它更像一个被包装成 HuggingFace CausalLM 的递归状态机：用户看到的是 `AutoModelForCausalLM`，模型内部却把 `hidden_states` 拆成 `q/k/v/g/beta`，再用 chunk/WY 表示把 per-dimension gate 的 delta-rule recurrence 做成可训练、可反向、可解码的算子路径。

本文不展开 Kimi Linear 论文的完整数学推导，也不讲 HuggingFace、Triton、DDP/FSDP 的通用原理；本文只回答一个工程问题：FLA 的 KDA 到底如何从配置进入模型、如何执行 forward/backward、哪些状态真的被保存、哪里有通信、哪里有显存收益，哪些路径看起来相关但并不是标准主流程。

# 前言

## 业务 / 工程背景

README 对 FLA 的定位是“efficient sequence modeling”，并强调实现基于 PyTorch 与 Triton（`README.md:9`）。KDA 在模型表中作为 2025 年的 `Kimi Linear: An Expressive, Efficient Attention Architecture` 实现出现（`README.md:101`），新闻区还记录了两个时间点：2025-10 添加 KDA 实现，2026-03 为 KDA/GDN 添加 Context Parallel 支持（`README.md:36-37`）。

这说明 KDA 在仓库里同时有两层身份：

1. **模型层身份**：像普通语言模型一样通过 `KDAConfig`、`KDAForCausalLM`、`AutoModelForCausalLM` 使用；
2. **算子层身份**：作为 delta-rule recurrent model 的高性能 `chunk_kda` / `fused_recurrent_kda` op，支持 varlen、反向、可选 CP 和 backend dispatch。

## 核心矛盾

KDA 的核心矛盾可以概括为三句话：

- 普通 softmax attention 的 KV cache 直观但长序列训练成本高；delta-rule recurrence 用矩阵状态替代显式 attention 矩阵，却引入了跨 token 的状态依赖。
- KDA 相比 GDN 把遗忘门从“每 head 一个标量”扩展到“每 key 维一个向量”，表达力更强，但 per-dim `Diag(alpha)` 如果直接塞进每个 kernel，会带来实现和性能压力。
- FLA 的选择是：模型入口保持 HuggingFace 风格，训练主路径走 chunk/WY + Triton，自回归短步走 fused recurrent，分布式长序列能力下沉到 op 级 CP，而不是默认塞进模型层。

## 本文主线

本文按机制展开，而不是按文件清单展开：

1. KDA 如何成为一个 HuggingFace CausalLM；
2. `KimiDeltaAttention` 如何把 `hidden_states` 拆成 `q/k/v/g/beta`；
3. `chunk_kda` 如何把 per-dim gate 变成可并行训练的 chunk recurrence；
4. 一次完整用户调用如何串起初始化、forward、cache、loss；
5. shape、state、rank、通信和显存如何变化；
6. backend dispatch、CP、FlashKDA、TileLang、save/load 哪些是主路径，哪些不是；
7. 测试证明了什么，还有哪些风险未被保护。

## 不展开的内容

本文不讲 Kimi Linear 论文推导，不讲 Triton kernel 编写教程，不讲 HF `PreTrainedModel` 通用保存/加载机制，不讲 DDP/FSDP/DeviceMesh 基础。源码中没有确认的设计意图，本文会写成“未在源码中确认”或“基于源码行为的推断”。

## 核心文件表

| 文件 | 职责 |
|---|---|
| `fla/models/kda/configuration_kda.py` | KDA 配置入口，定义 `model_type='kda'`、门控/卷积/融合 loss 等字段 |
| `fla/models/kda/__init__.py` | HuggingFace AutoClass 注册入口 |
| `fla/models/kda/modeling_kda.py` | `KDAForCausalLM -> KDAModel -> KDABlock` 模型壳与初始化逻辑 |
| `fla/layers/kda.py` | `KimiDeltaAttention` 层，生成 `q/k/v/g/beta` 并 dispatch 到 KDA op |
| `fla/ops/kda/chunk.py` | `chunk_kda` 用户级 op、autograd Function、shape/CP 参数约束 |
| `fla/ops/kda/chunk_fwd.py` | chunk forward 主编排：gate cumsum、WY、状态扫描、输出 |
| `fla/ops/kda/chunk_bwd.py` | chunk backward 主编排：重算、反向状态、梯度回传 |
| `fla/ops/kda/fused_recurrent.py` | 短序列 / decode 的 fused recurrent forward kernel |
| `fla/ops/cp/context.py` 与 `fla/ops/cp/chunk_delta_h.py` | op 级 Context Parallel context、跨 rank state pre-process / merge |
| `tests/ops/test_kda.py`、`tests/models/test_modeling_kda.py`、`tests/context_parallel/test_cp_kda.py` | op、模型、CP 证据链 |

💡 小结

- KDA 的用户入口是 HuggingFace 模型，不是裸 op。
- KDA 的核心复杂度在 per-dim gate + delta recurrent state 的工程化。
- README 中的 CP 支持需要区分 op 级能力和模型层默认能力。

# 一、入口与模型外壳：把一个算子接成 HuggingFace CausalLM

## 1.1 设计哲学与核心问题

KDA 首先要解决的不是 kernel 性能，而是“用户如何像加载普通 CausalLM 一样使用它”。如果没有模型壳，`chunk_kda` 再快也只是一个需要手动准备 `q/k/v/g/beta/A_log/dt_bias` 的函数；用户无法自然使用 `AutoModelForCausalLM.from_config()`、`generate()`、`labels` loss、`save_pretrained()` 等生态路径。

因此 FLA 的第一层设计是：把 KDA 包成标准 `PretrainedConfig + PreTrainedModel`。这层解决的是**入口兼容和模型生命周期问题**，不是分布式、通信或显存优化问题。

## 1.2 源码入口与关键对象

```text
fla/models/kda/configuration_kda.py
  - KDAConfig：定义 model_type、attn_mode、head_dim、num_heads、safe_gate 等配置

fla/models/kda/__init__.py
  - AutoConfig.register / AutoModel.register / AutoModelForCausalLM.register：接入 HF AutoClass

fla/models/kda/modeling_kda.py
  - KDAForCausalLM：语言模型入口，接 lm_head 和 loss
  - KDAModel：embedding、block 列表、final norm
  - KDABlock：在 KimiDeltaAttention 与普通 Attention 之间分流
  - KDAPreTrainedModel._init_weights：初始化 A_log / dt_bias / Linear / Conv / Embedding
```

`KDAConfig` 的 `model_type='kda'` 是 AutoClass 能识别它的基础（`fla/models/kda/configuration_kda.py:11-13`）。默认配置中，`attn_mode='chunk'`、`use_short_conv=True`、`use_cache=True`、`fuse_cross_entropy=True`、`safe_gate=False` 等字段都在构造时写入 config 对象（`configuration_kda.py:15-45`, `configuration_kda.py:48-74`）。

注册路径非常短，但很关键：`AutoConfig.register`、`AutoModel.register`、`AutoModelForCausalLM.register` 位于 `fla/models/kda/__init__.py:13-15`。这不是 monkey patch，而是 HuggingFace AutoClass 的标准扩展入口。

## 1.3 主流程拆解

用户最小路径可以写成：

```python
from transformers import AutoModelForCausalLM
from fla.models import KDAConfig

config = KDAConfig(hidden_size=H * D, num_heads=H, head_dim=D)
model = AutoModelForCausalLM.from_config(config)
```

测试中的模型创建也验证了这一路径：`create_model_and_config()` 构造 config 后调用 `AutoModelForCausalLM.from_config(config)`（`tests/models/test_modeling_utils.py:37-50`）。

真实构造链路是：

```text
User / test
  -> KDAConfig(...)
  -> AutoModelForCausalLM.from_config(config)
    -> KDAForCausalLM(config)
      -> KDAModel(config)
        -> [KDABlock(config, layer_idx) for layer_idx in num_hidden_layers]
          -> KimiDeltaAttention(...) 或 Attention(...)
```

在 `KDABlock.__init__` 中，源码首先构造 `attn_norm`。随后判断 `config.attn is not None and layer_idx in config.attn['layers']`：如果命中，就构造普通 `Attention`；否则构造 `KimiDeltaAttention`（`fla/models/kda/modeling_kda.py:48-75`）。这意味着 KDA 模型可以是 hybrid：不是每层都必须是 KDA。

`KDAModel.__init__` 创建 embedding、`ModuleList` blocks、final norm，并调用 `post_init()`（`modeling_kda.py:186-198`）。`KDAForCausalLM.__init__` 再包一层 `lm_head` 和 `criterion`（`modeling_kda.py:273-284`）。forward 中，`KDAForCausalLM` 先调用 `self.model(...)`，再根据训练和 labels 情况选择普通 `lm_head + CE` 或 fused linear cross entropy（`modeling_kda.py:338-372`）。

这里的第一个行为改变函数，不是某个 shell CLI，也不是 monkey patch，而是 `KDABlock.__init__` 中选择 `KimiDeltaAttention` 的分支（`modeling_kda.py:49-75`）。它决定当前层到底进入 KDA recurrent path，还是普通 softmax attention path。

## 1.4 初始化细节与误区澄清

`KDAPreTrainedModel._init_weights` 对 KDA 有一段特殊初始化：如果模块是 `KimiDeltaAttention` 且不在 meta device，它会初始化 `A_log` 和 `dt_bias`。`safe_gate=True` 时 `A_log` 置零；否则 `A_log` 从 `[1,16]` 均匀采样后取 log；`dt_bias` 使用类似 log-inverse-softplus 的时间常数初始化，并标记 `_is_hf_initialized`（`modeling_kda.py:134-147`）。普通 Linear/Conv/Embedding 则走标准 normal/zero 初始化（`modeling_kda.py:148-157`）。

这里有几个容易误解的点：

**误区一：Auto 注册就是 monkey patch。**  
不是。KDA 只是调用 HF registry（`fla/models/kda/__init__.py:13-15`），没有替换 transformers 模块命名空间，也没有提供恢复 patch 的 context manager。它是全局注册，但不是“把某个上游函数改掉”。

**误区二：KDA 模型每一层一定都是 KDA。**  
不一定。`config.attn` 可以指定若干层走普通 `Attention`（`modeling_kda.py:49-60`），`KDAConfig` 还会给 `attn` 补 `num_kv_heads/qkv_bias/window_size/rope_theta` 默认值（`configuration_kda.py:78-88`）。因此“模型类型是 KDA”和“所有层都是 KDA op”不是同一个结论。

**误区三：保存/加载有 KDA 专属 state_dict patch。**  
当前源码中未看到 KDA 覆写 `save_pretrained`、`from_pretrained`、`state_dict` 或 `load_state_dict`。`KDAPreTrainedModel` 继承 `PreTrainedModel`（`modeling_kda.py:118-124`），`KDAForCausalLM` 只声明 `_tied_weights_keys = ['lm_head.weight']`（`modeling_kda.py:273-280`）。因此权重保存加载主要走 HF 默认机制；运行时 recurrent cache/conv cache 不属于模型权重。

## 1.5 本章小结

💡 小结

- KDA 入口层解决的是 HuggingFace 生态兼容，不是 kernel 计算。
- 真正把模型导向 KDA 主路径的是 `KDABlock` 的层级分流。
- AutoClass 注册不是 monkey patch；KDA 也没有专属 save/load patch。
- Hybrid `attn.layers` 会让部分层绕开 KDA，进入普通 Attention。

# 二、KimiDeltaAttention 层：把 hidden_states 拆成 q/k/v/g/beta 的状态机

## 2.1 设计哲学与核心问题

KDA 层要解决的问题是：模型 block 只知道 `hidden_states: [B, T, hidden_size]`，而 `chunk_kda` 需要的是 `q/k/v/g/beta/A_log/dt_bias` 和可选 recurrent state。中间必须有一层把 Transformer 风格 hidden states 转成 delta-rule recurrent op 的输入。

这层同时处理五类问题：

- **数据问题**：如何从 hidden states 得到 `q/k/v/g/beta`；
- **shape 问题**：`H` 和 `HV` 分别是什么，GVA 如何表达；
- **状态问题**：generation 时如何读写 recurrent state 和 conv state；
- **调度问题**：训练走 chunk，短步推理走 fused recurrent；
- **mask 问题**：二维 padding mask 如何变成 varlen `cu_seqlens`。

## 2.2 源码入口与关键对象

```text
fla/layers/kda.py
  - KimiDeltaAttention.__init__：定义 projections、short conv、gate 参数和 output gate
  - KimiDeltaAttention.forward：训练 / 推理主调度，调用 chunk_kda 或 fused_recurrent_kda

fla/layers/utils.py
  - get_unpad_data / pad_input：padding mask 与 varlen flatten / restore
  - get_layer_cache / update_layer_cache：按 layer_idx 读写 Cache

fla/models/utils.py
  - FLALayer / FLACache / LegacyFLACache：保存 recurrent_state、conv_state 等运行时状态
```

`KimiDeltaAttention` 的 docstring 已经把 KDA 与 GDN 的关键差异写出来：KDA 的 forget gate `g` 形状为 `[B, T, H, K]`，是 per-key-dim gating；GDN 是 `[B, T, H]` scalar per-head gate（`fla/layers/kda.py:28-38`）。

## 2.3 主流程拆解

`KimiDeltaAttention.forward` 的主流程可以压缩成：

```text
hidden_states: [B, T, hidden_size]
  -> 若有 attention_mask：unpad 到 [1, total_nnz, hidden_size]，生成 cu_seqlens
  -> q_proj/k_proj/v_proj + optional ShortConvolution
  -> f_proj 得到 raw gate g，b_proj 得到 beta
  -> reshape:
       q,k: [B or 1, T or nnz, H, K]
       v:   [B or 1, T or nnz, HV, V]
       g:   [B or 1, T or nnz, HV, K]
       beta:[B or 1, T or nnz, HV]
  -> chunk_kda 或 fused_recurrent_kda
  -> update_layer_cache(recurrent_state, conv_state)
  -> o_norm + g_proj output gate + o_proj
  -> 若 unpad 过：pad 回 [B, T, hidden_size]
```

### mode 调度

源码先取 `batch_size, q_len`，然后做 mode 选择：

```python
mode = "fused_recurrent" if (q_len <= 64 and not self.training) else self.mode
if self.training:
    assert mode == "chunk", "Only chunk mode is supported in training."
```

对应 `fla/layers/kda.py:210-215`。这说明 `config.attn_mode='chunk'` 不是推理短步的绝对承诺；只要 eval 且 `q_len<=64`，layer 会强制切到 `fused_recurrent`。

### mask 到 varlen

如果传入 `attention_mask`，KDA 只接受二维 padding mask；任意 `[B,T,T]` attention mask 会被 assert 拒绝（`fla/layers/kda.py:203-208`）。随后它对最后 `q_len` 段 mask 调 `get_unpad_data()`，把 `[B,T,...]` flatten 成 `[1,total_nnz,...]`（`fla/layers/kda.py:218-222`）。`get_unpad_data()` 返回 `indices/cu_seqlens/max_seqlen`，其实现位于 `fla/layers/utils.py:79-103`。输出完成后再 `pad_input` 回原 batch 形状（`fla/layers/kda.py:309-310`）。

### q/k/v/g/beta 生成

`use_short_conv=True` 时，`q/k/v` 都是 Linear 后过独立 `ShortConvolution`（`fla/layers/kda.py:223-244`）。这不是普通 projection 后的轻量激活，而是会在 generation cache 中保留三路 conv state 的局部时序模块。

如果关闭 short conv，则直接使用 `F.silu(self.q_proj(...))` 等路径（`fla/layers/kda.py:245-248`）。

门控部分有三条线：

- `f_proj(hidden_states)` 生成 raw per-dim gate，`f_proj` 是两层低秩线性：`hidden_size -> head_v_dim -> num_v_heads * head_k_dim`（`fla/layers/kda.py:166-171`, `fla/layers/kda.py:250`）；
- `b_proj(hidden_states).sigmoid()` 生成 `beta: [B,T,HV]`（`fla/layers/kda.py:172`, `fla/layers/kda.py:251`）；
- `g_proj(hidden_states)` 在 op 输出后作为 output gate 传给 `FusedRMSNormGated`（`fla/layers/kda.py:187-191`, `fla/layers/kda.py:306-308`）。

reshape 后，`q/k` 是 query/key head 维度，`v/g/beta` 是 value-head 维度：

```text
q, k: [B, T, H,  K]
v:    [B, T, HV, V]
g:    [B, T, HV, K]
beta: [B, T, HV]
```

源码对应 `fla/layers/kda.py:253-257`。这里的 `HV` 不是 `num_kv_heads`，而是 `num_v_heads`。

### op 调用

chunk 训练路径调用：

```python
o, recurrent_state = chunk_kda(
    q=q, k=k, v=v, g=g, beta=beta,
    A_log=self.A_log, dt_bias=self.dt_bias,
    initial_state=recurrent_state,
    output_final_state=use_cache,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    safe_gate=self.safe_gate,
    lower_bound=self.lower_bound,
    cu_seqlens=cu_seqlens,
)
```

源码位于 `fla/layers/kda.py:263-278`。短步推理路径调用 `fused_recurrent_kda`，参数类似但没有 `safe_gate` 参数，只传 `lower_bound`（`fla/layers/kda.py:279-294`）。

### cache 写回

KDA 不写传统 KV cache。它写的是：

```python
update_layer_cache(
    self,
    past_key_values,
    recurrent_state=recurrent_state,
    conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
    offset=q_len,
)
```

对应 `fla/layers/kda.py:298-304`。`FLALayer.update()` 内部状态字典包含 `recurrent_state/attn_state/conv_state/ffn_state`（`fla/models/utils.py:60-69`），没有 `attn_state` 时用 `offset` 累加 seen tokens（`fla/models/utils.py:121-127`）。

## 2.4 关键细节与误区澄清

**误区一：KDA cache 是 KV cache。**  
不是。普通 `Attention` 会把 `k/v` 作为 `attn_state` 写 cache；KDA 写 `recurrent_state` 和可选三路 `conv_state`（`fla/layers/kda.py:298-304`）。这意味着 decode 显存结构从“随历史长度增长的 KV 序列”变成“每层一个 fp32 矩阵状态 + short conv 小窗口”。

**误区二：`num_kv_heads` 会影响 KDA 主层。**  
`num_kv_heads` 是 hybrid 普通 Attention 分支的配置补齐字段（`configuration_kda.py:85`）。KDA 主层使用的是 `num_v_heads`，并由 `self.num_v_heads`、`self.value_dim`、`self.gate_dim` 决定 value/gate 维度（`fla/layers/kda.py:116-121`, `fla/layers/kda.py:166-172`）。

**误区三：KDA 支持任意 mask。**  
不支持。源码只接受二维 padding mask（`fla/layers/kda.py:203-208`），再转换成 varlen。复杂稀疏 mask、中间洞、任意 attention bias 未在 KDA 层确认支持。

**误区四：`attn_mode='chunk'` 就表示 decode 用 chunk。**  
短步 eval 会强制 `fused_recurrent`（`fla/layers/kda.py:210-214`）。这是性能/调度上的特殊路径，读调用链时不能只看 config。

## 2.5 本章小结

💡 小结

- `KimiDeltaAttention` 是模型 hidden states 与 KDA op 合约之间的翻译层。
- KDA 的 gate 是 `HV × K` 的 per-dim gate，不是普通 per-head 标量。
- 训练默认 chunk；eval 短步默认 fused recurrent。
- KDA cache 保存 recurrent matrix state 和 conv state，不保存完整 KV 序列。

# 三、chunk_kda 训练主路径：用 WY 表示把 per-dim gate 做成可并行递推

## 3.1 设计哲学与核心问题

KDA 的数学核心是 delta-rule recurrence。最朴素写法是逐 token 更新状态矩阵 `S`：前一个 token 的状态影响下一个 token。这样做适合 decode，但训练长序列时串行依赖会成为吞吐瓶颈。

`chunk_kda` 解决的是**训练时把递归状态变成 chunk 内并行 + chunk 间扫描**的问题。更具体地说：

- per-dim gate 先被转换成 chunk 内累计 decay；
- chunk 内用 WY 表示生成 `w/u/qg/kg/Aqk/Akk`；
- 复用 common gated-delta state kernel 做跨 chunk recurrent state；
- backward 默认重算 `w/u/qg/kg/h/v_new`，用计算换显存。

## 3.2 源码入口与关键对象

```text
fla/ops/kda/chunk.py
  - chunk_kda：shape/CP/safe_gate 校验，调用 ChunkKDAFunction.apply
  - ChunkKDAFunction.forward：L2 norm、beta sigmoid、chunk_indices、保存 backward tensors
  - ChunkKDAFunction.backward：调用 chunk_kda_bwd 并处理 l2norm/beta sigmoid 梯度

fla/ops/kda/chunk_fwd.py
  - chunk_kda_fwd：gate cumsum、WY、CP preprocess、state scan、output

fla/ops/kda/chunk_intra.py
  - chunk_kda_fwd_intra：计算 Aqk/Akk、w/u/qg/kg

fla/ops/kda/wy_fast.py
  - recompute_w_u_fwd：forward/backward 复用的 WY 重算

fla/ops/common/chunk_delta_h.py
  - chunk_gated_delta_rule_fwd_h：共享 delta-rule state scan kernel
```

`chunk_kda` 是用户级 op，带 `@dispatch('kda')` 和 `@torch.compiler.disable`（`fla/ops/kda/chunk.py:160-162`）。dispatch 不是 patch；它是 wrapper 内部按 backend verifier 动态选择实现。

## 3.3 主流程拆解

### 入口合约

`chunk_kda` 文档明确约束：

```text
q/k: [B, T, H, K]
v:   [B, T, HV, V]
g:   [B, T, HV, K]
beta:[B, T, HV]
initial_state: [N, HV, K, V]
o: [B, T, HV, V]
final_state: [N, HV, K, V]
```

见 `fla/ops/kda/chunk.py:184-270`。实际代码还会检查：varlen 时 batch 必须是 1（`chunk.py:341-346`），`initial_state` 必须 float32（`chunk.py:352-353`），`K <= 256`（`chunk.py:367-370`），`HV % H == 0`（`chunk.py:370-373`），`g/beta` shape 与推导一致（`chunk.py:374-375`）。

### Autograd forward

`ChunkKDAFunction.forward` 内部固定 `chunk_size=64`（`fla/ops/kda/chunk.py:50`）。如果 `use_qk_l2norm_in_kernel=True`，它先调用 `l2norm_fwd` 处理 `q/k`（`chunk.py:52-56`）。如果 `use_beta_sigmoid_in_kernel=True`，会对 raw beta 做 fused sigmoid（`chunk.py:58-60`）。KDA layer 主路径传了 q/k L2 norm 和 gate in kernel，但没有传 beta sigmoid in kernel；因为 layer 已经在 `b_proj(...).sigmoid()` 得到 post-sigmoid beta。

随后构造 `chunk_indices`（varlen 时），调用 `chunk_kda_fwd`（`chunk.py:62-88`），并保存大量 backward tensor：`q/k/v/g_cumsum/g_input/beta/A_log/dt_bias/Aqk/Akk/w/u/qg/kg/v_new/h/initial_state/cu_seqlens/chunk_indices`（`chunk.py:95-99`）。这就是 chunk path 显存压力的重要来源之一。

### Forward 编排

`chunk_kda_fwd` 的顺序非常清晰：

1. **gate activation + chunk cumsum**：如果 `use_gate_in_kernel=True`，调用 `kda_gate_chunk_cumsum(g_org, A_log, dt_bias, scale=RCP_LN2, lower_bound=...)`；否则对已处理 `g` 做 `chunk_local_cumsum`（`fla/ops/kda/chunk_fwd.py:43-64`）。
2. **intra chunk / WY 表示**：`chunk_kda_fwd_intra` 生成 `w/u/qg/kg/Aqk/Akk`（`chunk_fwd.py:66-79`）。
3. **可选 CP preprocess**：如果传入 `cp_context`，用 `kg/w/u/g` 计算来自前序 rank 的 initial state（`chunk_fwd.py:81-92`）。
4. **跨 chunk state scan**：调用共享 `chunk_gated_delta_rule_fwd_h(k=kg,w=w,u=u,gk=g,...)` 得到 `h/v_new/final_state`（`chunk_fwd.py:94-106`）。
5. **可选 CP save_for_backward 压缩**：CP 下用 `compress_h0` 只保留可能跨 rank 延续的第一条 state（`chunk_fwd.py:108-113`）。
6. **输出计算**：`chunk_gla_fwd_o_gk(q=q, v=v_new, g=g, A=Aqk, h=h, ...)` 得到 `o`（`chunk_fwd.py:115-127`）。
7. **显存回收**：默认 `disable_recompute=False` 时，`w/u/qg/kg/v_new` 被置空，非 intermediate 模式下 `h` 也置空；gate-in-kernel 时 `g` 也置空（`chunk_fwd.py:128-134`）。

这解释了一个关键取舍：默认路径不保存所有中间量，backward 再重算，降低训练激活显存；`disable_recompute=True` 则保留更多中间量，可能更快但更吃显存。

### Backward 编排

`chunk_kda_bwd` 默认先重算 gate cumsum 和 WY 表示（`fla/ops/kda/chunk_bwd.py:448-469`），再重算 `h/v_new`（`chunk_bwd.py:470-485`）。如果 `disable_recompute=True`，直接取 forward 保存的 `w/u/qg/kg/v_new/h`（`chunk_bwd.py:486-492`）。

后续分四段：

1. `chunk_kda_bwd_dAv` 计算 `dAqk/dv`（`chunk_bwd.py:493-505`）；
2. CP 下先做 backward pre-process 得到本 rank 的 `dht`（`chunk_bwd.py:507-524`）；
3. `chunk_gated_delta_rule_bwd_dhu` 做 recurrent state 的反向（`chunk_bwd.py:526-540`）；
4. `chunk_kda_bwd_wy_dqkg_fused` 和 `chunk_kda_bwd_intra` 回传 `dq/dk/dv/db/dg/dAkk` 等（`chunk_bwd.py:542-576`）。

GVA 情况下，`dq/dk` 会从 `[B,T,HV,K]` reduce 回 `[B,T,H,K]`：`dq.view(...).sum(dim=3)`（`chunk_bwd.py:578-581`）。最后，`dg` 经过 reverse chunk cumsum；如果 gate in kernel，还要调用 `kda_gate_bwd` 得到 `dA/dbias`（`chunk_bwd.py:583-599`）。

## 3.4 关键细节与误区澄清

**误区一：`fused_kda_gate()` 是 KDA layer 主流程。**  
不是。KDA layer 调 `chunk_kda(... use_gate_in_kernel=True ...)`（`fla/layers/kda.py:263-278`），chunk forward 实际调用的是 `kda_gate_chunk_cumsum`（`fla/ops/kda/chunk_fwd.py:43-56`）。`fused_kda_gate()` 是单独 gate autograd helper，测试覆盖它（`tests/ops/test_kda.py:1003-1051`），但它不是标准 layer forward 的直接调用点。

**误区二：`chunk_size` 是 config 可调。**  
当前不是。`ChunkKDAFunction.forward` 内部硬编码 `chunk_size=64`（`fla/ops/kda/chunk.py:50`），`KDAConfig` 没有 chunk_size 字段。调参时不要以为改 config 就能改变 chunk 大小。

**误区三：`safe_gate=True` 自动保证输入安全。**  
如果通过 `KDAConfig`，至少会要求 `lower_bound` 非空（`configuration_kda.py:75-76`）；op 层在 `safe_gate and use_gate_in_kernel` 时要求 lower_bound 在 `[-5,0)`（`chunk.py:360-364`）。但如果直接调 op 且 `safe_gate=True, use_gate_in_kernel=False`，源码不会验证传入的 `g` 是否已经 clamp 到安全区间。benchmark 中 `chunk_kda` 配置了 `safe_gate=True, lower_bound=-5`，但输入 `g` 是 `logsigmoid` transform，未显式 `use_gate_in_kernel=True`（`benchmarks/ops/registry.py:289-298`）。这是一处需要谨慎解读的基准路径。

**误区四：`fused_recurrent_kda` 可以直接训练。**  
`fused_recurrent.py` 只有 forward kernel/wrapper；模型层训练时明确 assert 只支持 `chunk`（`fla/layers/kda.py:213-214`）。因此 fused recurrent 是 decode / inference 调度，不是训练 backward 主路径。

## 3.5 本章小结

💡 小结

- `chunk_kda` 用 chunk/WY 表示把串行 recurrence 变成训练可用路径。
- 默认 backward 用重算换显存；`disable_recompute=True` 会保留更多中间量。
- per-dim gate 的关键不是单独 gate 函数，而是 gate cumsum + 预门控 `kg/qg`。
- `K<=256`、`HV%H==0`、varlen batch=1 是硬约束。

# 四、完整主路径串联：一次用户调用到底发生了什么

## 4.1 完整调用栈

一次最常见的训练调用可以写成：

```text
User:
  config = KDAConfig(...)
  model = AutoModelForCausalLM.from_config(config)
  out = model(input_ids, attention_mask=..., labels=...)

  │
  ├─ Step 1: 配置与注册
  │     ├─ KDAConfig(model_type='kda')
  │     └─ AutoModelForCausalLM.register(KDAConfig, KDAForCausalLM)
  │
  ├─ Step 2: 模型初始化
  │     ├─ KDAForCausalLM
  │     ├─ KDAModel: embeddings + KDABlock[] + final norm
  │     └─ KDABlock: KimiDeltaAttention 或普通 Attention
  │
  ├─ Step 3: 前向进入 block
  │     ├─ embeddings(input_ids) -> hidden_states
  │     └─ for layer in layers: layer(hidden_states, ...)
  │
  ├─ Step 4: KimiDeltaAttention
  │     ├─ optional unpad -> cu_seqlens
  │     ├─ q/k/v short conv or silu projection
  │     ├─ f_proj -> g, b_proj -> beta
  │     ├─ chunk_kda 或 fused_recurrent_kda
  │     └─ update_layer_cache(recurrent_state, conv_state)
  │
  ├─ Step 5: CausalLM head/loss
  │     ├─ training + labels + fuse_cross_entropy -> FusedLinearCrossEntropyLoss
  │     └─ otherwise -> lm_head + CE
  │
  └─ Step 6: 保存/加载
        └─ 未见 KDA 专属 override，走 HF PreTrainedModel 默认机制
```

## 4.2 每一层做了什么

### 配置层

输入：用户传入的 config 字段。  
输出：`KDAConfig` 对象。  
状态变化：保存字段；如果 `attn` 非空，补齐普通 Attention 分支字段；如果 `safe_gate=True` 但 `lower_bound is None`，直接报错（`configuration_kda.py:75-88`）。  
通信：无。  
显存：无运行时显存影响，但字段会改变后续模块规模。

### 初始化层

输入：`KDAConfig`。  
输出：`KDAForCausalLM`。  
状态变化：创建参数，包括 `A_log/dt_bias/q_proj/k_proj/v_proj/f_proj/b_proj/g_proj/o_proj`；`post_init()` 触发 `_init_weights`。  
通信：无。  
显存：参数显存由 `hidden_size`、`num_heads`、`num_v_heads`、`expand_v` 决定。

### 前向层

输入：`input_ids` 或 `inputs_embeds`、可选 `attention_mask/past_key_values/labels`。  
输出：`CausalLMOutputWithPast`，包含 loss/logits/past_key_values。  
状态变化：eval/use_cache 时更新 `past_key_values`；训练默认 `use_cache=False`（`modeling_kda.py:221-224`）。  
通信：标准模型路径无 KDA 内置分布式通信。  
显存：训练主路径激活和 op scratch 是主要压力；`fuse_cross_entropy` 可避免显式保留全量 logits。

### op 层

输入：`q/k/v/g/beta/A_log/dt_bias`，可选 `initial_state/cu_seqlens/cp_context`。  
输出：`o` 与可选 `final_state`。  
状态变化：无 Python 全局状态；autograd context 保存 tensors；CP 时 compress initial_state。  
通信：仅当传入 `cp_context` 时发生 all-gather + merge。  
显存：保存或重算 WY、state、gate 等中间量。

## 4.3 哪些逻辑不在主路径

| 看似相关的逻辑 | 是否主路径 | 正确理解 |
|---|---:|---|
| `fused_kda_gate()` | 否 | 单独 gate helper；layer 主路径用 `kda_gate_chunk_cumsum` |
| FlashKDA backend | 否（默认模型主路径通常不命中） | 仅 inference、bf16、K/V=128、无 GVA、safe_gate、transpose state 等严格条件（`flashkda.py:64-88`） |
| TileLang KDA backend | 条件路径 | 仅替换 `chunk_kda_bwd_wy_dqkg_fused`，且 verifier 主要拒绝 GVA（`tilelang/__init__.py:31-55`） |
| Context Parallel | op 级路径，不是模型默认路径 | `chunk_kda` 接 `cp_context`，但 `KimiDeltaAttention.forward` 没传（`fla/layers/kda.py:263-278`） |
| benchmark 里的 layer patch | 否 | `benchmarks/cp/test_gdn_with_cp.py` 中会给 `cp_layer.forward` 赋 partial（约 `benchmarks/cp/test_gdn_with_cp.py:793-821`），服务 benchmark，不是生产模型代码 |
| KDA 专属 save/load patch | 否 | 未见 override；依赖 HF 默认 `PreTrainedModel` |
| 任意 attention mask | 否 | 只支持二维 padding mask（`fla/layers/kda.py:203-208`） |

## 4.4 本章小结

💡 小结

- KDA 的主路径是模型壳 + layer 翻译 + chunk op，不是裸 kernel 直接跑完一切。
- CP、FlashKDA、TileLang 都是条件路径，不能默认算入普通模型 forward。
- 保存/加载不是 KDA 特性复杂点；运行态 cache 不等于权重 checkpoint。

# 五、关键数据流 / 状态流 / shape 流程

## 5.1 Tensor shape 变化

以非 hybrid、`use_short_conv=True`、训练 chunk path 为例：

```text
原始输入:
  input_ids:      [B, T]
  hidden_states:  [B, T, hidden_size]

若有 padding mask:
  attention_mask: [B, T]
  indices:        [total_nnz]
  cu_seqlens:     [B+1]
  hidden_states:  [1, total_nnz, hidden_size]

投影 / short conv 后:
  q:    [B/1, T/nnz, H,  K]
  k:    [B/1, T/nnz, H,  K]
  v:    [B/1, T/nnz, HV, V]
  g:    [B/1, T/nnz, HV, K]
  beta: [B/1, T/nnz, HV]

chunk_kda 内部:
  g_cumsum: [B/1, T/nnz, HV, K]
  Aqk:      [B/1, T/nnz, HV, 64]
  Akk:      [B/1, T/nnz, HV, 64]
  w:        [B/1, T/nnz, HV, K]
  u:        [B/1, T/nnz, HV, V]
  h:        [B or N, NT, HV, K, V]  # return_intermediate_states 或内部中间状态
  final:    [N, HV, K, V]

输出:
  o before norm/proj: [B/1, T/nnz, HV, V]
  o after o_proj:     [B/1, T/nnz, hidden_size]
  若 unpad 过:
  output:             [B, T, hidden_size]
```

`Aqk/Akk/Akkd` 的实际分配可以在 `chunk_kda_fwd_intra` 看到：`Aqk=[B,T,HV,BT]`，`Akk=[B,T,HV,BT]`，`Akkd=[B,T,HV,BC]` 且 `Akkd` 为 fp32（`fla/ops/kda/chunk_intra.py:772-776`）。这也是训练显存的重要来源。

为什么要这样变换？因为 KDA 不生成 `[B,H,T,T]` attention score，而是在每个 chunk 内生成 WY 表示，把 token 间依赖压到 `K×V` 状态矩阵和 chunk 局部矩阵中。节省的是显式 attention 矩阵和 decode KV 序列缓存；但代价是引入 `Aqk/Akk/w/u/kg/qg/h` 等中间 buffer。

## 5.2 Rank / Mesh / Process Group 变化

普通 KDA 模型 forward 中，没有 `DeviceMesh`、`ProcessGroup`、`all_to_all`、`reduce_scatter` 或模型层 rank mapping。分布式相关逻辑在 op 级 CP 中。

CP context 的核心对象是 `FLACPContext`：它记录 `group/cu_seqlens/cu_seqlens_cpu/is_first_rank/is_last_rank/pre_num_ranks/post_num_ranks/conv1d_kernel_size/pre_num_conv_tokens`（`fla/ops/cp/context.py:22-33`）。`is_cp_enabled` 判断 `group is not None`（`context.py:54-58`）。

假设：

```text
world_size = 4
T = 10240
part_len = 2560
rank0: tokens [0, 2560)
rank1: tokens [2560, 5120)
rank2: tokens [5120, 7680)
rank3: tokens [7680, 10240)
```

`get_cp_cu_seqlens` 会根据全局 `cu_seqlens` 和 rank 范围切出本 rank 相关的 local `cu_seqlens`，并计算当前 rank 是否是某条 sequence 的 first/last rank，以及前面/后面需要串联多少 rank（`fla/ops/cp/context.py:80-152`）。

CP forward 的通信不是 all-to-all，而是：

```text
每个 rank 先本地计算 hm = [S_ext, M]
  -> all_gather_into_tensor(hm)
  -> 非 first rank 调 merge kernel 链接前序 rank 的 M/S_ext
  -> 得到本 rank 的 initial_state
  -> 执行本地 chunk_gated_delta_rule_fwd_h
```

`all_gather_into_tensor` 的实现是 `torch.distributed.all_gather_into_tensor`（`fla/ops/cp/comm.py:19-41`）。KDA forward preprocess 在 `chunk_gated_delta_rule_fwd_h_pre_process` 中调用 all-gather（`fla/ops/cp/chunk_delta_h.py:797-880`，尤其 `:859`），backward preprocess 对 `dhm` 也 all-gather（`chunk_delta_h.py:883-973`，尤其 `:948`）。

注意：`tests/context_parallel/test_cp_kda.py` 为了构造一致输入，会先从 rank0 broadcast 全局 q/k/v/g/beta/A_log/dt_bias（`tests/context_parallel/test_cp_kda.py:137-201`），又在验证时 all_gather 各 rank 输出/梯度（`test_cp_kda.py:349-372`）。这些是测试脚手架通信，不是 KDA op 的生产 forward 通信。

## 5.3 状态切换

KDA 里有三种值得区分的状态：

### 1. 运行时 Cache 状态

进入 generation：

```text
past_key_values=None
  -> KDAModel 如果 use_cache=True 且不是 Cache，会 Cache.from_legacy_cache
  -> 每层 KimiDeltaAttention get_layer_cache
  -> op 输出 recurrent_state
  -> update_layer_cache 写回 recurrent_state / conv_state
```

`KDAModel.forward` 训练时默认 `use_cache=False`，eval 才按 config 使用 cache（`fla/models/kda/modeling_kda.py:221-224`）。`Cache.from_legacy_cache` 在 `modeling_kda.py:236-237` 被调用。实际每层读写由 `get_layer_cache/update_layer_cache` 完成（`fla/layers/utils.py:212-223`）。

线程/进程安全方面，cache 是 Python 对象，属于当前模型调用的运行态，不是全局 registry；但它会被 generation 循环反复传入传出。

### 2. Backend dispatch 状态

`chunk_kda` 由 `@dispatch('kda')` 包装。dispatch 第一次调用会懒加载 `fla.ops.kda.backends`（`fla/ops/backends/__init__.py:123-135`），KDA backend 注册 FlashKDA 与 TileLang（`fla/ops/kda/backends/__init__.py:14-16`）。每次调用时，wrapper 遍历 backend，检查 `can_use()` 和 verifier，命中则调用 backend，否则回落默认实现（`fla/ops/backends/__init__.py:151-192`）。

这不是 monkey patch：没有替换原函数命名空间，只是被 decorator 包住后的调用期选择。

### 3. CP Context 状态

`build_cp_context(cu_seqlens, group, ...)` 返回 `FLACPContext`（`fla/ops/cp/context.py:155-172`）。它不是 thread-local/global context manager，而是显式传给 `chunk_kda(cp_context=...)` 的对象。谁写入？用户或外部训练框架。谁读取？`chunk_kda` 和 CP preprocess。

由于 `KimiDeltaAttention.forward` 当前没有接收/透传 `cp_context` 到 `chunk_kda`，模型层默认不会读这个状态。

## 5.4 本章小结

💡 小结

- KDA 的主 shape 从 `[B,T,hidden]` 变成 `q/k/v/g/beta`，再进入 `[N,HV,K,V]` recurrent state。
- 训练节省显式 attention 矩阵，但新增 WY/state/gate 中间 buffer。
- CP 的真实通信是 all-gather + merge，不是模型层自动 DeviceMesh 切换。
- Cache、backend dispatch、CP context 是三种不同状态，不能混为一谈。

# 六、核心机制深挖：dispatch、CP 与 cache 的职责边界

## 6.1 Backend Dispatch：零侵入 backend 选择，不是 monkey patch

### 设计哲学与核心问题

KDA 有默认 Triton path，也有可选 FlashKDA CUTLASS forward、TileLang backward。直接在业务代码里写多层 if 会污染主路径；FLA 用通用 backend registry 把选择逻辑包到 decorator 内。

### 源码入口与关键对象

```text
fla/ops/backends/__init__.py
  - BaseBackend：availability/env/verifier 抽象
  - BackendRegistry：注册和 lazy init
  - dispatch：调用期选择 backend

fla/ops/kda/backends/__init__.py
  - 注册 FlashKDABackend / KDATileLangBackend
```

`BaseBackend.is_enabled()` 每次读环境变量，默认值由 backend 定义（`fla/ops/backends/__init__.py:56-60`）。`dispatch` wrapper 会 lazy import backend 模块，再按优先级和 verifier 选择实现（`backends/__init__.py:151-192`）。

### 误区澄清

**误区：FlashKDA 会自动接管 KDA 模型推理。**  
不一定。FlashKDA verifier 要求非常窄：inference mode、bf16、K=128、V=128、无 GVA、`use_gate_in_kernel=True`、`use_qk_l2norm_in_kernel=True`、`use_beta_sigmoid_in_kernel=True`、`transpose_state_layout=True`、无 CP、`safe_gate=True`（`fla/ops/kda/backends/flashkda.py:64-88`）。而 `KimiDeltaAttention.forward` 调 chunk path 时没有传 `use_beta_sigmoid_in_kernel=True`，也没有传 `transpose_state_layout=True`（`fla/layers/kda.py:263-278`）。因此普通模型主路径通常不会命中 FlashKDA。

TileLang 也不是完整替换 KDA forward。它只实现 `chunk_kda_bwd_wy_dqkg_fused`，并且 verifier 基本只拒绝 GVA（`fla/ops/kda/backends/tilelang/__init__.py:31-85`）。安装了 tilelang 的环境可能走不同 backward path，但测试中未见 KDA TileLang 专门覆盖。

## 6.2 Context Parallel：op 级序列并行，而不是模型层默认分布式

### 设计哲学与核心问题

KDA 的状态依赖沿序列方向传播：rank1 的第一段 token 需要 rank0 处理完前一段后的状态。如果简单把序列切给不同 rank，各 rank 初始 state 都当零，结果必然错。

CP 解决的问题是：每个 rank 只计算本地 token chunk，但通过一个可合并的状态转移摘要，让 rank 能恢复“前面所有 rank 对 state 的贡献”。

### 源码实现

CP 文档直接写明，KDA 的 per-dim gate 会在 WY 阶段预门控 `kg/qg`，CP preprocess 和主 kernel 必须接收同样的预门控张量（`fla/ops/cp/README.md:154-166`, `README.md:288-348`）。源码也对应这一点：

```python
# KDA forward
w, u, qg, kg, Aqk, Akk = chunk_kda_fwd_intra(...)
initial_state = chunk_gated_delta_rule_fwd_h_pre_process(
    k=kg, w=w, u=u, gk=g, context=cp_context, use_exp2=True
)
h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
    k=kg, w=w, u=u, gk=g, initial_state=initial_state, use_exp2=True
)
```

源码路径是 `fla/ops/kda/chunk_fwd.py:66-106`。forward preprocess 内部：本地构造 `hm`，all-gather 所有 rank 的 `hm`，非 first rank 通过 merge kernel 链接前序 rank（`fla/ops/cp/chunk_delta_h.py:797-880`）。backward preprocess 类似，只是方向相反，all-gather `dhm` 后给非 last rank 合并后序 rank 的梯度状态（`chunk_delta_h.py:883-973`）。

### 隐藏假设与副作用

- `chunk_kda` CP 模式不支持用户传 `initial_state`，也不支持 `output_final_state=True`（`fla/ops/kda/chunk.py:332-340`）。因为初始 state 由跨 rank 预处理算出。
- CP 依赖 `cu_seqlens`，`build_cp_context` 会按 `world_size/rank` 切 local cu_seqlens（`fla/ops/cp/context.py:80-152`）。
- 通信是 all-gather，不是 reduce-scatter；每个 rank 收到所有 rank 的摘要，再本地 merge。
- CP 是 op 级参数。`KimiDeltaAttention.forward` 没传 `cp_context`，所以模型层默认不启用 CP。

### 误区澄清

**误区：README 说 KDA 支持 CP，所以 `KDAForCausalLM` 自动序列并行。**  
源码不能支持这个结论。README 新闻只说 KDA/GDN 添加 CP（`README.md:36`），CP README 的 Quick Start 也是直接调用 `chunk_kda(... cp_context=cp_context)`（`fla/ops/cp/README.md:46-65`）。模型层 `KimiDeltaAttention.forward` 调 `chunk_kda` 时没有 `cp_context` 参数（`fla/layers/kda.py:263-278`）。因此“op 支持 CP”和“模型自动 CP”必须区分。

## 6.3 Cache：用矩阵状态替代 KV 序列，但不是免费午餐

### 设计哲学与核心问题

普通 attention 的 decode cache 是历史 `k/v` 序列，长度随上下文增长。KDA 的 recurrence 可以把历史压进固定形状矩阵状态 `[N,HV,K,V]`；这对超长 decode 很有吸引力。但这个 state 是 fp32，且每层每 batch 都存在，并不等于“cache 显存为零”。

### 源码实现

`fused_recurrent_kda_fwd` 若 `output_final_state=True`，会分配 `final_state`：普通 layout 为 `[N,HV,K,V]` fp32，transpose layout 为 `[N,HV,V,K]` fp32（`fla/ops/kda/fused_recurrent.py:265-274`）。`chunk_kda` 文档也要求 `initial_state` 是 `[N,HV,K,V]`，并在代码中 assert dtype 为 fp32（`fla/ops/kda/chunk.py:203-208`, `chunk.py:352-353`）。

模型层在 eval/use_cache 时把这个 state 写入 cache，同时 short conv 会写三路 conv state（`fla/layers/kda.py:298-304`）。

### 误区澄清

**误区：KDA decode cache 一定比 KV cache 小。**  
它不随历史长度增长，这是优势；但单层 state 大小是 `N * HV * K * V * fp32`。当 `H/HV`、`K/V`、batch 或层数较大时，这个固定状态本身也不小。它解决的是“随 T 线性增长”的问题，不是消灭 cache 显存。

## 6.4 本章小结

💡 小结

- Backend dispatch 是调用期选择，不是全局 monkey patch。
- CP 是 op 级 all-gather + merge，不是模型层自动序列并行。
- KDA cache 是 fp32 recurrent matrix state，加上可选 short conv state。
- FlashKDA/TileLang 是条件 backend；不能把它们当默认性能路径。

# 七、显存、性能与通信分析

## 7.1 显存收益范围

| 内容 | 是否节省 | 原因 |
|---|---:|---|
| 参数 | ❌ | KDA 有 q/k/v/f/b/g/o projection、A_log/dt_bias、MLP；参数不因 KDA 自动 sharding |
| 训练 attention 矩阵 | ✅ | 不构造 `[B,H,T,T]` softmax attention score，改用 chunk/WY/state |
| 训练激活 | 部分 ✅ / 部分 ❌ | 默认 backward 重算减少保存；但仍有 Aqk/Akk/g/state/WY scratch |
| logits | ✅（训练 labels 且 fuse） | `fuse_cross_entropy and training and labels` 时可用 `FusedLinearCrossEntropyLoss` 避免显式 logits（`modeling_kda.py:350-370`） |
| optimizer state | ❌ | 源码未做 KDA 专属 optimizer sharding |
| 输入 batch | ❌ | mask unpad 只减少 padding token 计算，不改变 batch 分发 |
| decode KV cache | ✅ | 不保存历史 K/V 序列；保存 recurrent state |
| decode recurrent state | ❌ | `[N,HV,K,V]` fp32 固定状态本身是显存成本 |
| CP local token 激活 | ✅（op 级） | rank 只处理本地序列切片 |
| CP 通信 buffer | ❌ | all-gather `hm/dhm` 和 merge 需要额外 buffer |

源码确认的显存大头包括：

- forward intra 分配 `Aqk/Akk/Akkd`（`fla/ops/kda/chunk_intra.py:772-776`）；
- WY 重算/保存涉及 `w/u/qg/kg`（`fla/ops/kda/wy_fast.py:268-271`）；
- backward `db2=[NK,*beta.shape]` float32（`fla/ops/kda/chunk_intra.py:877-880`）；
- autograd 保存 tensors 较多（`fla/ops/kda/chunk.py:95-99`）；
- recurrent cache state fp32（`fla/ops/kda/fused_recurrent.py:270-272`）。

`disable_recompute=False` 默认会在 forward 末尾删除 `w/u/qg/kg/v_new/h/g` 等中间量以省显存（`fla/ops/kda/chunk_fwd.py:128-134`）。如果 `disable_recompute=True`，它们会保留给 backward，速度可能更好但显存峰值更高。

## 7.2 通信开销

普通模型 path：未见 KDA 内部通信 primitive。DDP/FSDP 等外部分布式通信不属于 KDA 源码特性。

op 级 CP path：每次 `chunk_kda` forward/backward 额外通信：

| 阶段 | 通信 | 源码 | group |
|---|---|---|---|
| CP forward preprocess | `all_gather_into_tensor(hm)` | `fla/ops/cp/chunk_delta_h.py:859` | `FLACPContext.group` |
| CP backward preprocess | `all_gather_into_tensor(dhm)` | `fla/ops/cp/chunk_delta_h.py:948` | `FLACPContext.group` |
| 测试数据准备 | `broadcast` q/k/v/g/beta/A_log/dt_bias | `tests/context_parallel/test_cp_kda.py:165-201` | 测试 `WORLD` |
| 测试验证 | `dist.all_gather` 输出/梯度 | `test_cp_kda.py:349-372` | 测试 `WORLD` |

`all_gather_into_tensor` 的底层实现就是 `dist.all_gather_into_tensor`（`fla/ops/cp/comm.py:19-41`）。没有在 KDA CP 主路径中看到 all-to-all、reduce-scatter 或 broadcast。文档和测试注释中会用“state passing / synchronization”描述，但实际源码通信原语要以 `all_gather_into_tensor` 为准。

通信是否每层发生？如果外部让每个 KDA layer 的 `chunk_kda` 都传入 `cp_context`，那么每个 KDA layer 的 forward/backward 都会有上述 CP preprocess 通信。但当前模型层没有透传 `cp_context`，所以普通 `KDAForCausalLM` 不会自动触发。

## 7.3 性能取舍

KDA 的主要取舍是：

- **用 chunk/WY + Triton 复杂度换训练吞吐**：避免逐 token 串行 recurrence；
- **用重算换显存**：默认 backward 重算 WY/state；
- **用 fp32 state 换数值稳定**：`initial_state/final_state` 要求 fp32；
- **用固定 recurrent cache 换随 T 增长的 KV cache**：适合长 decode，但固定 state 不小；
- **用 op 级 CP 通信换序列维显存/吞吐**：rank 本地处理序列切片，但增加 all-gather + merge；
- **用 backend dispatch 换可选性能路径**：FlashKDA/TileLang 可改善特定场景，但约束和测试覆盖更复杂。

源码中还有一个明确风险点：`chunk_kda_bwd_dAv` 的 tiling 判断里，`elif check_shared_mem:` 使用了函数对象本身而非调用结果（`fla/ops/kda/chunk_bwd.py:315-320`）。这会让该分支永真，可能影响非 Hopper / shared memory 条件下的 tiling 选择。本文不修复它，但在性能解读中必须把它作为维护风险。

## 7.4 本章小结

💡 小结

- KDA 节省的是显式 attention 矩阵和随 T 增长的 KV cache，不是所有显存。
- 训练中间 buffer 仍多，默认重算策略是重要显存优化。
- CP 通信是每次 op 的 all-gather + merge，不是无通信序列并行。
- 可选 backend 带来性能机会，也带来路径分裂和测试覆盖压力。

# 八、配置项、边界条件与坑点

## 8.1 配置如何改变源码路径

| 配置 / 参数 | 影响源码路径 | 行为变化 | 风险 / 坑点 |
|---|---|---|---|
| `attn_mode='chunk'` | `KimiDeltaAttention.forward` | 训练主路径要求 chunk | eval 且 `q_len<=64` 仍会强制 fused recurrent（`fla/layers/kda.py:210-215`） |
| `attn_mode='fused_recurrent'` | `KimiDeltaAttention.forward` | eval 可走 recurrent | training 会 assert 失败，只支持 chunk |
| `use_short_conv=True` | `KimiDeltaAttention.__init__/forward` | q/k/v 过三路 `ShortConvolution` | cache 多三路 conv state；CUDA backend 在部分组合会退回 Triton（`short_conv.py:173-185`） |
| `use_short_conv=False` | `KimiDeltaAttention.forward` | q/k/v 为 `silu(linear)` | 模型级测试未专门覆盖该配置 |
| `num_v_heads` | `KimiDeltaAttention.__init__` + `chunk_kda` | 决定 HV、value/gate/state 大小 | `num_v_heads < num_heads` 初始化不拒绝，但 op 要求 `HV % H == 0`（`chunk.py:370-375`） |
| `expand_v` | `head_v_dim/value_dim` | 改变 V 和输出维度 | 必须产生整数，否则 layer 初始化报错（`fla/layers/kda.py:124-139`） |
| `head_dim` | K/V head dim | 决定 K 和默认 scale | op 要求 `K<=256`，config/layer 未提前拒绝（`chunk.py:369`） |
| `safe_gate=True` | config + op | 使用 lower-bound gate / safe path | config 只要求 lower_bound 非空；op 仅在 gate-in-kernel 时校验范围（`configuration_kda.py:75-76`, `chunk.py:360-364`） |
| `lower_bound` | gate activation | 下界 clamp / sigmoid lower-bound gate | 推荐/测试常用 -5；越界只在部分路径报错 |
| `allow_neg_eigval=True` | `KimiDeltaAttention.forward` | `beta *= 2` | 改变 recurrence 谱性质，未见 KDA model-level 测试覆盖 |
| `attn={...}` | `KDABlock.__init__` | 指定层走普通 Attention | KDA 模型不再每层都是 KDA；`num_kv_heads` 属于此分支 |
| `fuse_cross_entropy=True` | `KDAForCausalLM.forward` | 训练 labels 下走 fused linear CE | 只影响 loss/logits 显存，不影响 KDA op |
| `use_l2warp=True` | loss 路径 | CE 后 l2 warp 或 fused CE 中启用 | 测试覆盖模型 forward/backward 两种值（`tests/models/test_modeling_kda.py:19-38`） |
| `cp_context`（op 参数） | `chunk_kda` | 开启 op 级 CP | 模型层不透传；不支持 initial_state/output_final_state |
| `FLA_FLASH_KDA` | backend dispatch | 允许 FlashKDA backend | 默认允许但需 package + 严格 verifier；普通模型 path 通常不命中 |
| `FLA_TILELANG` | backend dispatch | 允许 TileLang backend | 安装 tilelang 后可能替换 backward 子路径，缺少专门测试 |
| `FLA_INTRACARD_CP` | common backend | 推理 varlen 下 intra-card CP | 默认关闭；只在 inference mode + cu_seqlens 生效（`fla/ops/common/backends/intracard.py:59-67`） |

## 8.2 最小可用配置与默认行为

最小模型配置通常需要保持：

```python
KDAConfig(
    hidden_size=num_heads * head_dim,
    num_heads=num_heads,
    head_dim=head_dim,
)
```

如果 `num_v_heads=None`，layer 会令 `num_v_heads=num_heads`（`fla/layers/kda.py:116`）。默认 `expand_v=1.0`，所以 `value_dim=hidden_size`。默认 `attn_mode='chunk'`，训练可用；默认 `use_short_conv=True`；默认 `safe_gate=False`，不需要 `lower_bound`。

## 8.3 静默失效 / 不兼容组合

- `output_attentions=True`：`KDAModel` 会 warning 并设为 False（`fla/models/kda/modeling_kda.py:218-220`），`KimiDeltaAttention` 返回 attentions 为 `None`（`fla/layers/kda.py:312`）。
- `cp_context` 期望在模型层自动生效：不会，除非直接调用 op 或外部 patch layer。
- `FlashKDA` 期望通用推理加速：需要严格 verifier；否则 dispatch 会静默回退默认 Triton path。
- `safe_gate=True` 直接实例化 layer 且 `lower_bound=None`：`KimiDeltaAttention` 构造函数自身没有 `KDAConfig` 那样的检查（`fla/layers/kda.py:93-106`），错误可能延迟到 op。
- `python -O`：大量 shape/模式保护使用 assert，例如 `fla/layers/kda.py:140,203-214`、`fla/ops/kda/chunk.py:332-375`；优化模式会移除 assert，让错误更晚暴露。

## 8.4 保存 / 加载 / resume 差异

当前源码未确认 KDA 专属 checkpoint merge、state_dict remap、rank0-only loading 或 resume hook。权重保存/加载按 HF `PreTrainedModel` 默认路径；运行时 `recurrent_state`、`conv_state` 是 generation cache，不会作为模型权重自动保存。若外部训练框架做 FSDP/DDP/ZeRO checkpoint，那属于外部框架能力，不是 KDA 源码内建路径。

## 8.5 本章小结

💡 小结

- 很多配置不是“改数值”，而是直接改变源码路径：chunk/recurrent、KDA/Attention、short conv、fused CE、backend dispatch。
- KDA 的硬约束主要在 op 层，部分错误配置会延迟到 kernel 前才失败。
- CP、FlashKDA、TileLang 都需要显式条件，不能从配置名简单推断已启用。

# 九、测试、示例与覆盖缺口

## 9.1 已覆盖路径：测试证明了什么

### 模型级测试

`tests/models/test_modeling_kda.py` 覆盖两类模型行为：

- forward/backward：`L=4,B=4,T=1024,H=4,D=64`，bf16，`use_l2warp=True/False`（`tests/models/test_modeling_kda.py:19-38`）；
- generation：`L=2,B=4,T=2000,H=8,D=64`，fp16（`test_modeling_kda.py:44-61`）。

通用 base 测试中，fixed batch 输出 shape 被检查为 `(B,T,hidden_size)`，varlen `input_ids.view(1,B*T)` 与 fixed 输出近似一致并执行 backward（`tests/models/test_modeling_base.py:53-68`）。generation 测试比较无 cache 逐样本参考与 chunk prefill + 单 token decode 的 logits（`test_modeling_base.py:101-133`）。这证明了模型主路径、varlen flatten、cache decode 在常规配置下能对齐。

### op 级测试

`tests/ops/test_kda.py` 证据更细：

- naive chunk 对齐 naive recurrent（`tests/ops/test_kda.py:40-84`）；
- fused recurrent 对齐 naive recurrent，含 GVA 与 L2 norm in kernel（`test_kda.py:104-150`）；
- beta sigmoid in kernel（`test_kda.py:163-207`）；
- gate in kernel / safe gate / lower-bound gate（`test_kda.py:288-344`, `test_kda.py:1003-1051`）；
- chunk forward/backward 与 naive recurrent 对齐，检查 `o/ht/dq/dk/dv/dg/db/dh0/dA/dbias`（`test_kda.py:519-614`）；
- transpose state layout（`test_kda.py:632-693`）；
- varlen chunk 和 varlen prefill（`test_kda.py:789-982`）；
- `return_intermediate_states=True` 的 shape/dtype（`test_kda.py:1055-1125`）；
- FlashKDA optional backend（`test_kda.py:1128-1267`）。

### CP 测试

`tests/context_parallel/test_cp_kda.py` 明确说明用 `naive_recurrent_kda` 作数学基线（`test_cp_kda.py:13-25`），varlen 展平到 batch=1（`test_cp_kda.py:27-32`），CP 按序列维切 rank，forward/backward 需要跨 rank 状态/梯度（`test_cp_kda.py:34-54`）。实际 CP path 调 `build_cp_context` 和 `chunk_kda(... cp_context=context ...)`（`test_cp_kda.py:308-344`），再聚合输出/梯度比较（`test_cp_kda.py:349-396`）。场景包括 CP2/CP4/CP8、sequence cut、boundary aligned、single sequence、many short sequences、disable_recompute、transpose state（`test_cp_kda.py:448-587`）。

## 9.2 skip 与测试环境边界

- 基础模型测试在 Intel Alchemist 上跳过（`tests/models/test_modeling_base.py:28-31`）。
- 非 Hopper 且 `D==128` 会跳过以省 CI（`test_modeling_base.py:46-47`）。
- KDA 不在 `GENERATION_UNSUPPORTED` / `NOT_READY_FOR_TESTING` 中（`tests/models/test_modeling_utils.py:23-34`）。
- op 测试中 Intel Alchemist 对 D>128 有 skip（如 `tests/ops/test_kda.py:51-52`, `:116-117`, `:301-302`）。
- varlen 默认 cache config 下有 D=64/256 或 D=256 skip（`tests/ops/test_kda.py:799-800`, `:917-918`）。
- FlashKDA 测试需要 GPU 和 `flash_kda` package（`tests/ops/test_kda.py:1132-1136`）。
- CP 测试按 GPU 数量 skip，例如少于 2/4/8 GPU 时跳过（`tests/context_parallel/test_cp_kda.py:448-511`）。

## 9.3 未覆盖风险

| 风险点 | 当前是否有测试 | 可能后果 |
|---|---:|---|
| `num_v_heads < num_heads` | 未见 | 初始化能过，op 处 assert/崩溃 |
| `head_dim > 256` | 未见模型级非法配置测试 | kernel 前才失败 |
| `safe_gate=True` 但 `lower_bound` 越界 | 部分 op 校验 | 直接 layer 或非 gate-in-kernel 路径风险 |
| `safe_gate=True, use_gate_in_kernel=False` 输入未 clamp | 未见专门测试 | safe path 假设可能不成立 |
| `allow_neg_eigval=True` | 未见 KDA model-level 覆盖 | 改变 beta 范围和状态谱性质，生成/梯度风险 |
| `use_short_conv=False` model path | 未见专门覆盖 | 关闭 conv 的路径可能回归 |
| KDA model/layer CP | 未见 | op CP 正确不代表模型自动 CP 正确 |
| TileLang KDA backend | 未见专门测试 | 安装 tilelang 后 backward path 分裂 |
| save/load roundtrip | 未见 | config/state_dict 兼容风险无法由现有测试证明 |
| return_intermediate_states 数值 | 只测 shape/dtype | intermediate `h` 数值错误不一定被发现 |
| CP beta grad | warning-only | 测试注释说明 `db` 误差较高，可能掩盖边界误差（`test_cp_kda.py:378-395`） |

## 9.4 本章小结

💡 小结

- KDA op 的数学正确性测试较丰富，覆盖 forward/backward/varlen/GVA/gate。
- 模型级测试覆盖基本训练、varlen 和 generation，但配置组合较少。
- CP 测的是 op 级 CP，不是模型层 CP。
- 可选 backend、非法配置、save/load、极端 mask 是主要覆盖缺口。

# 十、局限性与已知优化点

## 10.1 硬约束

1. **`K <= 256`**：`chunk_kda` 明确 assert（`fla/ops/kda/chunk.py:369`）。
2. **GVA 约束 `HV % H == 0`**：op assert（`chunk.py:370-373`），layer 只检查 `num_v_heads > num_heads` 时能否整除（`fla/layers/kda.py:130-133`），对 `num_v_heads < num_heads` 缺少提前拒绝。
3. **varlen batch 必须为 1**：`chunk_kda` 要求 `q.shape[0] == 1`（`chunk.py:341-346`）。KDA layer 通过 unpad 把 masked batch flatten 成 `[1,total_nnz,...]`。
4. **训练只支持 chunk**：`KimiDeltaAttention.forward` 在 training 下 assert `mode == 'chunk'`（`fla/layers/kda.py:213-214`）。
5. **recurrent state 要 fp32**：`initial_state` dtype assert 为 float32（`fla/ops/kda/chunk.py:352-353`），fused recurrent final_state 也分配 fp32（`fla/ops/kda/fused_recurrent.py:270-272`）。
6. **mask 只支持二维 padding mask**：`fla/layers/kda.py:203-208`。
7. **FlashKDA 限制极窄**：见 `fla/ops/kda/backends/flashkda.py:64-88`。

## 10.2 维护成本

- **路径分裂**：chunk、fused recurrent、FlashKDA、TileLang、CP、intra-card CP 都可能影响不同场景。
- **assert 依赖**：不少输入校验是 assert，`python -O` 会移除。
- **backend 自动启用**：`FLA_TILELANG` 默认 unset 时可启用，安装 tilelang 会改变 backward 子路径；没有专门 KDA TileLang 测试。
- **精度状态不透明**：`A_log/dt_bias` 初始化为 fp32，但 `model.to(dtype)` 会把参数一起 cast。kernel 内 load 后再转 fp32，不能恢复源参数被 cast 的精度损失；当前测试未专门保证门控参数保持 fp32。
- **初始化标记不完全对称**：`dt_bias` 设置 `_is_hf_initialized`，`A_log` 未设置同样标记（`modeling_kda.py:134-147`）。这不必然是 bug，但重复初始化时维护者需要注意。

## 10.3 性能瓶颈

- **WY / intra chunk scratch**：`Aqk/Akk/Akkd` 与 `w/u/kg/qg` 都是大张量。
- **Backward 临时 buffer**：`db2=[NK,B,T,HV]` float32（`fla/ops/kda/chunk_intra.py:879`）。
- **CP all-gather**：每个 op forward/backward 都要 all-gather state summary；目前未见 overlap 逻辑。
- **fused recurrent 只有 forward**：decode 快，但不能训练。
- **FlashKDA 非通用**：只有严格 inference 场景可能命中。
- **tiling bug 风险**：`elif check_shared_mem:` 永真（`fla/ops/kda/chunk_bwd.py:315-320`）可能影响不同硬件下性能。

## 10.4 已知优化点与后续方向

源码和测试暗示的优化方向包括：

1. **模型层 CP 接入**：当前 CP 已在 op 层和测试层成立，但模型层不透传 `cp_context`。如果要支持完整 KDA 模型序列并行，需要设计 layer/model API、conv state CP、trainer 数据切分与 cache 语义。
2. **更早的 config validation**：把 `head_dim<=256`、`num_v_heads>=num_heads`、`HV%H==0`、safe_gate lower_bound 范围等提前到 config/layer 初始化，可减少 Triton 深处错误。
3. **backend 测试矩阵**：为 TileLang KDA backward、FlashKDA dispatch 命中/回退、intra-card CP 数值建立更稳定 CI 条件。
4. **显存策略可配置化**：`chunk_size=64` 当前硬编码；未来若开放 chunk size，需要同步 autotune、CP、safe_gate、varlen 测试。
5. **通信 overlap / 分块 all-gather**：CP 当前 all-gather 摘要后 merge；长层数、多 rank 时通信可能串行化。可以考虑异步 all-gather、分层 merge 或和计算 overlap。
6. **save/load roundtrip 测试**：即使 KDA 没有专属 patch，也应覆盖 `save_pretrained/from_pretrained`、`tie_word_embeddings`、dtype cast 后 gate 参数精度等生命周期问题。

## 10.5 本章小结

💡 小结

- KDA 的硬约束主要来自 op shape、dtype 和 mode。
- 最大维护成本来自多路径：chunk/recurrent/CP/backend/varlen/cache。
- 已知优化点不是单纯“写更快 kernel”，还包括模型层 CP API、配置校验、测试矩阵和 checkpoint 生命周期。

# 小结与展望

FLA 的 KDA 实现可以用几个关键词概括。

**关键词一：HuggingFace 外壳。**  
KDA 不是孤立 op，而是通过 `KDAConfig`、AutoClass 注册、`KDAForCausalLM` 接进 CausalLM 生命周期。它让用户入口熟悉，但也要求读源码时不要停在模型壳：真正改变计算行为的是 `KDABlock` 选择 `KimiDeltaAttention`。

**关键词二：per-dim gate 的状态机。**  
KDA 相比 GDN 的关键是 `g: [B,T,HV,K]`。这带来更细粒度遗忘能力，也迫使实现用 gate cumsum、预门控 `kg/qg`、WY 表示和 shared delta state kernel 来控制复杂度。

**关键词三：训练 chunk，推理 recurrent。**  
训练路径必须是 `chunk_kda`，短步 eval/generation 会自动切到 `fused_recurrent_kda`。这是一种很工程化的调度：训练要并行，decode 要低延迟和 cache 友好。

**关键词四：固定状态换长上下文。**  
KDA 不保存完整 KV 序列，而保存 `[N,HV,K,V]` fp32 recurrent state。这对长 decode 有优势，但并不意味着 cache 免费；head、value expansion、batch、层数都会放大固定状态成本。

**关键词五：op 级 CP 与条件 backend。**  
KDA 的 CP 是 `chunk_kda(cp_context=...)` 的 op 级能力，通过 all-gather + merge 同步跨 rank state；模型层默认没有自动 CP。FlashKDA、TileLang、intra-card CP 都是条件 backend，不应和标准主路径混淆。

这个实现适合的场景，是希望在 FLA/HF 生态中实验 Kimi Linear/KDA 风格模型，并需要长序列训练、varlen、generation cache 或 op 级 CP 的用户。它不适合被当作“无脑替换任意 attention”的黑盒：任意 mask、任意 head_dim、任意 backend、任意分布式 mesh 都不是当前源码默认保证的能力。

与普通 attention 相比，KDA 用 recurrent state 和 chunk kernel 换掉显式 attention 矩阵；与更简单的 DeltaNet/GDN 相比，KDA 用 per-dim gate 换表达力，也换来了更复杂的 gate/WY/CP 一致性要求；与外部分布式方案相比，KDA 的 CP 更贴近 recurrence 数学结构，但当前仍停留在 op 级，需要外部框架或 patch 才能成为完整模型训练路径。

后续最值得继续走读的方向有三个：第一，`fla/ops/cp` 如何把 GDN/KDA/DPLR 的 chunk transition 统一成 all-gather + merge；第二，FlashKDA / TileLang backend 在真实环境下如何命中 dispatch 与影响性能；第三，KDA 如果接入模型层 CP、checkpoint/resume 和 Trainer，会怎样设计 API 才能不污染普通 HuggingFace 用户路径。

💡 小结

- KDA 的设计主线是“表达力更强的 per-dim recurrence + 工程上可训练/可解码/可选分布式”。
- 最容易误读的是 CP、FlashKDA、cache 和 `attn_mode`：它们都有条件，不是默认万能路径。
- 源码实现已经覆盖了 op 数学正确性的主要路径，但模型级 CP、非法配置、可选 backend、save/load 仍是后续工程化重点。
