# Flash Linear Attention 源码走读：Gated DeltaNet 实现解析

在长序列建模里，Transformer 注意力的二次复杂度一直是显存与吞吐的主要矛盾。Flash Linear Attention（下文简称 FLA）把一组线性注意力、Delta Rule、RetNet、GLA、KDA、Gated DeltaNet 等模型统一放进 PyTorch/Triton 算子体系里，试图让“线性复杂度”不只是论文里的公式，而是可以训练、可以生成、可以和 HuggingFace 生态互操作的工程实现。

本文聚焦 `Gated DeltaNet`。它不是普通的 `nn.Module` 拼装：模型层里有短卷积、两类门控、GQA 式 value head、chunk 训练路径、fused recurrent 推理路径、Triton kernel、可变长输入、cache、以及算子级 Context Parallel（CP）能力。源码走读的主线不是“哪些文件存在”，而是：**Gated DeltaNet 如何在保持线性状态更新语义的同时，解决训练并行度、显存、变长序列和推理缓存之间的冲突。**

# 前言

## 业务 / 工程背景

FLA 的 README 把项目定位为“efficient sequence modeling”，并明确列出基于 PyTorch 与 Triton 的高性能实现（`README.md:9`）。Gated DeltaNet 在模型表中作为一种独立模型出现（`README.md:89`），同时 README 也记录了两个和本文关系很大的事实：其一，Gated DeltaNet 被加入项目（`README.md:58`）；其二，GDN 和 KDA 已支持 Context Parallel（`README.md:36`），并被集成进 Qwen3-Next（`README.md:39`）。

从场景上看，Gated DeltaNet 主要解决的是训练和推理中的序列建模效率问题：训练侧要能反向传播并控制激活显存；推理侧要能使用 recurrent state 做增量生成；长序列或多 GPU 场景下，还要处理序列切分后的状态边界。

## 核心矛盾

Gated DeltaNet 的工程矛盾可以压缩成三句话：

1. 线性注意力把全局注意力替换成递推状态，但训练时如果逐 token recurrent scan，会牺牲并行度。
2. chunk 化可以恢复并行训练，却需要额外保存或重算 WY 表示、块内矩阵和跨块状态。
3. Context Parallel 能把长序列切到多个 rank，但 Delta/Gated Delta 这类状态模型必须补齐切分边界处的历史状态，否则每段序列的递推初值就是错的。

## 本文主线

本文按机制展开：

- 第一章看配置、注册和模型装配：Gated DeltaNet 如何接入 HuggingFace 主路径。
- 第二章看 `GatedDeltaNet` 层：短卷积、投影、两类门控和 recurrent state 如何组织。
- 第三章看训练主路径：为什么标准训练走 chunk kernel，而不是 fused recurrent。
- 第四章串起一次真实调用：从用户 API 到 Triton kernel。
- 第五章专门梳理 shape、cache state、varlen 和 rank 维度。
- 第六章分析 Context Parallel：它是算子级能力，而不是模型层默认通信。
- 第七章讨论显存、通信和性能取舍。
- 第八至十章总结配置坑点、测试覆盖和局限优化。

## 不展开的内容

本文不讲 Gated DeltaNet 论文推导，不展开 Delta Rule 的数学证明，也不讲 Triton 语法细节。我们只看 FLA 源码如何把 Gated DeltaNet 变成一个可训练、可推理、可测试、可并行切分的工程特性。

## 核心文件表

| 文件 | 职责 |
|---|---|
| `fla/models/gated_deltanet/configuration_gated_deltanet.py` | HuggingFace 配置类，定义 `attn_mode`、head、gate、fuse 等行为开关 |
| `fla/models/gated_deltanet/__init__.py` | 注册 `AutoConfig` / `AutoModelForCausalLM` 入口 |
| `fla/models/gated_deltanet/modeling_gated_deltanet.py` | 模型、Block、LM head、初始化、forward 与 generation 适配 |
| `fla/layers/gated_deltanet.py` | Gated DeltaNet 核心层，完成投影、短卷积、mode 选择、cache 更新 |
| `fla/ops/gated_delta_rule/chunk.py` | chunk 训练主算子及自定义 autograd |
| `fla/ops/gated_delta_rule/fused_recurrent.py` | 短序列 / 推理 recurrent kernel，当前无 backward |
| `fla/ops/gated_delta_rule/gate.py` | GDN gate 的 cumsum、反向与 naive 参考实现 |
| `fla/ops/cp/context.py` | 构造 Context Parallel 的 rank-local 序列边界元数据 |
| `fla/ops/cp/chunk_delta_h.py` | CP 下跨 rank 初始状态补齐与反向梯度补齐 |
| `tests/ops/test_gated_delta.py` / `tests/context_parallel/test_cp_gdn.py` | 算子、varlen、GQA、gate-in-kernel、CP 对齐测试 |

# 一、从配置到模型装配：GDN 如何接入 Transformers 主链路

## 1.1 设计哲学与核心问题

Gated DeltaNet 首先要解决的不是 kernel 性能，而是“用户如何像加载普通 CausalLM 一样加载它”。如果这个问题没有解决，后面的 Triton 算子再快，也只能作为孤立函数存在，不能自然进入 `AutoModelForCausalLM.from_config()`、`generate()`、`save_pretrained()` 这类 HuggingFace 生态路径。

因此 FLA 对 GDN 的第一层设计是：**配置类负责描述模型形态，Block 负责在 hybrid attention 和 GDN layer 之间分流，PreTrainedModel 负责初始化、cache 和 LM loss 适配。** 这一层主要解决的是 API 兼容和模型装配问题，而不是通信或显存问题。

## 1.2 源码入口与关键对象

```text
fla/models/gated_deltanet/configuration_gated_deltanet.py
  - GatedDeltaNetConfig：定义 model_type、head 形状、gate、conv、loss fuse、hybrid attention 配置

fla/models/gated_deltanet/__init__.py
  - AutoConfig.register / AutoModelForCausalLM.register：把 GDN 接入 HF Auto 类

fla/models/gated_deltanet/modeling_gated_deltanet.py
  - GatedDeltaNetBlock：根据 config.attn 决定本层是 Attention 还是 GatedDeltaNet
  - GatedDeltaNetModel：embedding、layer stack、final norm 和 cache 传递
  - GatedDeltaNetForCausalLM：lm_head、loss、logits 策略与 generation guard
```

`GatedDeltaNetConfig` 在 `configuration_gated_deltanet.py:13-15` 定义，`model_type = 'gated_deltanet'`。配置字段包括 `attn_mode`、`hidden_size`、`expand_v`、`use_gate`、`use_short_conv`、`allow_neg_eigval`、`conv_size`、`head_dim`、`num_heads`、`num_v_heads`、`attn`、`fuse_norm`、`fuse_linear_cross_entropy` 等（`configuration_gated_deltanet.py:17-48`），随后写入实例属性（`configuration_gated_deltanet.py:50-76`）。

HF 注册发生在 `__init__.py:8-15`：配置类注册为 `gated_deltanet`，CausalLM 注册到 `AutoModelForCausalLM`。

## 1.3 主流程拆解

用户最自然的入口是：

```python
from transformers import AutoModelForCausalLM
from fla.models.gated_deltanet import GatedDeltaNetConfig

config = GatedDeltaNetConfig(...)
model = AutoModelForCausalLM.from_config(config)
```

真实调用链可以简化为：

```text
User
  -> AutoModelForCausalLM.from_config(config)
    -> GatedDeltaNetForCausalLM.__init__
      -> GatedDeltaNetModel.__init__
        -> GatedDeltaNetBlock.__init__ x num_hidden_layers
          -> Attention(...)       # 如果该层命中 config.attn['layers']
          -> GatedDeltaNet(...)   # 否则走 GDN layer
```

`GatedDeltaNetBlock.__init__` 是第一个真正改变行为的分叉点。源码在 `modeling_gated_deltanet.py:41-83`：

- 先创建 `RMSNorm`（`modeling_gated_deltanet.py:49`）。
- 如果 `config.attn` 存在并且当前 `layer_idx` 在 `attn['layers']` 中，就实例化标准 `Attention`（`modeling_gated_deltanet.py:50-60`）。
- 否则实例化 `GatedDeltaNet`（`modeling_gated_deltanet.py:62-75`）。
- 最后创建 MLP 和后置 norm（`modeling_gated_deltanet.py:76-83`）。

也就是说，GDN 模型不是强制每层都是 GDN；它允许 hybrid attention。配置类里还对 `attn` 做了默认与校验：如果指定了 `attn`，必须包含 `'layers'`，否则报错；同时补齐 `num_heads`、`num_kv_heads`、`window_size` 等默认值（`configuration_gated_deltanet.py:89-99`）。

Block forward 则是标准残差结构：

```text
hidden_states
  -> norm
  -> self.attn(...)   # Attention 或 GatedDeltaNet
  -> residual add
  -> MLP + residual
```

对应 `modeling_gated_deltanet.py:85-115`。这一层的输入是 `[B, T, hidden_size]`，输出仍是 `[B, T, hidden_size]`，状态副作用主要来自 `past_key_values` / `Cache` 的更新。

初始化也被单独处理。`GatedDeltaNetPreTrainedModel._init_weights` 在 `modeling_gated_deltanet.py:129-184`：

- 对 `A_log` 使用均匀初始化并取 log（`modeling_gated_deltanet.py:137-140`）。
- 对 `dt_bias` 用 inverse softplus 形式初始化（`modeling_gated_deltanet.py:141-145`）。
- 对 `_no_weight_decay` 参数打标（`modeling_gated_deltanet.py:146-148`）。
- 对线性层、卷积、embedding 做常规初始化（`modeling_gated_deltanet.py:149-158`）。
- 可选 residual scaling（`modeling_gated_deltanet.py:160-184`）。

这说明 `A_log` 和 `dt_bias` 不是普通权重：它们直接决定 recurrent gate 的时间尺度，初始化策略属于模型语义的一部分。

## 1.4 关键细节与误区澄清

这里有一个容易误解的点：README 里的 Gated DeltaNet 模型表链接指向 `fla/ops/gated_delta_rule`（`README.md:89`），容易让人以为用户入口就是算子。实际主入口是 HuggingFace 模型注册：`AutoConfig.register` 和 `AutoModelForCausalLM.register`（`fla/models/gated_deltanet/__init__.py:8-15`）。算子是核心实现，但不是普通用户最可能触发的第一层 API。

第二个误区是：`attn` 配置不是“给 GDN layer 增加一个 attention 子模块”，而是按层替换。如果当前层在 `config.attn['layers']` 中，`GatedDeltaNetBlock` 会创建 `Attention`；否则创建 `GatedDeltaNet`（`modeling_gated_deltanet.py:50-75`）。因此 hybrid attention 改变的是 layer 类型，而不是在同一层内同时执行两套注意力。

第三个误区是保存 / 加载。源码里没有 Gated DeltaNet 专属的 `state_dict`、`save_pretrained` 或 `from_pretrained` 改写；模型继承 `PreTrainedModel`，主要依赖 HuggingFace 默认保存加载路径。`GatedDeltaNetForCausalLM` 只定义了 `_tied_weights_keys = ['lm_head.weight']`（`modeling_gated_deltanet.py:277`），并没有实现自定义 checkpoint 合并逻辑。换句话说，GDN 的保存加载不是这个特性的复杂点，至少在当前源码中未看到专门适配。

## 1.5 本章小结

💡 小结

- GDN 的用户主入口是 HF Auto 类，算子不是普通用户的第一入口。
- `GatedDeltaNetBlock` 是模型行为分叉点：每层要么 Attention，要么 GatedDeltaNet。
- 初始化对 `A_log` / `dt_bias` 做了特殊处理，说明 recurrent gate 的时间尺度是模型语义的一部分。
- 保存加载基本走 HuggingFace 默认路径，源码中未确认 GDN 有专属 checkpoint 合并机制。

# 二、GatedDeltaNet 层：两套门控如何服务同一个状态更新问题

## 2.1 设计哲学与核心问题

进入 `fla/layers/gated_deltanet.py` 后，工程问题变得更具体：一个 `[B, T, hidden]` 的 token 表示，如何变成适合 Delta Rule recurrence 的 `q/k/v/g/beta`，并在输出时回到 `[B, T, hidden]`？

GDN layer 同时承担三类职责：

1. **投影职责**：从 hidden states 生成 `q/k/v`、decay gate `g`、delta update gate `beta`。
2. **局部建模职责**：可选短卷积给 `q/k/v` 增加局部上下文。
3. **状态职责**：在训练或生成中读写 recurrent state 与 conv state。

如果没有这一层，算子只能处理已经格式化好的 `q/k/v/g/beta`，无法自然接入 Transformer block，也无法和 generation cache 对齐。

## 2.2 源码入口与关键对象

```text
fla/layers/gated_deltanet.py
  - GatedDeltaNet.__init__：定义投影、gate 参数、短卷积、输出 gate/norm/o_proj
  - GatedDeltaNet.forward：选择 chunk/recurrent 路径，处理 varlen、cache、pad/unpad

fla/models/utils.py
  - Cache / FLACache / FLALayer：保存 recurrent_state、conv_state、seen_tokens 等状态

fla/layers/utils.py
  - get_unpad_data / pad_input：处理 attention_mask 下的变长输入
  - get_layer_cache / update_layer_cache：按 layer_idx 读写 cache
```

`GatedDeltaNet` 的文档字符串直接给出参数预算：`q/k` 各约 `0.75 * hidden_size^2`，`v/g/o` 各约 `1.5 * hidden_size^2`，总计约 `6 * hidden_size^2`（`gated_deltanet.py:35-48`）。这段注释很重要，因为它说明 GDN 不是单纯“更省参数”的模块；它的优势主要来自序列复杂度和状态形式，而不是把投影矩阵都缩小。

## 2.3 主流程拆解

构造函数先固化 head 形状：

- `key_dim = num_heads * head_dim`
- `value_dim = int(hidden_size * expand_v)`
- `num_v_heads = num_v_heads or num_heads`
- `head_v_dim = value_dim // num_v_heads`

这些逻辑在 `gated_deltanet.py:106-124`。随后源码做了几类校验：`expand_v` 必须能得到整数 value 维度（`gated_deltanet.py:126-131`）；如果 `num_v_heads > num_heads`，需要 `num_v_heads % num_heads == 0`（`gated_deltanet.py:132-135`）；`key_dim` 和 `value_dim` 要能被对应 head 数整除（`gated_deltanet.py:137-140`）；`mode` 只能是 `chunk` 或 `fused_recurrent`（`gated_deltanet.py:142`）。

核心投影在 `gated_deltanet.py:144-149`：

```text
hidden_states [B,T,D]
  -> q_proj: [B,T,H*K]
  -> k_proj: [B,T,H*K]
  -> v_proj: [B,T,HV*V]
  -> a_proj: [B,T,HV]      # recurrent decay preactivation
  -> b_proj: [B,T,HV]      # beta gate
```

随后初始化 recurrent gate 的两个参数：`A_log` 与 `dt_bias`（`gated_deltanet.py:150-167`）。这两个参数在后续 kernel 中把 `a_proj(hidden)` 变成负 decay：`naive_gdn_gate` 里公式是 `-A_log.exp() * softplus(g + dt_bias)`（`fla/ops/gated_delta_rule/gate.py:20-45`）。

短卷积分支在 `gated_deltanet.py:169-193`：如果 `use_short_conv=True`，`q/k/v` 会各自通过 `ShortConvolution`；否则源码会 warning，并保留线性投影路径（`gated_deltanet.py:189-193`）。

输出端还有另一套 gate：如果 `use_gate=True`，创建 `g_proj` 和 `FusedRMSNormGated`；否则使用普通 `RMSNorm`（`gated_deltanet.py:194-199`）。最后 `o_proj` 把 `value_dim` 投回 `hidden_size`（`gated_deltanet.py:199`）。

前向过程可以整理为：

```text
GatedDeltaNet.forward(hidden_states, attention_mask, past_key_values, ...)
  -> 根据 q_len/training 选择 mode
  -> 读取 cache 中的 recurrent_state / conv_state
  -> 如果 attention_mask 且无 cu_seqlens：unpad 到 [1,total_tokens,D]
  -> q/k/v 投影 + 短卷积
  -> reshape:
       q,k: [B,T,H,K]
       v:   [B,T,HV,V]
       beta:[B,T,HV]
  -> 调用 chunk_gated_delta_rule 或 fused_recurrent_gated_delta_rule
  -> 更新 recurrent_state / conv_state 到 cache
  -> output norm/gate + o_proj
  -> 如果之前 unpad，则 pad 回 [B,T,D]
```

源码里 mode 选择很直接：如果 `mode == 'fused_recurrent'` 或者 `q_len <= 64 and not self.training`，就走 fused recurrent；否则训练态必须是 `chunk`（`gated_deltanet.py:219-221`）。这也是后文区分训练主路径和推理短路径的关键证据。

cache 的状态写入发生在 `gated_deltanet.py:298-304`：只要 `past_key_values` 不为空，就把 `recurrent_state` 和可选 `conv_state` 写入当前层缓存，同时更新 `offset=q_len`。底层工具函数 `update_layer_cache` 在 `fla/layers/utils.py:219-223`，`FLALayer` 中 recurrent state、conv state、seen token 相关字段定义在 `fla/models/utils.py:60-127`。

## 2.4 关键细节与误区澄清

最容易误解的是 `use_gate`。从名字看，它像是在控制 Gated DeltaNet 是否“使用 gate”。但源码不是这样：`use_gate` 只控制输出端是否使用 `g_proj + FusedRMSNormGated`（`gated_deltanet.py:194-199`）。无论 `use_gate` 为真还是假，forward 调用 `chunk_gated_delta_rule` 时仍然会传入 `g=self.a_proj(hidden_states)`，并且默认 `use_gate_in_kernel=True`（`gated_deltanet.py:265-279`）。因此 recurrent decay gate 仍然存在。

第二个误区是 GQA 的约束位置。Layer 构造函数只在 `num_v_heads > num_heads` 时检查可整除（`gated_deltanet.py:132-135`），但 chunk op 会进一步要求 `q/k` head 数一致，并要求 `HV % H == 0`（`chunk.py:452-463`）。这意味着某些非法配置可能不是在构造 layer 时失败，而是在算子调用时失败。

第三个误区是 attention mask 与 `cu_seqlens`。`GatedDeltaNet.forward` 在没有 `cu_seqlens` 且提供 `attention_mask` 时，会自己 unpad 并生成 `indices/cu_seqlens`（`gated_deltanet.py:225-228`），最后用 `pad_input` 还原（`gated_deltanet.py:313-314`）。但如果调用者同时传入 `attention_mask` 和 `cu_seqlens`，源码中 `indices` 不会在该分支初始化，却仍可能走到 pad_input。这是基于源码行为发现的潜在坑点，测试中主路径通常是二选一，未在源码中看到显式保护。

## 2.5 本章小结

💡 小结

- `GatedDeltaNet` layer 是模型态和算子态之间的桥：它把 hidden states 变成 `q/k/v/g/beta`。
- 源码里有两套 gate：recurrent decay gate 由 `a_proj/A_log/dt_bias` 决定，输出 gate 由 `use_gate` 控制。
- 训练态默认走 chunk，短序列 eval 才会自动切到 fused recurrent。
- cache 更新发生在 layer forward 内，保存的是 recurrent state 和可选 conv state，而不是传统 KV cache。

# 三、Chunk 训练路径：为什么不能只用 recurrent scan

## 3.1 设计哲学与核心问题

线性注意力最直观的实现是 recurrent scan：一个 token 一个 token 更新状态。推理时这很自然，因为增量生成本来就是串行的；但训练时，如果每个 token 都依赖前一个 token 的状态，就会牺牲 GPU 并行度。

FLA 的 Gated DeltaNet 训练主路径采用 chunk 化：在块内构造 WY 表示，在块间扫描状态，再生成输出。这解决的是**训练并行度和递推依赖之间的矛盾**。代价是中间张量更多，反向传播也要重算部分表示。

## 3.2 源码入口与关键对象

```text
fla/ops/gated_delta_rule/chunk.py
  - chunk_gated_delta_rule：公开 API，做 shape/CP/varlen 校验并进入 autograd Function
  - chunk_gated_delta_rule_fwd：前向主流程，gate、WY、state scan、output
  - chunk_gated_delta_rule_bwd：反向主流程，重算 WY/state 并传播 dq/dk/dv/dg/dbeta
  - ChunkGatedDeltaRuleFunction：保存必要张量，桥接 PyTorch autograd

fla/ops/gated_delta_rule/chunk_fwd.py
  - chunk_gated_delta_rule_fwd_intra：块内 KKT + solve + recompute w/u

fla/ops/gated_delta_rule/wy_fast.py
  - recompute_w_u_fwd / prepare_wy_repr_bwd：WY 表示前后向辅助

fla/ops/common/chunk_delta_h.py
  - chunk_gated_delta_rule_fwd_h / chunk_gated_delta_rule_bwd_dhu：跨块 recurrent state scan

fla/ops/gated_delta_rule/gate.py
  - gdn_gate_chunk_cumsum / gdn_gate_bwd：GDN gate 的块内前后向
```

## 3.3 主流程拆解

Layer 进入 chunk op 的调用在 `gated_deltanet.py:265-279`：

```text
q,k:   [B,T,H,K]
v:     [B,T,HV,V]
g:     [B,T,HV]    # a_proj(hidden_states)
beta:  [B,T,HV]
state: [N,HV,K,V] or None
```

`chunk_gated_delta_rule` 的公开 API 文档也明确了这些 shape：`q/k` 是 `[B,T,H,K]`，`v` 是 `[B,T,HV,V]`，`g/beta` 是 `[B,T,HV]`，`initial_state/final_state` 是 `[N,HV,K,V]`（`chunk.py:371-421`）。

前向主流程在 `chunk_gated_delta_rule_fwd`（`chunk.py:29-119`）：

```text
1. gate 处理
   if use_gate_in_kernel:
       g, A = gdn_gate_chunk_cumsum(g, A_log, dt_bias)
   else:
       g, A = chunk_local_cumsum(g)

2. 块内 WY 表示
   w, u, A = chunk_gated_delta_rule_fwd_intra(q, k, v, beta, g, A)

3. CP 前处理（如果传入 cp_context）
   initial_state = chunk_gated_delta_rule_fwd_h_pre_process(...)

4. 跨块状态扫描
   h, v_new, final_state = chunk_gated_delta_rule_fwd_h(k, w, u, ...)

5. CP 保存优化
   initial_state = compress_h0(initial_state, cu_seqlens, cp_context)

6. 输出
   o = chunk_fwd_o(q, k, v_new, h, g, ...)
```

gate 这一步的关键是 `use_gate_in_kernel=True`。源码先把原始 `a_proj(hidden)` 交给 `gdn_gate_chunk_cumsum`（`chunk.py:47-57`），而不是要求 layer 预先算好 decay。`gdn_gate_chunk_cumsum` 内部会调用 Triton kernel，把 raw gate 加上 `dt_bias`、经过 softplus，再乘以 `-exp(A_log)` 并做块内累积（`gate.py:48-103`）。`naive_gdn_gate` 给了同样语义的参考公式（`gate.py:20-45`）。

WY 表示的前向在 `chunk_fwd.py:331-410`。注释写得很清楚：概念上要做 KKT、`solve_tril`、`recompute_w_u`，但实现把前两个 kernel 融合，减少从 3 个 kernel 到 2 个 kernel，并避免 `A` 的 HBM 往返（`chunk_fwd.py:341-350`）。这就是 FLA 这类库的典型工程取舍：不是只把公式翻成 PyTorch，而是围绕 HBM 访问和 kernel 数量重排计算。

跨块状态扫描在 common op 中。`chunk_gated_delta_rule_fwd_h` 由 backend dispatch 装饰（`fla/ops/common/chunk_delta_h.py:662-721`），内部要求 `K <= 256`（`chunk_delta_h.py:689`），分配 `h`、`final_state`、`v_new` 并发起 Triton kernel（`chunk_delta_h.py:691-720`）。这一步把块内表示串成跨 chunk 的 recurrent state。

反向主流程在 `chunk_gated_delta_rule_bwd`（`chunk.py:122-248`）。它不是简单读取前向所有中间量，而是：

- 重算 `w/u`（`chunk.py:143-152`）。
- 如果 CP，展开压缩过的初始状态（`chunk.py:154-155`）。
- 重算 `h/v_new`（`chunk.py:157-168`）。
- 计算局部 `dv`（`chunk.py:169-178`）。
- 如果 CP，先做反向预处理（`chunk.py:180-197`）。
- 反传 recurrent state（`chunk.py:199-213`）。
- 计算 `dq/dk/dw/dg` 和 WY 反向（`chunk.py:214-244`）。
- 如果 gate 在 kernel 中，调用 `gdn_gate_bwd`（`chunk.py:246-248`）。

`ChunkGatedDeltaRuleFunction.forward` 保存了 `q/k/v/g/beta/A/initial_state/cu_seqlens/chunk_indices/g_input/A_log/dt_bias` 等张量（`chunk.py:299-303`）。这解释了为什么 chunk path 虽然降低了 attention 的二次复杂度，但并不意味着激活显存可以忽略。

## 3.4 关键细节与误区澄清

第一个误区：`fused_recurrent` 看起来更简单，为什么训练不直接用它？答案在源码里非常明确。`FusedRecurrentFunction.backward` 直接抛出 `NotImplementedError`，注释说明 backward 需要完整 hidden states，当前没有实现（`fused_recurrent.py:302-309`）。因此训练主路径不能依赖 fused recurrent。

第二个误区：`g` 的语义在所有路径中都一样。实际不是。`chunk_gated_delta_rule_fwd` 在 `use_gate_in_kernel=True` 时，把输入 `g` 当作 raw preactivation，并保留为 `g_input`（`chunk.py:47`），然后通过 `gdn_gate_chunk_cumsum` 得到累积后的 log gate（`chunk.py:48-57`）。如果 `use_gate_in_kernel=False`，则把 `g` 交给 `chunk_local_cumsum`（`chunk.py:58-65`），语义更接近已经准备好的 log decay。CP README 也专门说明代码中的 `g` 在 cumsum 前后语义不同：输入是 log alpha 或 preactivation，cumsum 后是 log gamma（`fla/ops/cp/README.md:99-104`）。

第三个误区：线性注意力就一定极省训练显存。chunk path 的确避免了标准 attention 的 `[B,H,T,T]` attention matrix，但它仍然会产生 `A [B,T,HV,BT]`、`w/u/v_new [B,T,HV,*]`、`h [B,NT,HV,K,V]`，并保存多组张量给 backward。`chunk_fwd.py:368-371` 可以看到 `w/u/A` 的输出 shape；`chunk_delta_h.py:691-698` 可以看到 `h/final_state/v_new` 的分配。显存收益是真实的，但不是“没有中间激活”。

## 3.5 本章小结

💡 小结

- 标准训练路径是 chunk，不是 fused recurrent；后者当前没有 backward。
- chunk path 用 WY 表示和跨块 state scan 换取训练并行度。
- gate-in-kernel 让 layer 不必预先计算 decay，但也让 `g` 的语义依赖配置路径。
- 线性注意力节省的是二次 attention matrix，并不消除所有中间 buffer。

# 四、完整主路径串联：从用户调用到 Triton kernel

## 4.1 完整调用栈

下面按一次普通训练 forward 串联主路径：

```text
User: AutoModelForCausalLM.from_config(GatedDeltaNetConfig)
  │
  ├─ Step 1: 配置加载与注册
  │     └─ GatedDeltaNetConfig / AutoModelForCausalLM.register
  │        configuration_gated_deltanet.py:13-99
  │        __init__.py:8-15
  │
  ├─ Step 2: 模型装配
  │     └─ GatedDeltaNetForCausalLM -> GatedDeltaNetModel -> GatedDeltaNetBlock
  │        modeling_gated_deltanet.py:187-200, 275-287, 41-83
  │
  ├─ Step 3: Block forward
  │     └─ norm -> Attention 或 GatedDeltaNet -> residual -> MLP
  │        modeling_gated_deltanet.py:85-115
  │
  ├─ Step 4: GatedDeltaNet layer forward
  │     └─ 投影 / 短卷积 / mode 选择 / cache 读写
  │        fla/layers/gated_deltanet.py:201-316
  │
  ├─ Step 5: Gated Delta Rule 算子
  │     └─ chunk_gated_delta_rule -> ChunkGatedDeltaRuleFunction
  │        fla/ops/gated_delta_rule/chunk.py:352-518
  │
  └─ Step 6: Triton kernels
        └─ gate cumsum / WY / state scan / output / backward kernels
           gate.py, chunk_fwd.py, chunk_delta_h.py, chunk_o.py, wy_fast.py
```

## 4.2 每一层做了什么

**配置层**接收用户传入的模型超参，输出 `GatedDeltaNetConfig`。这一层不触发通信，不分配 GPU 激活，主要影响后续 layer 类型、head shape、gate、conv、loss fuse 等路径。

**模型装配层**接收 config，创建 embedding、layer stack、final norm 和 LM head。`GatedDeltaNetModel.__init__` 在 `modeling_gated_deltanet.py:187-200` 创建 `embed_tokens`、`layers`、`norm`，并调用 `post_init()`。这里会分配参数，并触发 `_init_weights`，但不执行训练 step。

**Block 层**接收 hidden states 和 cache 参数，输出同 shape hidden states。它决定当前层是标准 Attention 还是 GDN layer。这里没有跨 rank 通信；显存影响来自 norm、attention/GDN、MLP 激活。

**GatedDeltaNet layer**接收 `[B,T,D]`，输出 `[B,T,D]`，并可能修改 cache。它会根据 `self.training` 和 `q_len` 选择 `chunk` 或 `fused_recurrent`（`gated_deltanet.py:219-221`）。如果有 attention mask 且没有 `cu_seqlens`，它会 unpad 到 `[1,total_tokens,D]`（`gated_deltanet.py:225-228`），输出后再 pad 回来（`gated_deltanet.py:313-314`）。

**算子层**接收结构化的 `q/k/v/g/beta`。训练主路径进入 `chunk_gated_delta_rule`，做 shape 校验、可变长校验、CP 约束检查，然后进入 autograd Function（`chunk.py:452-515`）。这一层是计算和显存的核心。

**kernel 层**执行 Triton kernel。gate cumsum、WY 表示、state scan、output projection 前的聚合都在这里完成。普通模型 forward 没有分布式通信；只有显式传入 `cp_context` 或启用 intra-card backend 时，才会改变 state scan 行为。

## 4.3 哪些逻辑不在主路径

| 看似相关的逻辑 | 是否在普通模型训练主路径 | 正确理解 |
|---|---:|---|
| `fused_recurrent_gated_delta_rule` | 否 | eval 短序列 / 推理路径；backward 未实现（`fused_recurrent.py:302-309`） |
| Context Parallel `cp_context` | 否 | 算子支持，但生产 `GatedDeltaNet.forward` 没有把 `cp_context` 透传给 `chunk_gated_delta_rule` |
| `benchmarks/cp/test_gdn_with_cp.py` 的 layer monkeypatch | 否 | benchmark/实验路径，用 patch 让 layer 支持 CP，不是默认模型路径 |
| Intra-card CP backend | 否 | backend dispatch 下的推理 varlen 优化，需要 env 和 verifier 条件 |
| `tests/context_parallel/test_cp_gdn.py` 的 all_gather 输出 | 否 | 测试为对齐参考结果而 gather local outputs，不是 GDN op 本身每次都 gather logits |
| 自定义 save/load | 否 | 当前未看到 GDN 专属 checkpoint 逻辑，主要走 HF 默认路径 |

这里最值得强调的是 CP：`chunk_gated_delta_rule` API 支持 `cp_context`（`chunk.py:470-476`），CP 测试也直接调用 op（`tests/context_parallel/test_cp_gdn.py:193-224`），但 `GatedDeltaNet.forward` 的调用参数里没有把 `kwargs.get('cp_context')` 传给 `chunk_gated_delta_rule`（`gated_deltanet.py:265-279`）。因此“GDN 算子支持 CP”和“普通 GDN 模型层默认启用 CP”是两回事。

## 4.4 本章小结

💡 小结

- 主路径是 HF 模型 -> Block -> GDN layer -> chunk op -> Triton kernels。
- 普通训练路径没有跨 rank 通信，Context Parallel 是显式算子路径。
- fused recurrent 是推理 / 短 eval 路径，不是可训练替代路径。
- benchmark 和测试里出现的 patch、gather、direct op call 需要和生产模型路径区分开。

# 五、关键数据流 / 状态流 / Shape 流程

## 5.1 Tensor shape 变化

GatedDeltaNet 的 shape 流程可以用一个具体例子表示：

```text
原始输入:
  hidden_states: [B, T, D]

投影后:
  q:    [B, T, H*K]
  k:    [B, T, H*K]
  v:    [B, T, HV*V]
  g:    [B, T, HV]
  beta: [B, T, HV]

reshape 后:
  q:    [B, T, H,  K]
  k:    [B, T, H,  K]
  v:    [B, T, HV, V]
  g:    [B, T, HV]
  beta: [B, T, HV]

chunk op 输出:
  o:    [B, T, HV, V]

flatten + output norm/gate:
  o:    [B, T, HV*V] == [B, T, value_dim]

最终投影:
  out:  [B, T, D]
```

默认配置里 `hidden_size=2048`、`num_heads=6`、`head_dim=256`，所以 `key_dim=1536`；`expand_v=1.5`，所以 `value_dim=3072`；`num_v_heads=12` 时 `head_v_dim=256`。这与 layer 注释中的 `q/k` 各 0.75D、`v/g/o` 各 1.5D 对齐（`gated_deltanet.py:35-48`）。

如果存在 padding，且调用者只传入 `attention_mask`，layer 会先变成 varlen 形式：

```text
原始:
  hidden_states:  [B, T, D]
  attention_mask: [B, T]

unpad:
  indices:        [total_tokens]
  cu_seqlens:     [B+1]
  hidden_states:  [1, total_tokens, D]

算子内部:
  q/k/v/g/beta 的 batch 维为 1
  cu_seqlens 标记每个原始样本的边界

pad 回:
  out: [B, T, D]
```

对应 `get_unpad_data`（`fla/layers/utils.py:79-103`）、`index_first_axis` 调用（`gated_deltanet.py:225-228`）和 `pad_input`（`gated_deltanet.py:313-314`）。这样做的收益是 kernel 不必在 padding token 上浪费计算；代价是引入 gather/scatter，以及潜在的 contiguity copy。

## 5.2 状态流：GDN 的 cache 不是 KV cache

Transformer generation 习惯保存 KV cache，而 GDN 保存的是 recurrent state 和短卷积 state：

```text
past_key_values[layer_idx]
  -> recurrent_state: [N, HV, K, V]
  -> conv_state:      由 ShortConvolution 维护
  -> seen_tokens / offset
```

`get_layer_cache` 要求使用 cache 的 layer 必须有 `layer_idx`，否则抛错（`fla/layers/utils.py:212-216`）。`update_layer_cache` 负责把新的 recurrent/conv state 写回 cache（`fla/layers/utils.py:219-223`）。底层 `FLALayer` 明确有 `recurrent_state`、`conv_state`、`attention_state` 等字段（`fla/models/utils.py:60-66`）。

`GatedDeltaNetModel.forward` 还兼容 legacy cache：如果传入的是旧格式，会通过 `Cache.from_legacy_cache` 转换（`modeling_gated_deltanet.py:238-239`）。Generation 相关输入裁剪由 `FLAGenerationMixin.prepare_inputs_for_generation` 处理（`fla/models/utils.py:424-492`）。

这解释了一个重要差异：GDN 推理时的 cache 大小不随历史 token 数线性增长，而主要取决于 `[HV,K,V]` 的状态形状；但训练时仍然需要保存或重算 chunk 级中间状态。

## 5.3 Rank / Mesh / Process Group 变化

普通 GDN 模型 forward 没有构建 device mesh，也没有创建 process group。所有 rank 维度都来自两类路径：

1. 测试或用户直接调用 `chunk_gated_delta_rule(..., cp_context=...)`。
2. benchmark 或自定义 patch 把 `cp_context` 接进 layer。

CP context 的构造在 `fla/ops/cp/context.py:155-172`。它会根据 `dist.get_rank(group)` 和 `dist.get_world_size(group)` 计算当前 rank 的 token 区间（`context.py:69-85`），然后用 `searchsorted` 找出与本 rank 区间重叠的序列（`context.py:87-101`），最后生成 local `cu_seqlens`、pre/post rank 元数据和 group 信息（`context.py:103-140`）。

一个典型形状是：

```text
world_size = 4
全局 total_tokens = T
part_len = T // 4

rank0: [0, part_len)
rank1: [part_len, 2*part_len)
rank2: [2*part_len, 3*part_len)
rank3: [3*part_len, 4*part_len)
```

源码里 `part_len = total_tokens // world_size`（`context.py:80-85`），测试也显式要求 `T % world_size == 0`（`tests/context_parallel/test_cp_gdn.py:120-121`）。因此当前 CP 测试覆盖的是均匀切分，非整除 remainder 路径未在源码中确认。

## 5.4 状态切换：backend dispatch 不是 monkey patch

FLA 有一个 backend dispatch 机制，`@dispatch('common')` 装饰在 common op 上，例如 `chunk_gated_delta_rule_fwd_h`（`chunk_delta_h.py:662`）。dispatch 装饰器在调用时懒加载后端，按优先级找可用 backend；如果没有命中，则回到默认函数（`fla/ops/backends/__init__.py:139-195`）。

这不是 monkey patch。它没有替换 Python 模块命名空间，也没有把某个全局函数永久改掉；它是 wrapper 内部的动态选择。Intra-card backend 的 verifier 也很明确：只在 inference mode 且存在 `cu_seqlens` 时生效（`fla/ops/common/backends/intracard.py:41-67`），并且默认不开启，环境变量是 `FLA_INTRACARD_CP`（`intracard.py:32-35`，`ENVs.md:57`）。

## 5.5 本章小结

💡 小结

- GDN 的核心 shape 是 `q/k [B,T,H,K]`、`v [B,T,HV,V]`，输出再投回 hidden size。
- GDN cache 保存 recurrent state 和 conv state，不是 Transformer 的 KV cache。
- 普通模型路径没有 rank/group；CP rank 逻辑只在显式 `cp_context` 路径出现。
- backend dispatch 是调用期选择，不是全局 monkey patch。

# 六、Context Parallel：算子级能力如何跨 rank 补齐状态

## 6.1 设计哲学与核心问题

Context Parallel 试图解决长序列在单卡上放不下或吞吐不足的问题：把同一条长序列切到多个 rank 上算。但对 Gated DeltaNet 这类状态模型来说，切分不是简单地把 token 分段。rank1 的第一个 token 需要 rank0 末尾累积出来的状态，rank2 又需要 rank1 的状态。否则每个 rank 都会从错误的初始状态开始。

所以 CP 的核心问题不是“怎么 split input”，而是：**如何让每个 rank 在只持有局部 tokens 的情况下，获得正确的 recurrent 初始状态，并在反向时把状态梯度正确传回前面的 rank。**

## 6.2 源码入口与关键对象

```text
fla/ops/cp/context.py
  - FLACPContext：记录 rank、world_size、group、local cu_seqlens、pre/post rank 元数据
  - build_cp_context：从全局 cu_seqlens 和 process group 构造本 rank 上下文

fla/ops/cp/chunk_delta_h.py
  - chunk_gated_delta_rule_fwd_h_pre_process：前向补齐跨 rank 初始状态
  - chunk_gated_delta_rule_bwd_dhu_pre_process：反向补齐跨 rank 状态梯度
  - compress_h0 / expand_h0：减少 autograd 保存的初始状态

fla/ops/cp/comm.py
  - all_gather_into_tensor：当前 CP 状态交换使用的通信封装
```

## 6.3 主流程拆解

CP README 对架构的描述很直接：每个 rank 先做本地计算，构造 transition / message，再通过 all-gather 收集其他 rank 的 compact state，随后 merge 出本 rank 所需的初始状态（`fla/ops/cp/README.md:170-185`）。GDN 前向流程在 README 中也有独立小节（`README.md:249-286`）。

源码里的前向预处理是 `chunk_gated_delta_rule_fwd_h_pre_process`（`fla/ops/cp/chunk_delta_h.py:797-880`）：

```text
if cp_context is None:
    return initial_state

assert initial_state is None

hm = zeros([HV, K, V+K])
initial_state = zeros([N, HV, K, V])

if not last_rank:
    计算本 rank 的 transition/message -> hm

all_gather_into_tensor(hm)

if not first_rank:
    merge 前面 ranks 的 hm -> initial_state[0]
```

关键通信是 `all_gather_into_tensor`（`chunk_delta_h.py:859`），不是 all-to-all，也不是 reduce-scatter。通信对象不是完整 hidden states，而是压缩后的 transition/message，shape 接近 `[HV,K,V+K]`。这比直接 gather 全部 token 激活更小，但仍然是每个使用 CP 的 GDN op / layer 都要付出的通信成本。

反向预处理是 `chunk_gated_delta_rule_bwd_dhu_pre_process`（`chunk_delta_h.py:883-973`）：

```text
if cp_context is None:
    return dht, None

assert dht is None

dhm = zeros([HV, K, V+K])
dht = zeros([N, HV, K, V])

if not first_rank:
    从本 rank 的状态梯度构造 dhm

all_gather_into_tensor(dhm)

if not last_rank:
    merge 后面 ranks 的 dhm -> dht[-1]
```

反向方向和前向相反：前向要从前面的 rank 得到历史状态，反向要把未来 token 对状态的梯度传回前面的 rank。

`compress_h0` / `expand_h0` 是一个显存细节。前向在 CP 下会把 `initial_state` 压缩保存：如果是多序列局部上下文，只保存第一份初始状态的 clone（`chunk_delta_h.py:976-980`）；反向再根据 `cu_seqlens` 展开（`chunk_delta_h.py:983-989`）。CP README 也把它列为 initial state memory optimization（`fla/ops/cp/README.md:379-385`）。

## 6.4 关键细节与误区澄清

第一，CP 是算子级能力，不是普通 `GatedDeltaNet` layer 默认能力。`chunk_gated_delta_rule` 明确有 `cp_context` 参数和相关约束（`chunk.py:470-476`），但 `GatedDeltaNet.forward` 调用它时没有传入 `cp_context`（`gated_deltanet.py:265-279`）。因此在不 patch layer 的情况下，用户只创建 `GatedDeltaNetForCausalLM` 并不会自动触发 CP。

第二，CP 当前有硬约束。公开 API 用 assert 要求：`initial_state is None`、`output_final_state is False`、`cu_seqlens is not None`（`chunk.py:470-476`）。这意味着 CP 面向 varlen / prefill 式计算，而不是带已有 recurrent cache 的增量生成。还要注意这些是 `assert`，Python `-O` 模式下可能被优化掉，属于维护风险。

第三，CP 测试里的通信不要误读。`tests/context_parallel/test_cp_gdn.py` 中为了和单卡参考结果对齐，会 gather 各 rank 的局部输出和梯度（`test_cp_gdn.py:229-252`）。这是测试验证逻辑，不代表算子主流程每层都会 all-gather 全部 output。算子内部通信发生在状态预处理里的 compact state all-gather。

第四，CP README 中有一处测试引用偏旧：它在测试引用处主要提到 KDA/conv（`fla/ops/cp/README.md:388-391`），但仓库实际已经有 `tests/context_parallel/test_cp_gdn.py` 覆盖 GDN CP。应以源码和测试为准。

## 6.5 本章小结

💡 小结

- CP 解决的是序列切分后 recurrent 初始状态不正确的问题。
- GDN CP 的核心通信是 compact transition/message 的 all-gather，而不是 gather 全部 token。
- 前向从前序 rank 补状态，反向从后续 rank 补状态梯度。
- 当前 CP 支持停留在算子层；普通 GDN model forward 未默认接入 `cp_context`。

# 七、显存、性能与通信分析

## 7.1 显存收益范围

GDN 的显存收益不能简单理解为“线性注意力 = 显存全省”。更准确地说，它主要省掉标准 attention 的二次 attention matrix，同时把成本转移到 recurrent state、chunk 中间表示、WY 表示和若干重算上。

| 内容 | 是否节省 | 源码依据 / 原因 |
|---|---:|---|
| 参数 | ❌ | Layer 注释给出约 `6 * hidden_size^2` 参数预算（`gated_deltanet.py:35-48`），不是参数压缩方案 |
| 标准 attention matrix | ✅ | GDN op 不构造 `[B,H,T,T]` 注意力矩阵，改用 chunk state / WY 表示 |
| 训练激活 | 部分 ✅ | 省掉二次 attention，但保存 `q/k/v/g/beta/A` 等（`chunk.py:299-303`）和 chunk buffer |
| recurrent state | ✅/代价较小 | 推理 cache 主要是 `[HV,K,V]` 状态，不随历史 token 线性增长 |
| logits | 可选 ✅ | `fuse_linear_cross_entropy=True` 且有 labels 时可避免完整 logits 常驻（`modeling_gated_deltanet.py:361-376`） |
| padding token 计算 | ✅ | attention_mask 路径会 unpad 到 `[1,total_tokens,D]`（`gated_deltanet.py:225-228`） |
| 中间 buffer | ❌ | chunk path 分配 `A/w/u/h/v_new` 等，仍有显著 buffer |
| optimizer state | ❌ | 源码中未看到 GDN 专属 optimizer sharding 或压缩 |

真正的大头分两类：训练时的中间激活与 LM logits。前者由 chunk 算子内部控制，后者由 `fuse_linear_cross_entropy` 控制。配置类还禁止同时开启 `fuse_norm` 和 `fuse_linear_cross_entropy`，否则直接报错（`configuration_gated_deltanet.py:78-87`），说明 fused CE 是一个有约束的显存优化，而不是无条件开关。

## 7.2 通信开销

普通 GDN 模型 forward 没有通信原语。通信只在 CP 或测试验证中出现。

| 路径 | 通信类型 | 频率 | 通信内容 | 说明 |
|---|---|---:|---|---|
| 普通模型训练 | 无 | 0 | 无 | GDN layer 本身不创建 process group |
| CP 前向 | all-gather | 每个 CP GDN op/layer | `hm [HV,K,V+K]` 类 compact state | `chunk_delta_h.py:859` |
| CP 反向 | all-gather | 每个 CP GDN op/layer | `dhm [HV,K,V+K]` 类 compact gradient message | `chunk_delta_h.py:948` |
| CP 测试验证 | all-gather | 测试阶段 | 局部输出和梯度 | `test_cp_gdn.py:229-252`，不是 op 主流程 |
| Intra-card backend | 无网络通信 | backend 命中时 | 单卡内 subseq split/merge | `intracard.py` + `intracard_cp.py` |

CP 没有使用 all-to-all 或 reduce-scatter 来交换 token 激活；`fla/ops/cp/comm.py` 中 send/recv 风格函数当前也用 all-gather 封装以简化实现（`comm.py:64-137`）。这降低了实现复杂度，但也意味着每个 rank 会看到所有 rank 的 compact message，规模随 world size 增长。

## 7.3 性能取舍

训练主路径的性能取舍是：**用 chunk/WY/kernel fusion 换训练并行度，用中间 buffer 和部分重算换掉二次 attention。**

几个源码细节能说明这一点：

- `chunk_gated_delta_rule_fwd_intra` 把 KKT 和 triangular solve 融合，减少 HBM 往返（`chunk_fwd.py:341-350`）。
- `prepare_wy_repr_bwd` 在 GQA 场景下会对 `dk` 做 group reduce（`wy_fast.py:354-355`），避免直接复制 q/k head。
- `fused_recurrent` 前向更接近推理状态更新，但 backward 未实现（`fused_recurrent.py:302-309`），所以不能替代训练 chunk。
- `benchmarks/ops/run.py` 中注释记录了 fuse-gdn 在部分配置上前向约 1.08-1.21x、前后向约 1.01-1.06x 的加速（`benchmarks/ops/run.py:56-68`）。这是 benchmark 注释，不等同所有模型端到端收益，但能说明 kernel fusion 的收益主要在算子局部。

CP 的性能取舍则是：**用每层 compact all-gather 换长序列切分后的状态正确性。** 如果模型层数多，且每层都是 GDN，那么 CP 通信会按层累积；源码中未看到 overlap 通信与计算的实现，当前更像串行状态预处理。

## 7.4 本章小结

💡 小结

- GDN 主要省掉二次 attention matrix，不省参数，也不自动省 optimizer state。
- chunk 训练路径仍有 `A/w/u/h/v_new` 等中间 buffer，显存收益是“部分”而不是绝对。
- CP 通信是 compact state all-gather，按使用 CP 的 GDN op/layer 发生。
- 性能收益高度依赖序列长度、head 形状、是否 varlen、是否启用 fused CE 和 CP。

# 八、配置项、边界条件与坑点

## 8.1 配置如何改变源码路径

| 配置 / 条件 | 影响源码路径 | 行为变化 | 风险 / 坑点 |
|---|---|---|---|
| `attn_mode='chunk'` | `GatedDeltaNet.forward` | 训练主路径进入 `chunk_gated_delta_rule` | 训练态只允许 chunk（`gated_deltanet.py:219-221`） |
| `q_len <= 64 and not training` | `GatedDeltaNet.forward` | 自动切到 fused recurrent | 如果 eval 但仍需要 grad，会遇到 backward 未实现 |
| `use_gate` | layer 输出端 | 开启/关闭 `g_proj + FusedRMSNormGated` | 不控制 recurrent decay gate |
| `use_short_conv` | q/k/v 投影后 | 使用 `ShortConvolution` 和 conv cache | 关闭时只 warning，不是错误（`gated_deltanet.py:189-193`） |
| `allow_neg_eigval` | beta 计算 | `beta = b_proj(...).sigmoid()` 后可映射到 `[-1,1]` | 改变 update gate 数值范围（`gated_deltanet.py:260-262`） |
| `num_v_heads` / `num_heads` | layer/op shape | GVA/GQA 式 value heads | 部分非法配置可能到 op 才失败（`chunk.py:452-463`） |
| `fuse_linear_cross_entropy` | LM forward loss | labels 存在时可避免完整 logits loss 路径 | 与 `fuse_norm` 冲突（`configuration_gated_deltanet.py:78-87`） |
| `attention_mask` 无 `cu_seqlens` | layer varlen | unpad -> op -> pad | 同时传 `attention_mask` 与 `cu_seqlens` 可能触发 `indices` 未定义风险 |
| `cp_context` | `chunk_gated_delta_rule` | 开启 CP 状态补齐 | 普通 layer 未透传，需直接调 op 或 patch |
| `FLA_INTRACARD_CP` | backend dispatch | 推理 varlen 下可能走 intra-card backend | 默认关闭，只在 verifier 条件满足时生效 |

## 8.2 硬约束与静默失效

几个约束在源码里很明确：

- `chunk_gated_delta_rule` 要求 `q` 和 `k` head 数一致（`chunk.py:452-457`）。
- `HV % H == 0`，否则 GVA mapping 不成立（`chunk.py:458-463`）。
- `head_first` 已 deprecated，仍传会报错（`chunk.py:465-468`）。
- CP 要求 `initial_state is None`、`output_final_state is False`、`cu_seqlens is not None`（`chunk.py:470-476`）。
- varlen 模式要求 `B == 1`，且 initial_state 的 N 与 `cu_seqlens` 对齐（`chunk.py:478-488`）。
- `use_gate_in_kernel=True` 时必须有 `A_log`（`chunk.py:489-494`）。
- common state scan kernel 要求 `K <= 256`（`chunk_delta_h.py:689`，反向为 `chunk_delta_h.py:744`）。

静默或半静默风险主要来自三处：

1. `assert` 约束：CP 和部分 gate 约束用 assert 表达，Python `-O` 下可能被去掉。
2. backend env：`FLA_INTRACARD_CP` 默认关闭，如果用户以为 CP 自动开启，会发现普通模型路径并无变化。
3. hybrid attention：`attn` 只替换指定层，如果 layers 配置不符合预期，模型结构会改变但不一定立刻报错。

## 8.3 保存 / 加载 / resume 差异

当前源码中未发现 GDN 专属的保存、加载、resume 或 state_dict merge 逻辑。训练参数保存主要依赖 HuggingFace `PreTrainedModel` 默认机制；generation cache 是运行时状态，不属于模型权重。测试中也未看到针对 GDN save/load/resume 的专门用例。因此，如果把 GDN 和外部分布式 checkpoint 框架结合，权重保存本身应按普通模型看待，但 recurrent cache、CP context 这类运行态不应误认为会被 checkpoint 自动恢复。

## 8.4 本章小结

💡 小结

- 最小开启方式是创建 `GatedDeltaNetConfig` 并通过 HF Auto 类加载模型。
- `mode/training/q_len` 决定 chunk 与 fused recurrent 路径，训练主路径被限制在 chunk。
- CP、intra-card backend、fused CE 都是条件路径，不会无条件生效。
- 多个关键约束在 op 层才校验，配置合法不代表运行路径一定合法。

# 九、测试、示例与覆盖缺口

## 9.1 已覆盖路径

测试不是简单证明“能跑”，而是分别覆盖了几个关键机制。

| 测试 / 示例 | 覆盖的行为 | 说明 |
|---|---|---|
| `tests/models/test_modeling_gated_deltanet.py:19-38` | 模型 forward/backward | 通过 base helper 测固定长度与 varlen backward |
| `tests/models/test_modeling_gated_deltanet.py:44-61` | generation cache | 比较分块生成与无 cache 参考输出 |
| `tests/ops/test_gated_delta.py:104-169` | chunk fwd/bwd vs naive | 验证 chunk 算子数值与梯度 |
| `tests/ops/test_gated_delta.py:443-529` | chunk varlen fwd/bwd | 验证 `cu_seqlens` 变长路径 |
| `tests/ops/test_gated_delta.py:624-708` | gate-in-kernel | 验证 raw gate 进 kernel 的前后向 |
| `tests/ops/test_gated_delta.py:726-808` | GQA/GVA | 覆盖 `num_v_heads` 与 `num_heads` 不同的路径 |
| `tests/context_parallel/test_cp_gdn.py:193-273` | GDN CP | 分布式场景对齐单卡参考输出与梯度 |
| `tests/ops/test_intracard_cache.py:129-183` | GDN intra-card backend | 验证启用 / 禁用 backend 的输出一致性 |
| `benchmarks/ops/registry.py:276-286` | benchmark 注册 | 把 `chunk_gdn` 放入算子 benchmark 体系 |

模型测试 helper 里，varlen 测试会把输入 reshape 成 `[1, B*T]` 并传入 `cu_seqlens`（`tests/models/test_modeling_base.py:61-66`），随后执行 backward（`test_modeling_base.py:67`）。generation helper 会先 prefill，再 token-by-token decode，并和无 cache reference 比较（`test_modeling_base.py:107-133`）。

## 9.2 未覆盖风险

| 风险点 | 当前是否有测试 | 可能后果 |
|---|---:|---|
| 普通 `GatedDeltaNet.forward` 透传 `cp_context` | ❌ | 用户以为模型层支持 CP，但实际只有 op/direct/patch 路径 |
| eval 模式短序列但需要 backward | 未见专门测试 | 自动进入 fused recurrent，backward 抛 NotImplemented |
| 同时传 `attention_mask` 和 `cu_seqlens` | 未见专门测试 | 可能触发 pad 阶段 `indices` 未定义 |
| save/load/resume | 未见 GDN 专门测试 | checkpoint 集成风险未覆盖 |
| Python `-O` 下 assert 约束 | 未见测试 | CP/gate 约束可能失效后进入错误 kernel |
| 多机 CP | 主要依赖多 GPU 测试，未确认多节点 | 网络拓扑、group 初始化和性能风险未覆盖 |
| 性能 / 显存收益断言 | benchmark 有，但非 CI 语义 | 无法防止后续改动引入显存回退 |
| benchmark layer monkeypatch 与生产 layer 差异 | 未见一致性测试 | benchmark 路径可能和真实模型行为偏离 |

还有一些测试是有条件 skip 的。例如 chunk varlen 测试可被 `SKIP_TEST_CHUNK_VARLEN=1` 跳过（`tests/ops/test_gated_delta.py:455-458`），多处测试会在特定 Intel GPU 且维度过大时 skip（如 `test_gated_delta.py:117-118`）。这很合理，但也意味着 CI 覆盖强度取决于硬件和环境变量。

## 9.3 本章小结

💡 小结

- 算子数值、varlen、gate-in-kernel、GQA、generation cache 都有覆盖。
- CP 有直接 op 级分布式测试，但普通模型层默认 CP 路径没有覆盖，因为源码也未接入。
- save/load/resume、异常配置和性能回归是当前更明显的测试缺口。
- benchmark 能说明性能方向，但不能替代 CI 中的显存与吞吐断言。

# 十、局限性与已知优化点

## 10.1 硬约束

Gated DeltaNet 当前实现有几类硬约束：

- `K <= 256`：state scan kernel 中有 assert（`chunk_delta_h.py:689`，`chunk_delta_h.py:744`）。
- `HV % H == 0`：GVA mapping 要求 value head 数是 query head 数的整数倍（`chunk.py:458-463`）。
- `fused_recurrent` 无 backward：不能作为训练主路径（`fused_recurrent.py:302-309`）。
- CP 不支持已有 initial_state，也不输出 final_state（`chunk.py:470-476`）。
- varlen op 路径要求 `B == 1`（`chunk.py:478-488`）。
- CP 测试要求总 token 可被 world size 整除（`tests/context_parallel/test_cp_gdn.py:120-121`），源码使用 floor `part_len`，非整除行为未在源码中确认。

## 10.2 维护成本

第一类维护成本来自语义复用：`g` 在不同阶段代表 raw preactivation、log alpha 或 cumulative log gamma。CP README 已经解释了这一点（`fla/ops/cp/README.md:99-104`），但对新贡献者仍然容易误解。

第二类来自条件路径：普通 chunk、fused recurrent、varlen、GQA、gate-in-kernel、CP、intra-card backend 都有不同前置条件。新增优化时，如果只测固定长度、无 CP、无 GQA，很容易破坏边界路径。

第三类来自 benchmark patch。`benchmarks/cp/test_gdn_with_cp.py` 中存在为了测试 CP layer 场景而写的 monkeypatch 路径；它服务 benchmark，不是生产模型代码。维护者需要避免把 benchmark patch 的行为当作 `fla/layers/gated_deltanet.py` 的真实行为。

## 10.3 性能瓶颈

几个瓶颈比较明确：

- chunk path 有多阶段 kernel：gate、WY、state scan、output、反向重算。虽然局部融合减少了 HBM 往返，但仍不是单 kernel 完成。
- CP 每层需要前向和反向 compact all-gather。通信规模小于 token 激活，但按层重复，源码未看到 overlap。
- `input_guard` 会把非 contiguous tensor 转成 contiguous（`fla/utils.py:191-206`），某些上游切片输入可能隐含复制。
- LM logits 如果不启用 fused CE，仍可能是大显存峰值；启用 fused CE 又和 `fuse_norm` 冲突（`configuration_gated_deltanet.py:78-87`）。
- Intra-card CP 依赖 backend dispatch 与环境变量，默认关闭，不应作为默认性能假设。

## 10.4 已知优化点

源码注释和实现中能看到几个优化方向：

- `chunk_fwd.py` 已经把 KKT 和 solve 融合，后续仍可继续减少中间 `A/w/u` 的 HBM 压力。
- `wy_fast.py` 中有 Blackwell `safe_dot` workaround / TODO（`wy_fast.py:17-27`），说明不同 GPU 架构上仍有后端细节要维护。
- CP 当前使用 all-gather 简化状态交换，可以探索更细粒度或拓扑感知通信，但要保持前后向状态语义。
- 普通 layer 未透传 `cp_context`，如果要产品化 CP 模型层，需要设计清楚 context 的来源、生命周期、cache 兼容和 generation 限制。
- fused recurrent backward 未实现；如果实现，需要解决保存完整 hidden states 与显存收益之间的矛盾。

## 10.5 本章小结

💡 小结

- 当前实现的核心限制集中在 head 形状、chunk state 维度、CP 条件和 fused recurrent backward。
- 维护难点不是单个函数复杂，而是多条条件路径的语义一致性。
- 性能瓶颈主要在 chunk 多阶段 kernel、CP per-layer all-gather 和大 logits。
- 后续优化应优先明确 CP 模型层接入与 fused recurrent backward 的工程边界。

# 小结与展望

Flash Linear Attention 的 Gated DeltaNet 实现可以用几个关键词概括。

**关键词一：HF 生态接入。**  
GDN 不是孤立算子，而是通过 `GatedDeltaNetConfig`、Auto 类注册、`GatedDeltaNetForCausalLM` 和 generation mixin 接入普通模型使用路径。用户可以像创建 CausalLM 一样创建它，内部再分流到 Attention 或 GDN layer。

**关键词二：双门控。**  
源码里有 recurrent decay gate，也有输出 gate。`a_proj/A_log/dt_bias` 控制状态递推衰减；`use_gate` 控制输出端 `g_proj + FusedRMSNormGated`。混淆这两者，是阅读 GDN layer 时最常见的误区。

**关键词三：Chunk/WY 训练主路径。**  
训练不是逐 token recurrent scan，而是 chunk 化：块内 WY 表示、块间 state scan、输出聚合、反向重算。它用更复杂的 kernel 和中间 buffer 换来训练并行度，并避免标准 attention 的二次矩阵。

**关键词四：算子级 Context Parallel。**  
CP 的核心是跨 rank 补齐 recurrent 初始状态和反向状态梯度。它在 op 层通过 compact all-gather 实现，当前普通 `GatedDeltaNet.forward` 未默认透传 `cp_context`。因此“算子支持 CP”和“模型默认启用 CP”必须区分。

**关键词五：通信换状态正确性，融合换 HBM 压力。**  
GDN 的性能收益来自两类交换：训练侧用 kernel fusion 和重算减少 HBM 压力；长序列 CP 侧用 per-layer compact 通信换取状态边界正确性。它适合长序列建模、需要高效 prefill / generation cache 的场景；不适合期望无条件减少参数、无通信开销、或希望 CP 在模型层开箱即用的场景。

和标准 Transformer attention 相比，GDN 的取舍不是简单“更快更省”，而是把复杂度从 `[T,T]` attention matrix 转移到 recurrent state、chunk 表示、Triton kernel 和状态通信协议上。后续最值得继续走读的方向有三个：第一，Qwen3-Next 如何在上层模型中使用 GDN；第二，CP 如何产品化接入模型层而不是只停留在算子接口；第三，fused recurrent backward 是否能在不破坏显存优势的前提下补齐训练路径。

