# FLA 源码走读：RWKV7 implementations 实现解析

在长序列建模里，RWKV 系列一直试图绕开标准注意力的二次复杂度：不把所有历史 token 都保存在 KV cache 中，而是把历史压进随时间演化的矩阵状态。对 `flash-linear-attention`（下文简称 FLA）来说，RWKV7 的接入并不是“新增一个模型类”这么简单：它既要接入 HuggingFace `AutoModelForCausalLM`，又要在训练时走可反传的 chunk kernel，在推理时走低延迟 recurrent kernel，还要用一组小的 fused op 把 RWKV7 特有的混合、key 更新和输出修正压到 Triton 里。

本文不展开 RWKV7 论文里的完整数学推导，也不证明 DPLR delta rule 的理论正确性。我们只顺着 FLA 当前源码回答一个工程问题：**RWKV7 在这个仓库里到底是怎样从 `RWKV7Config` 走到 Triton kernel 的？它在哪些地方节省了时间或显存，又在哪些地方埋下了 shape、缓存、配置和维护风险？**

# 前言

## 业务 / 工程背景

FLA 是一个面向线性注意力、RNN-like attention 与 Triton kernel 的实现集合。README 在 2025-01 的更新日志中写到“Add RWKV7 implementations (both kernels and models) to `fla`”（`README.md:56`），模型列表也把 RWKV7 指向 `fla/ops/rwkv7`（`README.md:90`）。这说明 RWKV7 在仓库里有双重身份：

- **模型身份**：`RWKV7Config`、`RWKV7Model`、`RWKV7ForCausalLM`，服务 HuggingFace 风格训练、推理和保存加载。
- **算子身份**：`chunk_rwkv7`、`fused_mul_recurrent_rwkv7`、`fused_addcmul_rwkv7`、`fused_k_rwkv7`、`gate_output_correction` 等，服务长序列并行、短序列解码和局部融合。

## 核心矛盾

RWKV7 在 FLA 中的实现围绕三个工程冲突展开：

1. **训练与推理的冲突**：训练需要可并行、可反传的 chunk 算法；短序列推理则希望 recurrent 状态一步步更新，避免 chunk kernel 的启动和中间状态开销。
2. **表达力与显存的冲突**：RWKV7 多了 `a`、`kk`、`v_first`、输出修正等机制，表达力更强，但前向会制造多份 `[B, T, hidden]` 与 `[B, T, H, D]` 中间张量。
3. **HuggingFace 兼容与 kernel 约束的冲突**：外层看起来是普通 `AutoModelForCausalLM`，底层却对 `head_dim`、`value_dim`、`cu_seqlens`、padding mask、cache state 有严格假设。

## 本文主线

本文按机制而不是按文件展开：先看入口和配置，再看模型骨架与初始化，然后进入 attention 前向、DPLR chunk kernel、推理 cache、shape/state/通信流，最后讨论显存性能、配置坑点和测试缺口。

## 不展开的内容

本文不讲 RWKV7 论文的完整理论，不讲 HuggingFace `PreTrainedModel` 的通用实现，不逐行讲所有 DPLR kernel，也不把 `fla/ops/generalized_delta_rule` 写成完整源码地图。凡是涉及算法处，只说明源码如何把 RWKV7 变量映射到 FLA 已有 DPLR 机制。

## 核心文件表

| 文件 | 职责 |
|---|---|
| `fla/models/rwkv7/configuration_rwkv7.py` | 定义 `RWKV7Config`、默认配置、head/value 维度归一化、混合 attention 配置校验 |
| `fla/models/rwkv7/modeling_rwkv7.py` | HuggingFace 模型骨架：block、model、causal LM、loss、load 迁移 |
| `fla/layers/rwkv7.py` | RWKV7 attention 主实现：初始化、token shift、投影、kernel 分流、cache 更新 |
| `fla/ops/rwkv7/chunk.py` | `chunk_rwkv7` 适配层，把 RWKV7 参数映射到 generalized DPLR delta rule |
| `fla/ops/generalized_delta_rule/dplr/chunk.py` | chunk DPLR 的 autograd 主体，训练和长序列路径依赖它 |
| `fla/ops/rwkv7/fused_recurrent.py` | 短序列 / decode recurrent kernel，含 RWKV7 专用 `fused_mul_recurrent_rwkv7` |
| `fla/ops/rwkv7/fused_addcmul.py` | 六路 `hidden + delta * x_*` 融合 |
| `fla/ops/rwkv7/fused_k_update.py` | `k * (1 + (a - 1) * k_a)` 融合 |
| `fla/ops/rwkv7/gate_output_correction.py` | 输出修正与 gate 融合 |
| `tests/ops/test_rwkv7.py` / `tests/models/test_modeling_rwkv7.py` | 算子与模型路径测试 |

💡 小结

- RWKV7 在 FLA 中既是模型接口，也是 kernel 族。
- 训练主线是 chunk DPLR；推理解码主线是 recurrent state。
- 后文所有判断以 Python 源码为准；markdown 推导或 README 若与源码冲突，以源码为准。

# 一、入口与配置归一化：用户如何真正开启 RWKV7

## 1.1 设计哲学与核心问题

RWKV7 的入口被设计成 HuggingFace 兼容形式。用户不需要知道底层有 Triton kernel，只要构造 `RWKV7Config` 并交给 `AutoModelForCausalLM`。这层解决的是**接入问题和几何归一化问题**：模型工厂要能识别 `model_type='rwkv7'`，而 attention 层要拿到一致的 `hidden_size / head_dim / num_heads / value_dim`。

如果没有这层，用户必须手动 import 具体类；如果配置归一化不严，后面的 `rearrange('b t (h d) -> b t h d')` 会在运行时才暴露 shape 错误。

## 1.2 源码入口与关键对象

```text
fla/models/rwkv7/__init__.py
  - AutoConfig.register / AutoModel.register / AutoModelForCausalLM.register：接入 HF Auto API

fla/models/rwkv7/configuration_rwkv7.py
  - RWKV7Config：用户配置、默认值、维度归一化、混合 attention 参数校验

fla/models/__init__.py / fla/__init__.py
  - 对外重新导出 RWKV7Config / RWKV7Model / RWKV7ForCausalLM
```

注册发生在 `fla/models/rwkv7/__init__.py:13-15`：

```python
AutoConfig.register(RWKV7Config.model_type, RWKV7Config, exist_ok=True)
AutoModel.register(RWKV7Config, RWKV7Model, exist_ok=True)
AutoModelForCausalLM.register(RWKV7Config, RWKV7ForCausalLM, exist_ok=True)
```

因此标准入口是：

```python
from transformers import AutoModelForCausalLM
from fla.models import RWKV7Config

config = RWKV7Config()
model = AutoModelForCausalLM.from_config(config)
```

也可以直接实例化：

```python
from fla.models import RWKV7Config, RWKV7ForCausalLM
model = RWKV7ForCausalLM(RWKV7Config())
```

仓库里没有发现专门的 RWKV7 训练 CLI；RWKV7 专用 CLI 是转换脚本 `utils/convert_from_rwkv7.py`，入口在 `utils/convert_from_rwkv7.py:150-156`。

## 1.3 主流程拆解

配置入口是 `RWKV7Config.__init__`（`configuration_rwkv7.py:18-49`）。几个真正改变行为的字段如下：

| 配置项 | 默认值 | 影响源码路径 |
|---|---:|---|
| `attn_mode` | `"chunk"` | 传给 `RWKV7Attention(mode=...)`（`modeling_rwkv7.py:153-154`），但前向分流还受 training 与 `seq_len` 控制 |
| `head_dim` / `num_heads` | `64` / `None` | `configuration_rwkv7.py:58-62` 尝试互推，`RWKV7Attention` 再根据 `head_dim` 优先计算 heads（`rwkv7.py:60-68`） |
| `value_dim` | `None` | 被归一化为每层 list（`configuration_rwkv7.py:63-77`），每层传入 attention（`modeling_rwkv7.py:165`） |
| `attn` | `None` | 非空时可把部分层替换成标准 `Attention`（`modeling_rwkv7.py:141-151`） |
| `fuse_norm` | `True` | 选择 FLA fused `LayerNorm` / `GroupNorm` 或 PyTorch norm（`modeling_rwkv7.py:130-140`、`rwkv7.py:123-137`） |
| `fuse_cross_entropy` / `fuse_linear_cross_entropy` | `True` / `False` | 控制 LM loss 路径（`modeling_rwkv7.py:520-542`） |
| `use_cache` | `True` | 推理时允许 recurrent / token-shift / FFN state 缓存；训练中默认被关闭（`modeling_rwkv7.py:381`） |

维度归一化的关键逻辑在 `configuration_rwkv7.py:58-73`：

```python
if head_dim is None and num_heads is not None:
    head_dim = int(hidden_size // num_heads)
elif head_dim is not None and num_heads is None:
    num_heads = int(hidden_size // head_dim)

if value_dim is None:
    value_dim = [hidden_size] * num_hidden_layers
elif isinstance(value_dim, int):
    assert value_dim >= hidden_size
    assert value_dim % hidden_size == 0
    value_dim = [value_dim] * num_hidden_layers
```

这段代码说明两件事：第一，`head_dim` 和 `num_heads` 并不是总会互相校验；第二，`value_dim` 虽然被设计成可逐层变化，但配置层只要求它是 `hidden_size` 的整数倍，而不是只要求能被 `num_heads` 整除。

## 1.4 关键细节与误区澄清

**误区一：`attn_mode='fused_recurrent'` 就会让训练走 recurrent kernel。**

不是。`RWKV7Attention.__init__` 只校验 mode 在 `['chunk', 'fused_recurrent']` 中（`fla/layers/rwkv7.py:54-55`），但真正前向分流是：训练或 `seq_len >= 64` 时走 `chunk_rwkv7`，否则走 `fused_mul_recurrent_rwkv7`（`fla/layers/rwkv7.py:305-334`）。源码中没有 `if self.mode == ...` 的运行时分支。

**误区二：`num_heads` 一定生效。**

默认 `head_dim=64` 非空。如果用户只传 `num_heads=16`，config 不会进入 `head_dim is None` 分支；到 `RWKV7Attention` 又优先使用 `head_dim`，重新计算 `num_heads = hidden_size // head_dim`（`rwkv7.py:62-67`）。这意味着 `num_heads` 可能被静默忽略。

**误区三：`attn` 混合 attention 配置已经是稳定主路径。**

配置层确实校验 `attn['layers']` 与 `attn['num_heads']`（`configuration_rwkv7.py:107-117`），block 初始化也会创建标准 `Attention`（`modeling_rwkv7.py:141-151`）。但 `RWKV7Block.forward` 固定按四个返回值解包，包括 `v_first`（`modeling_rwkv7.py:195-204`），而标准 `Attention.forward` 返回三个值（`fla/layers/attn.py:84-92`、`178-181`）。因此这条路径看起来被配置支持，但按当前源码存在运行时契约不一致风险。

## 1.5 本章小结

💡 小结

- RWKV7 的用户入口主要是 HuggingFace Auto API 注册，不是仓库内专用训练 CLI。
- 配置层做了 head/value 归一化，但仍存在 `num_heads` 被默认 `head_dim` 覆盖的静默风险。
- `attn_mode` 更像声明字段；当前前向 kernel 分流主要由 training 状态和 `seq_len` 决定。

# 二、模型骨架与初始化：把 RWKV7 塞进标准 Causal LM

## 2.1 设计哲学与核心问题

这一层解决的是**模型组织问题**：RWKV7 不是一个单独 attention op，而是完整 LM block。它要处理 embedding、block 堆叠、norm、FFN、LM head、loss、generation cache，以及 HF 的保存加载接口。

如果只实现 `chunk_rwkv7`，用户只能调用裸 op；如果只实现 model，不做 RWKV 特化初始化，训练初期的动态 decay、LoRA bias、输出投影会偏离官方结构。

## 2.2 源码入口与关键对象

```text
fla/models/rwkv7/modeling_rwkv7.py
  - RWKV7FeedForward：RWKV7 channel-mix 风格 FFN，但没有调用 ops/rwkv7/channel_mixing.py
  - RWKV7Block：norm + attention + norm + FFN 的 block 编排
  - RWKV7PreTrainedModel._init_weights：embedding、lm_head、非 RWKV 模块初始化
  - RWKV7Model：embedding、layers、final norm、主 forward
  - RWKV7ForCausalLM：lm_head、loss、generate 兼容处理

fla/layers/rwkv7.py
  - RWKV7Attention._initialize_weights：RWKV7 attention 参数初始化
```

## 2.3 主流程拆解

`RWKV7Model.__init__` 的骨架很直接（`modeling_rwkv7.py:290-305`）：

```python
self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
self.layers = nn.ModuleList([RWKV7Block(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
self.norm = LayerNorm(...) if config.fuse_norm else nn.LayerNorm(...)
self.post_init()
```

每个 `RWKV7Block` 再决定 attention 类型（`modeling_rwkv7.py:141-167`）：

```text
if config.attn is not None and layer_idx in config.attn['layers']:
    self.attn = Attention(...)
else:
    self.attn = RWKV7Attention(mode=config.attn_mode, ... value_dim=config.value_dim[layer_idx])
```

标准 RWKV7 block 前向可以写成：

```text
hidden_states
  -> optional pre_norm（仅 norm_first 且第 0 层）
  -> attn_norm
  -> RWKV7Attention(..., v_first, past_key_values, cu_seqlens)
  -> residual + ffn_norm
  -> RWKV7FeedForward(..., past_key_values, cu_seqlens)
  -> residual add
```

对应源码在 `modeling_rwkv7.py:182-218`。

初始化值得单独看，因为 RWKV7 很多参数不是普通 `normal_(0, initializer_range)`：

- `RWKV7Attention._initialize_weights` 按层号构造 `ratio_0_to_1`、`ratio_1_to_almost0`，初始化 `x_r/x_w/x_k/x_v/x_a/x_g`、`k_a`、`r_k`、`k_k`、LoRA bias、GroupNorm weight、投影矩阵，并把 `o_proj.weight` 置零（`fla/layers/rwkv7.py:155-213`）。
- `RWKV7FeedForward._initialize_weights` 初始化 `x_k`，对 `key.weight` 做 orthogonal，对 `value.weight` 置零（`modeling_rwkv7.py:84-97`）。
- `RWKV7PreTrainedModel._init_weights` 对 embedding 使用小 uniform，对 `lm_head` 做带 gain 的 orthogonal，对非 RWKV 模块走普通初始化（`modeling_rwkv7.py:233-264`）。

注意 `RWKV7Attention.__init__` 末尾有显式 warning（`fla/layers/rwkv7.py:148-153`）：作者提示该 FLA 实现可能存在问题，报告数值前应与官方 repo 交叉验证。这不是注释，而是运行时 warning。

## 2.4 关键细节与误区澄清

**误区一：`fla/ops/rwkv7/channel_mixing.py` 是模型 FFN 主路径。**

不是。模型中的 `RWKV7FeedForward.forward` 使用 `token_shift -> self.key -> act_fn -> self.value`（`modeling_rwkv7.py:98-115`），没有调用 `channel_mixing_rwkv7`。后者是导出的 op 并有测试（`tests/ops/test_rwkv7.py:23-77`），但不是当前 `RWKV7Model` 的标准 FFN 路径。

**误区二：单层 RWKV7 模型天然支持。**

当前初始化里 `ratio_0_to_1 = self.layer_idx / (self.num_hidden_layers - 1)`（`fla/layers/rwkv7.py:162-164`）。`num_hidden_layers=1` 会除零。测试只覆盖 `L=4`（`tests/models/test_modeling_rwkv7.py:19-39`），所以单层 toy model 不是受保护路径。

**误区三：`output_attentions=True` 会返回 RWKV attention map。**

`RWKV7Model.forward` 如果看到 `output_attentions`，会 warning 并强制设为 False（`modeling_rwkv7.py:376-378`）。RWKV7 主路径没有标准 attention 矩阵可返回。

## 2.5 本章小结

💡 小结

- RWKV7 模型层复用了 HF Causal LM 外壳，但 attention、FFN、初始化都有 RWKV 特化逻辑。
- `channel_mixing_rwkv7` 是导出 op / 测试路径，不是当前模型 FFN 主流程。
- 初始化是功能的一部分，不只是参数美化；其中也暴露了单层模型除零这样的边界风险。

# 三、Attention 前向主链路：从 token shift 到 DPLR 状态更新

## 3.1 设计哲学与核心问题

RWKV7Attention 是整套实现的核心。它解决的不是“如何算一次线性层”，而是**如何把 RWKV7 的动态状态更新拆成若干可融合、可并行、可缓存的步骤**。

没有这一层，模型无法在训练和推理之间切换：训练需要 chunk kernel 来并行扫描序列并支持 backward；推理需要 recurrent state 逐 token 更新；padding、cache、`v_first`、output gate 又必须贯穿同一个 block。

## 3.2 源码入口与关键对象

```text
fla/layers/rwkv7.py
  - RWKV7Attention.forward：attention 主路径
  - fused_addcmul_rwkv7：六路时间混合
  - fused_k_rwkv7：key update
  - chunk_rwkv7：训练 / 长序列 kernel
  - fused_mul_recurrent_rwkv7：短序列推理 kernel
  - gate_output_correction：输出修正 + gate

fla/modules/token_shift.py
  - token_shift：生成 delta 和 cache_out

fla/layers/utils.py
  - get_layer_cache / update_layer_cache：读写每层 cache
```

## 3.3 主流程拆解

`RWKV7Attention.forward` 的输入是：

```text
hidden_states: [B, T, hidden_size]
attention_mask: [B, T] or None
past_key_values: Cache or None
v_first: [B, T, value_dim]，由第 0 层产生并传给后续层
cu_seqlens: varlen flatten 模式下的累计长度
```

主流程如下：

```text
hidden_states [B,T,D]
  -> padding mask 乘到 hidden_states（如果有）
  -> get_layer_cache: 取 conv_state / recurrent_state
  -> token_shift: delta [B,T,D], conv_state [N,D]
  -> fused_addcmul_rwkv7: xr/xw/xk/xv/xa/xg 六个 [B,T,D]
  -> r_proj/k_proj/v_proj + w_lora/a_lora/g_lora
  -> v_first 跨层混合
  -> kk = normalize(k * k_k)
  -> fused_k_rwkv7(k, a, k_a)
  -> reshape r,w,k,a: [B,T,H,K], v: [B,T,H,V]
  -> 训练/长序列: chunk_rwkv7(a=-kk, b=kk*a)
     推理短序列: fused_mul_recurrent_rwkv7(kk=kk, a=a)
  -> update_layer_cache(recurrent_state, conv_state)
  -> group norm
  -> gate_output_correction
  -> o_proj
```

源码对应关系：

- padding mask 只接受二维 `[B, T]`，任意三维 mask 会 assert（`fla/layers/rwkv7.py:233-240`）。
- cache 读取在 `fla/layers/rwkv7.py:242-254`。
- token shift 在 `fla/layers/rwkv7.py:256-258`，底层 `token_shift` 要求 varlen 时 `x.shape[0] == 1`（`fla/modules/token_shift.py:549-552`）。
- 六路 addcmul 在 `fla/layers/rwkv7.py:259-260`。
- decay `w = -0.6065306597126334 * sigmoid(...)` 在 `fla/layers/rwkv7.py:271`。
- 第 0 层保存 `v_first`，后续层用 `torch.lerp` 混合（`fla/layers/rwkv7.py:276-280`）。
- kernel 分流在 `fla/layers/rwkv7.py:305-334`。
- cache 更新在 `fla/layers/rwkv7.py:336-342`。
- gate correction 和输出投影在 `fla/layers/rwkv7.py:344-350`。

其中 kernel 分流是最重要的工程判断：

```python
if self.training or seq_len >= 64:
    o, recurrent_state = chunk_rwkv7(..., a=-kk, b=kk * a, safe_gate=True, chunk_size=64)
else:
    o, recurrent_state = fused_mul_recurrent_rwkv7(..., kk=kk, a=a)
```

这说明：训练总是 chunk；eval 时如果 prefill 很长也 chunk；只有短序列 eval / decode 才走 recurrent。

## 3.4 关键细节与误区澄清

**误区一：padding 被 unpad 后送入 RWKV7 kernel。**

不是。这里没有调用 `unpad_input` / `pad_input`。RWKV7Attention 对 mask 的处理是把 padding 位置乘零：先对 `hidden_states` 乘 mask（`rwkv7.py:244-245`），后面又对 `v` 乘 mask（`rwkv7.py:298-300`）。真正的 varlen 路径靠用户传入 flatten 后的 `[1, total_T, D]` 加 `cu_seqlens`，而不是自动从 `attention_mask` unpad。

**误区二：`safe_gate=True` 是用户配置项。**

在模型主路径里不是。`chunk_rwkv7` 的参数支持 `safe_gate`（`fla/ops/rwkv7/chunk.py:25`），但 `RWKV7Attention.forward` 写死 `safe_gate=True` 与 `chunk_size=64`（`fla/layers/rwkv7.py:318-320`）。它成立的依据是 `w` 被限制在 `(-0.6065, 0)`（`rwkv7.py:271`），符合 `chunk.py` 文档里 safe gate 约 `[-5, 0)` 的说明（`fla/ops/rwkv7/chunk.py:56-59`）。

**误区三：`fused_addcmul_rwkv7` 会在所有场景使用 Triton。**

不是。`fused_addcmul_rwkv7` 在 `hidden_states.shape[1] == 1` 时直接回退到 PyTorch `torch.addcmul`（`fla/ops/rwkv7/fused_addcmul.py:273-286`）。短 decode 避免了为一个 token 启动 Triton kernel。

## 3.5 本章小结

💡 小结

- RWKV7Attention 是状态、shape、kernel 分流的交汇点。
- 训练/长 prefill 走 `chunk_rwkv7`；短 eval/decode 走 `fused_mul_recurrent_rwkv7`。
- padding mask 只做乘零；真正 varlen 要用户传 flatten 输入和 `cu_seqlens`。

# 四、完整主路径串联：一次真实用户调用如何落到 kernel

## 4.1 完整调用栈

```text
User: AutoModelForCausalLM.from_config(RWKV7Config()).forward(input_ids, labels?, use_cache?, cu_seqlens?)
  │
  ├─ Step 1: HF Auto 注册与模型构造
  │     └─ fla/models/rwkv7/__init__.py:13-15
  │
  ├─ Step 2: 配置归一化
  │     └─ RWKV7Config.__init__
  │        configuration_rwkv7.py:18-125
  │
  ├─ Step 3: 模型 / block / layer 构建
  │     ├─ RWKV7Model.__init__
  │     │  modeling_rwkv7.py:290-305
  │     └─ RWKV7Block.__init__
  │        modeling_rwkv7.py:130-180
  │
  ├─ Step 4: CausalLM forward
  │     └─ RWKV7ForCausalLM.forward
  │        modeling_rwkv7.py:482-554
  │
  ├─ Step 5: Base model forward
  │     └─ RWKV7Model.forward
  │        modeling_rwkv7.py:363-432
  │
  ├─ Step 6: 每层 block forward
  │     └─ RWKV7Block.forward
  │        modeling_rwkv7.py:182-218
  │
  ├─ Step 7: RWKV7 attention forward
  │     └─ RWKV7Attention.forward
  │        fla/layers/rwkv7.py:222-352
  │
  ├─ Step 8A: 训练 / 长序列主 kernel
  │     └─ chunk_rwkv7 -> chunk_dplr_delta_rule
  │        fla/ops/rwkv7/chunk.py:13-81
  │        fla/ops/generalized_delta_rule/dplr/chunk.py:388-501
  │
  └─ Step 8B: 短序列推理 kernel
        └─ fused_mul_recurrent_rwkv7
           fla/ops/rwkv7/fused_recurrent.py:240-314
```

## 4.2 每一层做了什么

- **配置层**接收 Python kwargs，输出 `RWKV7Config` 对象；只在初始化时执行，不触发通信，不直接占 GPU 显存。
- **模型构造层**创建参数、norm、block，并执行 `post_init()`；只在初始化 / load 时执行。RWKV7 attention 初始化里会构造临时张量 `ddd/www/zigzag/linear`（`fla/layers/rwkv7.py:166-211`），属于初始化内存峰值。
- **CausalLM 层**接收 `input_ids/labels/shift_labels`，输出 `loss/logits/past_key_values`。如果 `fuse_linear_cross_entropy=True` 且有 labels，会不提前物化 logits，而在 loss 内融合线性层和 CE（`modeling_rwkv7.py:520-542`）。
- **Base model 层**接收 token 或 embedding，输出最后 hidden states。它在训练时默认关闭 cache：`use_cache = config.use_cache if not self.training else False`（`modeling_rwkv7.py:381`）。
- **Block 层**负责 residual/norm/attention/FFN 编排，每层都会执行。
- **Attention 层**每 step / 每层执行，产生大量中间张量，并根据 training 与长度选择 kernel。
- **Kernel 层**训练时触发 autograd function；短推理时触发 recurrent forward 并返回最终 state。

通信方面，标准 `RWKV7Model.forward` 不创建 process group，也不主动 all-gather / all-reduce。只有当用户直接调用底层 `chunk_dplr_delta_rule(..., cp_context=...)` 时，DPLR 的 context parallel 路径才触发通信；`RWKV7Attention.forward` 当前没有 `cp_context` 参数传入 `chunk_rwkv7`。

## 4.3 哪些逻辑不在主路径

| 看似相关的逻辑 | 为什么容易误解 | 正确理解 |
|---|---|---|
| `fla/ops/rwkv7/channel_mixing.py` | 文件名像 RWKV7 FFN fused op | 当前 `RWKV7FeedForward` 没调用它；它是导出 op / 测试路径 |
| `RWKV7(Goose).md` 中 `recurrent_rwkv7(head_first=True)` | 文档里有 recurrent 伪实现 | Python 主路径使用 `fla/ops/rwkv7/fused_recurrent.py`；且源码 wrapper 已拒绝旧 `head_first` 参数 |
| `fused_recurrent_rwkv7` | 名字像模型 decode 主路径 | 模型短推理实际调用 `fused_mul_recurrent_rwkv7`（`rwkv7.py:323`），不是 generic wrapper |
| `config.attn` 混合 attention | 配置层和构造层都支持 | forward 返回值契约当前不一致，不能当作稳定主路径 |
| `load_state_dict` 迁移 | 看起来处理所有旧权重 | override 在 `RWKV7Model` 上，且只扫描 `model.layers.*` 前缀；顶层 LM load 是否触发需谨慎看待 |

💡 小结

- RWKV7 的真实主路径是 HF model -> block -> `RWKV7Attention` -> chunk/recurrent kernel。
- `attn_mode`、channel mixing、generic recurrent wrapper 都容易被误读为主路径。
- 标准模型前向没有分布式通信；CP 通信属于底层 DPLR op 的可选路径，不由 `RWKV7Attention` 自动启用。

# 五、关键数据流 / 状态流 / shape 流程

## 5.1 Tensor shape 变化

以 `hidden_size = H * K`、`value_dim = H * V` 为例，主路径 shape 如下：

```text
原始输入:
  input_ids: [B, T]

Embedding:
  hidden_states: [B, T, hidden_size]

Token shift:
  delta:      [B, T, hidden_size]
  conv_state: [N, hidden_size]

六路混合:
  xr/xw/xk/xv/xa/xg: [B, T, hidden_size]

投影与 LoRA:
  r: [B, T, hidden_size]
  k: [B, T, hidden_size]
  v: [B, T, value_dim]
  w: [B, T, hidden_size]
  a: [B, T, hidden_size]
  g: [B, T, value_dim]

重排后:
  r/w/k/a/kk: [B, T, H, K]
  v:          [B, T, H, V]

DPLR recurrent/chunk 输出:
  o: [B, T, H, V]

GroupNorm + gate correction:
  o: [B, T, value_dim]

输出投影:
  o_proj(o): [B, T, hidden_size]
```

如果使用 varlen flatten：

```text
固定 batch:
  hidden_states: [B, T, D]

varlen flatten:
  hidden_states: [1, total_tokens, D]
  cu_seqlens:    [N + 1]
```

`token_shift` 对 varlen 的要求是 `x.dim()==3` 且 `x.shape[0]==1`（`fla/modules/token_shift.py:549-552`）；`chunk_dplr_delta_rule` 同样要求有 `cu_seqlens` 时 `q.shape[0] == 1`（`fla/ops/generalized_delta_rule/dplr/chunk.py:472-477`）。测试里模型 varlen 路径就是把 `input_ids.view(1, B*T)` 加 `cu_seqlens` 传入（`tests/models/test_modeling_base.py:61-66`）。

显存上，六路 `xr/xw/xk/xv/xa/xg` 是显著中间值：`fused_addcmul` 减少 kernel launch 和中间表达式写回，但输出仍然是六个完整张量（`fla/ops/rwkv7/fused_addcmul.py:226-257`）。

## 5.2 Rank / Mesh / Process Group 变化

标准 RWKV7 模型没有 rank / mesh 构建。相关分布式逻辑在 context parallel（CP）底层：

```text
用户直接调用:
  build_cp_context(cu_seqlens_global, group=dist.group.WORLD)
    -> FLACPContext(group, local cu_seqlens, first/last rank metadata)
  chunk_dplr_delta_rule(..., cp_context=context)
```

CP context 的构建在 `fla/ops/cp/context.py:61-172`：它根据 `world_size/rank` 把全局 token 区间切成每 rank 的 `[rank_start, rank_end)`，生成 local `cu_seqlens`、`is_first_rank`、`is_last_rank`、`pre_num_ranks`、`post_num_ranks`。DPLR chunk 在 `cp_context` 非空时会：

- forward 预处理跨 rank initial state（`fla/ops/generalized_delta_rule/dplr/chunk.py:77-89`）；
- backward 预处理跨 rank state gradient（`fla/ops/generalized_delta_rule/dplr/chunk.py:286-300`）。

真正通信发生在 `chunk_gated_delta_rule_fwd_h_pre_process` 和 backward 预处理里，它们调用 `all_gather_into_tensor`（`fla/ops/cp/chunk_delta_h.py:859`、`948`）。通信包装在 `fla/ops/cp/comm.py:19-41`，底层是 `dist.all_gather_into_tensor`。

但是要特别澄清：`RWKV7Attention.forward` 调用 `chunk_rwkv7` 时没有传 `cp_context`（`fla/layers/rwkv7.py:308-321`），`chunk_rwkv7` wrapper 本身签名也没有暴露 `cp_context`（`fla/ops/rwkv7/chunk.py:13-27`）。所以 CP 是 DPLR 底层能力与测试路径，不是 RWKV7 HF 模型的自动分布式主路径。

## 5.3 状态切换

RWKV7 的状态不是全局变量，而是 per-layer cache：

```text
进入 forward:
  past_key_values: Cache 或 None

Attention:
  last_state = get_layer_cache(self, past_key_values)
  conv_state = last_state['conv_state']
  recurrent_state = last_state['recurrent_state']

FFN:
  ffn_state = state[self.layer_idx]['ffn_state']

退出 attention/ffn:
  update_layer_cache(layer_idx, recurrent_state=..., conv_state=..., ffn_state=..., offset=...)
```

`get_layer_cache` / `update_layer_cache` 在 `fla/layers/utils.py:212-223`。Cache 的状态字典包含 `recurrent_state`、`attn_state`、`conv_state`、`ffn_state`（`fla/models/utils.py:60-66`）。对于 RWKV7 这类没有标准 attention KV 的层，seen tokens 通过 `offset` 增加（`fla/models/utils.py:121-129`）。

状态 shape：

```text
conv_state / ffn_state:
  [N, hidden_size]

recurrent_state:
  [N, H, K, V]
```

`N` 在普通 batch 下等于 `B`，在 varlen flatten 下等于 `len(cu_seqlens)-1`。

## 5.4 本章小结

💡 小结

- RWKV7 的 shape 主线是 `[B,T,D] -> 多路 [B,T,D] -> [B,T,H,K/V] -> [B,T,D]`。
- 标准模型没有自动 context parallel；CP 通信只在底层 DPLR op 显式传 `cp_context` 时发生。
- 生成状态集中在 FLA `Cache` 中，不靠全局 registry；但每层 recurrent state 的 `[N,H,K,V]` 会成为推理显存大头。

# 六、核心机制深挖

## 6.1 DPLR 适配层：RWKV7 为什么不自己写一套 chunk 主体

### 设计哲学与核心问题

`chunk_rwkv7` 看起来像 RWKV7 的核心 op，但它其实非常薄。真正的工程选择是：**复用 generalized DPLR delta rule 的 chunk 体系，而不是为 RWKV7 单独维护一套训练 kernel。**

这能减少重复实现，也让 CP、varlen、chunk backward 重计算等机制复用已有基础设施；代价是读源码时必须跨目录理解 `rwkv7 -> generalized_delta_rule/dplr` 的映射。

### 源码入口与关键对象

```text
fla/ops/rwkv7/chunk.py
  - chunk_rwkv7：参数改名与透传

fla/ops/generalized_delta_rule/dplr/chunk.py
  - chunk_dplr_delta_rule：DPLR autograd 封装
  - chunk_dplr_fwd：forward 多阶段调度

fla/ops/generalized_delta_rule/dplr/chunk_A_fwd.py
  - chunk_dplr_fwd_intra：块内 A 矩阵 / gate 预处理

fla/ops/generalized_delta_rule/dplr/wy_fast_fwd.py
  - prepare_wy_repr_fwd：准备 WY 表示

fla/ops/generalized_delta_rule/dplr/chunk_h_fwd.py
  - chunk_dplr_fwd_h：跨 chunk state 更新

fla/ops/generalized_delta_rule/dplr/chunk_o_fwd.py
  - chunk_dplr_fwd_o：组装输出
```

### 主流程拆解

`chunk_rwkv7` 的实现只有一层映射（`fla/ops/rwkv7/chunk.py:67-81`）：

```python
return chunk_dplr_delta_rule(
    q=r,
    k=k,
    v=v,
    a=a,
    b=b,
    gk=w,
    ...
)
```

在模型里传入的是：

```python
chunk_rwkv7(
    r=r, w=w, k=k, v=v,
    a=-kk,
    b=kk * a,
    safe_gate=True,
    chunk_size=64,
)
```

也就是说，RWKV7 的动态状态修正被翻译成 DPLR 的 `Diag(gk) + a b^T` 结构。DPLR forward 的阶段在 `chunk_dplr_fwd` 里非常清楚（`fla/ops/generalized_delta_rule/dplr/chunk.py:32-123`）：

```text
1. chunk_rwkv6_fwd_cumsum(gk) -> gi/ge
2. chunk_dplr_fwd_intra(q,k,a,b,gi,ge) -> A_ab/A_qk/A_ak/A_qb/qg/kg/ag/bg
3. prepare_wy_repr_fwd(ag,A_ab,A_ak,v) -> w/u/A_ab_inv
4. optional CP pre-process
5. chunk_dplr_fwd_h(kg,bg,v,w,u,gk) -> h/v_new/final_state
6. optional compress_h0 for backward
7. chunk_dplr_fwd_o(qg,v,v_new,A_qk,A_qb,h) -> o
```

训练 backward 默认走重计算以节省显存：如果 `disable_recompute=False`，forward 只保存较少输入；backward 中重新计算 `gi/ge/A.../w/u/h/v_new`（`fla/ops/generalized_delta_rule/dplr/chunk.py:231-271`）。注释直接写到“不这样 GPU memory will be exhausted”（`chunk.py:231-232`）。

### 关键细节与误区澄清

这里有一个容易误解的点：`fused_recurrent_dplr_delta_rule` 看起来也能训练，因为它是 autograd Function。但它的 backward 明确 `raise NotImplementedError`，并提示训练使用 `chunk_dplr_delta_rule`（`fla/ops/generalized_delta_rule/dplr/fused_recurrent.py:190-198`）。所以训练主路径必须是 chunk。

### 本章小结

💡 小结

- `chunk_rwkv7` 是适配层，核心训练机制来自 generalized DPLR delta rule。
- DPLR chunk 用多阶段中间表示换取长序列并行和 backward。
- backward 默认重计算，是典型“算力换显存”。

## 6.2 Recurrent 解码：为什么短序列要绕开 chunk

### 设计哲学与核心问题

训练时 chunk 化很划算，但单 token decode 时，chunk kernel 的固定开销和中间张量会变得不划算。RWKV7 因为有 recurrent state，本来就适合用 `[N,H,K,V]` 状态逐步更新。

### 源码入口与关键对象

```text
fla/ops/rwkv7/fused_recurrent.py
  - fused_recurrent_rwkv7_fwd_kernel：Triton recurrent forward kernel
  - fused_recurrent_rwkv7_fwd：分配输出与 final_state
  - fused_recurrent_rwkv7：generic DPLR wrapper
  - fused_mul_recurrent_rwkv7：模型实际短序列路径
```

### 主流程拆解

模型短序列 eval 路径调用的是 `fused_mul_recurrent_rwkv7`（`fla/layers/rwkv7.py:323-334`），而不是 generic `fused_recurrent_rwkv7`。原因在参数形态：模型已经有 `kk` 和 sigmoid 后的 `a`，所以专用 kernel 在内部构造：

```text
act_a = -kk
b = kk * a
```

Triton kernel 中的核心更新在 `fla/ops/rwkv7/fused_recurrent.py:88-129`：

```text
b_h: [BK, BV] float32 state tile

b_h = exp(w)[:, None] * b_h
    + b_b[:, None] * sum(b_act_a[:, None] * b_h, 0)[None, :]
b_h += k[:, None] * v[None, :]
o = sum(b_h * r[:, None], 0)
```

`fused_recurrent_rwkv7_fwd` 根据 `T == 1` 设置 `IS_DECODE`（`fused_recurrent.py:146-150`），kernel 内部对 decode 有单步 fast path（`fused_recurrent.py:88-103`）。如果 `output_final_state=True`，会分配 `ht` 为 `[N,H,K,V]` 的 float32 state（`fused_recurrent.py:151-156`）。

### 关键细节与误区澄清

**误区：`fused_recurrent_rwkv7` 是模型实际短 decode 调用。**

不是。模型调用 `fused_mul_recurrent_rwkv7`（`fla/layers/rwkv7.py:323`），它把 `kk/a` 形式的 RWKV7 参数直接传给专用 kernel。`fused_recurrent_rwkv7` 是 public wrapper，内部转给 generic `fused_recurrent_dplr_delta_rule`（`fla/ops/rwkv7/fused_recurrent.py:183-237`）。

**误区：recurrent path 可用于训练省显存。**

当前模型 forward 用 `self.training` 强制训练走 chunk（`fla/layers/rwkv7.py:305-308`）。同时 generic DPLR recurrent backward 不支持（`fla/ops/generalized_delta_rule/dplr/fused_recurrent.py:190-198`），专用 `fused_mul_recurrent_rwkv7` 也没有自定义 backward。

### 本章小结

💡 小结

- Recurrent path 是推理解码优化，不是训练替代路径。
- 专用 `fused_mul_recurrent_rwkv7` 避免物化 `a=-kk`、`b=kk*a`，更贴近模型当前变量形态。
- recurrent state 的 `[N,H,K,V]` 是推理缓存收益与显存成本的核心。

## 6.3 辅助 fused kernels：小算子为何值得单独融合

### 设计哲学与核心问题

RWKV7Attention 前向在进入 DPLR recurrence 前有许多逐元素和小矩阵操作。若全用 PyTorch 表达式，会产生更多 kernel launch 和中间写回。FLA 把其中几段拆成 fused op，解决的是**内存带宽和 launch 开销问题**。

### 源码入口与关键对象

```text
fla/ops/rwkv7/fused_addcmul.py
  - Rwkv7FusedAddcmul：六路 hidden + delta * x 融合

fla/ops/rwkv7/fused_k_update.py
  - KUpdateFunction / fused_k_rwkv7：key update 融合

fla/ops/rwkv7/gate_output_correction.py
  - GateOutputCorrection：输出修正 + gate 融合
```

### 主流程拆解

**六路 addcmul**：

`fused_addcmul_rwkv7` 对 `T==1` 回退 PyTorch，否则返回六个张量（`fla/ops/rwkv7/fused_addcmul.py:273-299`）：

```python
oxr = hidden + delta * xr
oxw = hidden + delta * xw
...
oxg = hidden + delta * xg
```

backward 分两段：`addcmul_bwd1` 算 `d_hidden/d_delta`（`fused_addcmul.py:174-205`），`addcmul_bwd2` 用 `torch.compile` 计算各 `x_*` 梯度规约（`fused_addcmul.py:208-216`）。是否启用 compile 由环境变量 `FLA_USE_COMPILE` 控制，默认 `1`，Python < 3.11 自动回退 identity decorator（`fused_addcmul.py:29-37`；文档 `ENVs.md:43`）。

**Key update**：

源码 reference 是 `k.addcmul(k * (a - 1), ka)`（`fla/ops/rwkv7/fused_k_update.py:18-19`）。Triton 版本按长度分 short / long kernel：`T <= 512` 走 short，否则走 long（`fused_k_update.py:222-264`）。decode `T==1` 回退 reference（`fused_k_update.py:353-356`）。

**Gate output correction**：

reference 公式在 `gate_output_correction_ref`（`fla/ops/rwkv7/gate_output_correction.py:15-34`）：

```text
correction = ((r * k * r_k).sum(-1, keepdim=True) * v).view(o.shape)
output = (o + correction) * g
```

自定义 autograd 在 `GateOutputCorrection` 中保存 `o/r/k/r_k/v/g`，forward/backward 都按 65536 token 分段启动 kernel（`gate_output_correction.py:216-248`）。

### 关键细节与误区澄清

这里最容易误解的是“融合就等于显存节省”。`fused_addcmul` 确实减少了多个 PyTorch op 的中间表达式和 launch，但它仍然输出六个完整 `[B,T,D]` 张量（`fused_addcmul.py:226-257`）。`gate_output_correction` backward 还会创建 float32 `grad_r_k`，shape 类似 `r`（`gate_output_correction.py:188-195`）。所以这些 fused op 更多是减少临时表达式和带宽浪费，不是让 RWKV7 前向变成低激活内存。

另外，代码 review 发现 long sequence 分段里存在边界风险：`fused_addcmul` kernel 用 `i_b * (T + T_OFFSET)` 作为 batch base（`fused_addcmul.py:68`、`146`），如果 `B>1` 且总长超过 65536，早期分段的 batch stride 可能不是原始序列长；`gate_output_correction` 分段时把 `T_SIZE` 传给 kernel，但 kernel 用全局 `t_idx < T` 判断 mask（`gate_output_correction.py:89-91`、`140-141`），offset 后分段可能被 mask 掉。测试没有覆盖这些组合。

### 本章小结

💡 小结

- fused 小算子主要优化 launch 与内存带宽，不消除所有大激活。
- `FLA_USE_COMPILE` 只影响 RWKV7 fused addcmul backward 的一段 torch.compile 规约。
- 长序列分段的全局/局部 offset 是 Triton 维护风险点。

## 6.4 保存、加载与转换：兼容旧权重还是主训练逻辑？

### 设计哲学与核心问题

RWKV7 的保存加载大体复用 HuggingFace `PreTrainedModel`。仓库额外做的是两个兼容动作：从官方 / 外部权重转换到 FLA 命名，以及从旧版 FLA attention 参数 `x_x` 迁移到新版 `x_r/x_w/...`。

### 源码入口与关键对象

```text
utils/convert_from_rwkv7.py
  - convert：读取 .pth，推断 config，翻译权重名，save_pretrained

fla/models/rwkv7/modeling_rwkv7.py
  - RWKV7Model.load_state_dict：迁移旧 x_x 参数
```

### 主流程拆解

转换脚本 `convert` 读取权重并从 shape 推断 config（`utils/convert_from_rwkv7.py:21-44`），然后构建模型（`convert_from_rwkv7.py:55-58`），通过 `translate_into_fla` 把官方名字映射到 FLA 名字（`convert_from_rwkv7.py:71-111`），最后 `model.save_pretrained(output, max_shard_size="1000GB")`（`convert_from_rwkv7.py:145-147`）。

`RWKV7Model.load_state_dict` 的迁移逻辑在 `modeling_rwkv7.py:313-361`。它扫描 `model.layers.{idx}.attn.x_x`，然后拆成：

```python
x_r = x_x[0].unsqueeze(0).unsqueeze(0)
x_w = x_x[1].unsqueeze(0).unsqueeze(0)
...
x_g = x_x[5].unsqueeze(0).unsqueeze(0)
```

对应 `modeling_rwkv7.py:338-350`。

### 关键细节与误区澄清

**误区：这个 load migration 覆盖所有加载入口。**

源码只在 `RWKV7Model` 上 override 了 `load_state_dict`（`modeling_rwkv7.py:313`）。`RWKV7ForCausalLM` 自身没有 override；而且扫描前缀只处理 `model.layers.`（`modeling_rwkv7.py:320-340`）。如果用户加载 base model checkpoint，常见 key 可能是 `layers.*`；如果顶层 LM 直接 load，是否进入嵌套 `self.model.load_state_dict` 不能简单假设。文章中只能说“源码提供了一个可见迁移路径”，不能宣称覆盖所有 HF load 场景。

**误区：转换脚本属于训练主路径。**

不是。`utils/convert_from_rwkv7.py` 是离线权重转换 CLI，不在模型 forward、train step 或 generate 中调用。

### 本章小结

💡 小结

- 标准保存加载主要继承 HF；RWKV7 专用逻辑是离线转换和旧参数迁移。
- `x_x` 迁移路径范围有限，尤其要注意 top-level LM 与 base model key 前缀差异。
- 转换脚本是初始化前工具，不是训练时 patch 或 hook。

# 七、显存、性能与通信分析

## 7.1 显存收益范围

RWKV7 的显存收益主要来自两处：序列历史不以 attention KV 全量保存，以及可选 fused linear cross entropy 避免 logits 物化。但它不是“所有激活都省”。

| 内容 | 是否节省 | 原因 |
|---|---|---|
| 参数 | ❌ | RWKV7 只是模型结构变化，没有参数 sharding |
| optimizer state | ❌ | 源码没有 optimizer 分片或 ZeRO/FSDP 集成 |
| attention KV cache | ✅ | 推理保存 recurrent state `[N,H,K,V]`、token-shift/FFN state，而不是每层完整历史 KV |
| 训练激活 | 部分 ✅ / 部分 ❌ | chunk DPLR backward 默认重计算节省部分中间状态；但前向仍产生六路混合、投影、DPLR 中间矩阵 |
| logits | 可选 ✅ | `fuse_linear_cross_entropy=True` 且有 labels 时不先算 logits（`modeling_rwkv7.py:520-540`） |
| 输入 batch | ❌ | mask 只乘零，不自动 unpad；varlen 需要用户显式 flatten |
| recurrent state | ❌ | `[N,H,K,V]` state 是推理显存大头，且通常 float32 final state |
| 中间 buffer | ❌ | `fused_k_update` backward 的 `dka_tmp` 是 float32 大张量（`fused_k_update.py:279-322`），gate correction backward 也有 float32 intermediate（`gate_output_correction.py:188-213`） |

真正的显存大头有三类：

1. **多路 token mix 激活**：`xr/xw/xk/xv/xa/xg` 六个 `[B,T,D]`。
2. **DPLR chunk 中间表示**：`A_qk/A_qb/A_ak/A_ab/qg/kg/ag/bg/w/u/h/v_new` 等；默认 backward 重计算正是为了不全存。
3. **推理 recurrent state**：每层 `[N,H,K,V]`。例如 `H=32,K=64,V=64` 时，每层每 batch state 元素数是 `32*64*64=131072`，float32 约 0.5MB；层数和 batch/varlen 序列数会线性放大。

## 7.2 通信开销

标准 RWKV7 HF 模型路径没有通信：不构建 process group，不 all-gather，不 all-reduce，不 broadcast。通信只出现在底层 CP 路径：

| 路径 | 通信类型 | 源码依据 | 是否 RWKV7 模型自动触发 |
|---|---|---|---|
| CP forward state 预处理 | `all_gather_into_tensor` | `fla/ops/cp/chunk_delta_h.py:859` | 否 |
| CP backward state gradient 预处理 | `all_gather_into_tensor` | `fla/ops/cp/chunk_delta_h.py:948` | 否 |
| CP context 构建 | 无通信，读取 rank/world size | `fla/ops/cp/context.py:70-73` | 否 |
| CP 测试聚合结果 | `dist.all_gather` | `tests/context_parallel/test_cp_dplr.py:260-287` | 测试路径 |

CP DPLR 测试说明了 rank 维度如何切序列：先在 rank0 准备全局张量并 broadcast 到所有 rank（测试代码 `tests/context_parallel/test_cp_dplr.py:130-137`），然后每个 rank 切本地 `[start_idx:end_idx]`（`test_cp_dplr.py:226-237`），调用 `chunk_dplr_delta_rule(cp_context=context)`（`test_cp_dplr.py:244-255`），最后 all-gather 输出和梯度（`test_cp_dplr.py:260-287`）。这证明 CP 能力存在，但不等于 `RWKV7ForCausalLM` 自动支持 sequence parallel。

## 7.3 性能取舍

RWKV7 的性能设计可以概括为：

- **用 chunk 并行换训练吞吐**：长序列训练不逐 token recurrent 扫描，而是 chunk DPLR 多阶段计算。
- **用重计算换显存**：DPLR backward 默认 recompute forward 中间量（`fla/ops/generalized_delta_rule/dplr/chunk.py:231-271`）。
- **用 recurrent state 换推理 KV cache**：短 decode 只更新 `[H,K,V]` 状态，不保留全历史 KV。
- **用 fused 小算子换带宽/launch**：addcmul、key update、gate correction 都减少 PyTorch op 链，但并不消除大张量输出。
- **用可选 fused linear CE 换 logits 显存**：配置层警告该路径可能降低精度（`configuration_rwkv7.py:100-105`；README 也提示 `fuse_linear_cross_entropy` 更省显存但可能影响数值稳定，`README.md:263-267`）。

瓶颈上，训练阶段更可能受显存带宽和中间 buffer 控制，而不是纯算力；推理阶段则取决于 recurrent state 大小、每层 kernel launch、以及短序列分流是否成功避开 chunk。

💡 小结

- RWKV7 节省的是 attention 历史表示和可选 logits，不是参数或 optimizer state。
- 标准模型没有通信；CP 是底层 DPLR 的显式 op 能力。
- 性能收益来自 chunk 并行、重计算、fused 小算子和 recurrent cache 的组合。

# 八、配置项、边界条件与坑点

配置不要只看默认值，要看它如何改变源码路径。

| 配置 / 条件 | 影响源码路径 | 行为变化 | 风险 / 坑点 |
|---|---|---|---|
| `attn_mode` | `modeling_rwkv7.py:153-154`，`rwkv7.py:305-334` | 初始化时传入 attention | forward 没按 `self.mode` 分支；训练仍 chunk，短 eval 才 recurrent |
| `head_dim` + `num_heads` | `configuration_rwkv7.py:58-62`，`rwkv7.py:60-68` | 决定 `H/K` | 默认 `head_dim=64` 可能让用户传入的 `num_heads` 被忽略 |
| `value_dim` | `configuration_rwkv7.py:63-77`，`rwkv7.py:68/303/349` | 允许 per-layer value 扩展 | gate correction kernel 按 `head_dim` 假设，`value_dim>hidden_size` 路径高风险且无测试覆盖 |
| `attn` | `configuration_rwkv7.py:107-117`，`modeling_rwkv7.py:141-151` | 部分层替换成标准 Attention | `RWKV7Block.forward` 固定四返回值，标准 Attention 返回三值，当前路径不稳定 |
| `fuse_norm` | `modeling_rwkv7.py:130-140`，`rwkv7.py:123-137` | 使用 FLA fused norm 或 PyTorch norm | 改变数值和 kernel 路径，需要分别测试 |
| `fuse_cross_entropy` | `modeling_rwkv7.py:523-542` | 使用 fused CE | 默认开启；和 `fuse_linear_cross_entropy` 不能同时开启（`configuration_rwkv7.py:96-99`） |
| `fuse_linear_cross_entropy` | `configuration_rwkv7.py:100-105`，`modeling_rwkv7.py:520-540` | 有 labels 时跳过 logits 物化 | 更省 logits 显存，但源码和 README 都提示可能降低精度 |
| `use_cache` + training | `modeling_rwkv7.py:381` | 训练时默认关闭 cache | 用户以为 config 开 cache 即训练缓存，这是错的 |
| `cu_seqlens` | `token_shift.py:549-552`，`chunk.py:472-477` | varlen flatten 路径 | 必须 `[1,total_T,D]`，不是普通 `[B,T,D]` 加 cu_seqlens |
| `attention_mask` | `rwkv7.py:233-245` | padding 乘零 | 不支持 3D mask，不自动 unpad |
| `num_hidden_layers=1` | `rwkv7.py:162-164` | 初始化 ratio 计算 | 除零，测试未覆盖 |
| `FLA_USE_COMPILE` | `fused_addcmul.py:29-37`，`ENVs.md:43` | 控制 addcmul backward 部分 torch.compile | Python < 3.11 自动回退；只影响此路径，不是全局 compile 开关 |

## 最小开启配置

最小使用方式：

```python
from fla.models import RWKV7Config
from transformers import AutoModelForCausalLM

config = RWKV7Config()
model = AutoModelForCausalLM.from_config(config)
```

如果要构造小模型，建议显式保持 `hidden_size == head_dim * num_heads`，并避免 `num_hidden_layers=1`，例如：

```python
config = RWKV7Config(hidden_size=256, head_dim=64, num_hidden_layers=2)
```

## 静默失效与不兼容组合

- `num_heads` 可能被默认 `head_dim` 静默覆盖。
- `attn_mode='fused_recurrent'` 不会让训练走 recurrent。
- `output_attentions=True` 被强制关闭。
- `value_dim>hidden_size` 配置层允许，但 gate correction 目前按 `head_dim` 写 kernel，存在严重 shape 风险。
- `attn` 混合层配置看似支持，但 forward 返回值契约不一致。
- 旧 `head_first` 参数在 `chunk_rwkv7` 和 recurrent wrapper 中会触发 deprecation warning / error（`fla/ops/rwkv7/chunk.py:63-66`，`fused_recurrent.py:284-287`）。

💡 小结

- RWKV7 配置的危险点主要在“看起来支持，但运行时契约不完整”。
- 推荐先使用默认 `value_dim == hidden_size`、纯 RWKV7 attention、`num_hidden_layers >= 2` 的路径。
- `cu_seqlens` 是 flatten varlen 接口，不是普通 padding mask 的替代开关。

# 九、测试、示例与覆盖缺口

## 9.1 已覆盖路径

| 测试 / 示例 | 覆盖的行为 | 说明 |
|---|---|---|
| `tests/models/test_modeling_rwkv7.py:19-39` | 模型 forward/backward | 覆盖 `L=4`、`B=4`、`T=1024`、bf16、`D=64/128` |
| `tests/models/test_modeling_base.py:61-67` | varlen flatten 与固定 batch 输出对齐 | `input_ids.view(1,B*T)` + `cu_seqlens` |
| `tests/models/test_modeling_rwkv7.py:46-63` | generation/cache | 调用通用 generation 测试 |
| `tests/models/test_modeling_base.py:73-133` | chunked cached decoding 对齐 full reference logits | 带 left padding attention mask |
| `tests/ops/test_rwkv7.py:80-137` | `fused_mul_recurrent_rwkv7` forward | 对比 generic DPLR recurrent |
| `tests/ops/test_rwkv7.py:140-230` | `fused_addcmul_rwkv7` forward/backward | 覆盖 `T=20/1024/4100/131072`，但 B=1 |
| `tests/ops/test_rwkv7.py:233-265` | `fused_k_rwkv7` forward/backward | 覆盖 `T=13/4096/8000` |
| `tests/ops/test_rwkv7.py:268-306` | `gate_output_correction` forward/backward | 覆盖 bf16、`T=4096` |
| `tests/context_parallel/test_cp_dplr.py` | DPLR CP forward/backward | 覆盖底层 DPLR CP，不是 HF RWKV7 模型自动路径 |
| `benchmarks/ops/registry.py:336-349` | benchmark 注册 `chunk_rwkv7` | 使用 `safe_gate=True`、`chunk_size=64` |

## 9.2 未覆盖风险

| 风险点 | 当前是否有测试 | 可能后果 |
|---|---|---|
| `value_dim > hidden_size` | 未见 RWKV7 模型测试 | gate correction shape/stride 错误，输出或梯度污染 |
| 混合 `attn` 层 | 未见 RWKV7 专项测试 | forward 解包失败 |
| `num_hidden_layers=1` | 未覆盖 | 初始化除零 |
| `num_heads` 与默认 `head_dim` 冲突 | 未覆盖 | 用户得到非预期 head 数 |
| `fused_addcmul` 的 `B>1,T>65536` | 未覆盖 | 长序列多 batch offset 错误 |
| `gate_output_correction` 的 `T>65536` | 未覆盖 | 后续分段 mask/输出错误 |
| `channel_mixing_rwkv7` 返回 last state shape | 测试只比较梯度，不 assert shape | 外部调用者拿到 `[T,D]` 而非 `[B,D]` |
| `RWKV7Model.load_state_dict` 旧权重迁移 | 未见专项测试 | base / LM checkpoint 迁移失败 |
| 多机 / 多节点 RWKV7 HF 模型 | 未见 | 标准模型没有 cp_context 注入，用户可能误以为支持 sequence parallel |
| 性能 / 显存收益 | benchmark 有 op 注册，但不是断言测试 | 回归不易被 CI 捕获 |

## 9.3 示例情况

仓库 README 给了通用 FLA model 初始化示例，但不是 RWKV7 专项示例（`README.md:164-172`）。`examples/` 下未发现 RWKV7 专用示例。权重转换脚本 `utils/convert_from_rwkv7.py` 是最接近实用入口的 RWKV7 脚本，但它服务 checkpoint 转换，不服务训练配置推荐。

💡 小结

- 已有测试证明了主算子、模型 forward/backward、generation 的基本可用性。
- 缺口集中在扩展 value 维度、混合 attention、极长序列分段、旧权重迁移和分布式 HF 集成。
- CP 测试证明 DPLR 底层能力，不等于 RWKV7 模型自动具备 CP。

# 十、局限性与已知优化点

## 10.1 硬约束

- `RWKV7Attention` 要求 `mode in ['chunk', 'fused_recurrent']`（`fla/layers/rwkv7.py:54-55`），但运行时不按 mode 分流。
- `head_dim` 或 `num_heads` 至少一个要给；默认给了 `head_dim=64`。
- `cu_seqlens` varlen 输入必须 flatten 成 batch size 1（`token_shift.py:549-552`，`dplr/chunk.py:472-477`）。
- `attention_mask` 只支持 `[B,T]` padding mask（`rwkv7.py:233-240`）。
- DPLR h kernel 要求 `BK <= 256`，且 `NK == 1`，否则不支持（`fla/ops/generalized_delta_rule/dplr/chunk_h_fwd.py:135-153`）。
- `num_hidden_layers=1` 当前初始化会除零。

## 10.2 维护成本

- RWKV7 模型依赖 generalized DPLR 目录下多阶段 kernel，读者和维护者必须跨模块追调用链。
- `attn` 混合 attention 复用标准 `Attention`，但 forward 返回契约不一致，说明抽象边界没有完全统一。
- `value_dim` 配置与 `gate_output_correction` kernel 的 head dim 假设不一致，属于配置层与 kernel 层脱节。
- `RWKV7(Goose).md` 中仍有旧 `head_first` 风格片段（`fla/ops/rwkv7/RWKV7(Goose).md:475-520`），而 Python wrapper 已移除 `head_first`，文档片段容易误导。
- 源码运行时 warning 明确提示实现可能与官方有差异（`fla/layers/rwkv7.py:148-153`），这本身就是维护风险信号。

## 10.3 性能瓶颈

- 每层 attention 前要生成六个混合张量，再做多路投影，内存带宽压力大。
- chunk DPLR forward / backward 有多阶段中间表示和重计算，复杂度高。
- `fused_k_update` backward 的 `dka_tmp` float32 大 buffer 与 gate correction backward 的 `grad_r_k` float32 intermediate 都可能成为显存峰值。
- 长序列分段存在全局/局部 offset 维护成本，且当前测试未覆盖关键组合。
- 推理 recurrent state 随层数、batch/sequence 数、`H*K*V` 线性增长。

## 10.4 已知优化点

- 修复或禁止 `value_dim != hidden_size`，直到 gate correction 支持独立 `head_v_dim`。
- 给 `RWKV7Block.forward` 增加 hybrid attention 分支，统一返回契约。
- 为 `fused_addcmul` 和 `gate_output_correction` 增加 `B>1,T>65536` / `T>65536` 测试并修正 offset。
- 对 `num_heads/head_dim` 增加一致性校验，避免静默覆盖。
- 明确 CP 是否要从 `RWKV7Attention.forward` 暴露；如果要支持 HF 模型级 CP，需要设计 `cp_context` 传递接口。
- 为 `load_state_dict` 迁移增加 base model 与 LM model 两条路径测试。

💡 小结

- 当前实现的核心风险不是“没有 kernel”，而是配置层、模型层、kernel 层之间的契约边界。
- 多数优化点都可以通过更严格 validation 和更针对性的测试先降低风险。
- 如果未来要做分布式长上下文训练，CP 需要从底层 op 能力升级为模型级显式接口。

# 小结与展望

FLA 的 RWKV7 实现可以用几个关键词概括。

**关键词一：HF 外壳，RWKV 内核。**  
外层是标准 `RWKV7Config -> AutoModelForCausalLM -> CausalLMOutputWithPast`，入口友好；内层则是 token shift、DPLR recurrence、gate correction 这些强 shape 约束的 kernel。

**关键词二：chunk/recurrent 双路径。**  
训练和长 prefill 走 `chunk_rwkv7 -> chunk_dplr_delta_rule`，短 eval/decode 走 `fused_mul_recurrent_rwkv7`。这不是由 `attn_mode` 单独决定，而是由 training 状态和 `seq_len` 决定。

**关键词三：融合小算子堆叠。**  
`fused_addcmul_rwkv7`、`fused_k_rwkv7`、`gate_output_correction` 把 RWKV7 前向的逐元素链路压到 Triton 中，减少 launch 和中间表达式，但不意味着激活显存完全消失。

**关键词四：重计算换显存。**  
DPLR chunk backward 默认重算大量 forward 中间量，这是训练长序列时的重要显存策略，也让 backward 调用链更复杂。

**关键词五：契约边界仍需加固。**  
`value_dim`、hybrid attention、单层模型、旧权重迁移、极长序列分段都暴露出“配置看似支持，kernel 或 forward contract 未完全跟上”的问题。

这个实现适合想在 FLA/HuggingFace 生态中试用 RWKV7、验证 chunk kernel 和 recurrent decode 路径的场景；不适合直接拿来作为未交叉验证的 benchmark 依据，也不适合在未补测试的情况下开启扩展 `value_dim`、混合 attention 或模型级 context parallel。

与 Transformer attention 相比，它用 recurrent state 和 DPLR chunk 机制换取长序列效率；与纯手写 RWKV 官方实现相比，它换来了 FLA kernel 复用和 HF 接口，但也引入跨模块抽象和维护成本。后续最值得继续走读的方向，是 `generalized_delta_rule/dplr` 的 chunk backward 细节、`fla.ops.cp` 如何把状态跨 rank 合并，以及 FLA 其他模型如何把类似 kernel 能力包装成稳定模型级 API。
