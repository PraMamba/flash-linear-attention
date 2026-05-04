# flash-linear-attention 源码走读：DeltaProduct implementation 实现解析

在长序列语言模型里，线性 RNN / 线性注意力常被期待解决 softmax attention 的二次复杂度问题。但把注意力替换成 recurrent state 之后，新的矛盾随之出现：状态读写是否足够表达复杂上下文？训练时是否还能并行？实现上是否会因为多次状态更新把显存和 kernel 调度成本又吃回来？

`flash-linear-attention`（下文简称 FLA）里的 DeltaProduct 实现，正是围绕这个矛盾展开的。README 把它标为 2025 年加入的 `DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products`（`README.md:51`、`README.md:93`），但源码里真正落地的名字不是 `DeltaProduct`，而是 `GatedDeltaProduct`：它既有 HuggingFace `AutoModel` 入口，也有独立 layer，还有底层 Triton op。

本文不展开 DeltaProduct 论文中的外部理论推导，也不讲 FSDP、DDP 或 HuggingFace `PreTrainedModel` 的基础用法。我们只沿着 FLA 源码回答一个问题：这个特性如何从用户配置一路走到 Triton kernel，它为了表达多次 Householder / delta update 做了哪些工程折中，又留下了哪些显存、性能和测试层面的坑。

# 前言

## 工程背景：DeltaProduct 在 FLA 中解决什么问题

FLA 是一个把多种线性注意力、线性 RNN、状态空间风格模型集成到 PyTorch / Triton / HuggingFace 生态里的仓库。DeltaProduct 在这里不是一个单独脚本，而是一条完整链路：

- 模型侧：`GatedDeltaProductConfig`、`GatedDeltaProductModel`、`GatedDeltaProductForCausalLM`；
- layer 侧：`GatedDeltaProduct`；
- op 侧：`chunk_gated_delta_product` 和对应 Triton kernel；
- 测试侧：算子正确性、varlen、模型 forward/backward、generation cache。

它解决的是线性 RNN 的**状态表达能力与训练并行效率之间的矛盾**：朴素 recurrent 写法容易表达“每个 token 做多次状态更新”，但训练长序列时不能逐 token Python 循环；chunk 并行写法能训练，却需要把多次状态更新、forget gate、intra-chunk readout 和 backward 组织成 GPU 友好的形式。

## 核心矛盾

DeltaProduct 的源码主线可以压缩成三句话：

1. 模型想在每个真实 token 上执行 `num_householder` 次状态更新，以增强线性 RNN 的 state tracking。
2. 训练不能真的按 Python recurrent loop 跑，所以源码把真实长度 `T` 展开成 `T * num_householder` 的 delta-rule 序列，再用 chunk / Triton kernel 做并行。
3. forward 已经有 DeltaProduct 专用 kernel，但 backward 仍复用 gated delta rule / delta rule 的 backward 路径，源码中明确留下 TODO，因此训练侧显存和性能并不是完全“专用优化”。

## 本文主线

本文按机制而不是按文件展开：

1. 用户入口与配置归一化：为什么 README 叫 DeltaProduct，源码却叫 GatedDeltaProduct；
2. 模型层装配：如何把 DeltaProduct 放进 HuggingFace block；
3. forward shape 流：`[B,T,D]` 如何变成 `T * num_householder`；
4. chunk forward kernel：如何把 recurrent update 变成并行计算；
5. backward、cache、varlen、通信与保存：哪些是主流程，哪些只是兼容路径；
6. 显存、性能、边界条件与测试缺口。

## 本文不展开的内容

本文不解释 DeltaProduct / Householder Products 的数学动机，不推导 WY representation，也不展开 HuggingFace `generate`、Triton autotune、RMSNorm 或 fused cross entropy 的通用原理。所有判断以本仓库源码为准；若文档、测试与源码不一致，以源码行为为准。

## 核心文件表

| 文件 | 职责 |
|---|---|
| `fla/models/gated_deltaproduct/configuration_gated_deltaproduct.py` | HF 配置入口，定义 `model_type` 与 DeltaProduct 专属字段 |
| `fla/models/gated_deltaproduct/__init__.py` | 注册 `AutoConfig` / `AutoModel` / `AutoModelForCausalLM` |
| `fla/models/gated_deltaproduct/modeling_gated_deltaproduct.py` | HF 模型、block、CausalLM wrapper 与 loss 路径 |
| `fla/layers/gated_deltaproduct.py` | 真正的 DeltaProduct layer：投影、shape 展开、cache、chunk/fused recurrent 分流 |
| `fla/ops/gated_delta_product/chunk.py` | `chunk_gated_delta_product` 公共 op 与自定义 autograd Function |
| `fla/ops/gated_delta_product/chunk_deltaproduct_h.py` | forward state / `v_new` / final state 的 Triton kernel wrapper |
| `fla/ops/gated_delta_product/chunk_deltaproduct_o.py` | output readout 的 Triton kernel wrapper |
| `fla/ops/gated_delta_product/chunk_ref.py` | reference 实现：把 DeltaProduct 展开成 delta rule 对照路径 |
| `fla/ops/gated_delta_product/naive.py` | 朴素 recurrent 语义参考，主要用于理解算法 |
| `tests/ops/test_gated_delta_product.py` | 带 forget gate 的 op 正确性与 varlen 测试 |

# 一、入口与配置归一化：README 里的 DeltaProduct 如何变成可执行模型

## 1.1 设计哲学与核心问题

读源码前最容易困惑的是命名：README 写的是 `DeltaProduct`，但项目里没有一个叫 `DeltaProductConfig` 的主类。真正用户入口是 `GatedDeltaProductConfig` 和 `GatedDeltaProductForCausalLM`。

这层存在的核心原因是生态接入：FLA 并不要求用户直接调用底层 Triton op，而是通过 HuggingFace 风格配置、AutoModel 注册和 CausalLM wrapper，让 DeltaProduct 可以像其他模型一样被 `from_config` / `from_pretrained` 使用。配置层解决的是**模型类型识别、默认超参、混合 attention 分流、loss fuse 开关**这些工程问题，而不是底层算法本身。

如果没有这一层，DeltaProduct 只能作为裸 op 被测试，无法自然进入训练、生成、保存加载和模型 zoo。

## 1.2 源码入口与关键对象

```text
fla/models/gated_deltaproduct/configuration_gated_deltaproduct.py
  - GatedDeltaProductConfig：定义 model_type、默认超参和 DeltaProduct 专属字段

fla/models/gated_deltaproduct/__init__.py
  - AutoConfig.register / AutoModel.register / AutoModelForCausalLM.register：HF 自动映射入口

fla/models/__init__.py
  - 对外导出 GatedDeltaProductConfig / Model / ForCausalLM

fla/__init__.py
  - 库级导出 GatedDeltaProduct layer 与模型类
```

关键证据：

- `GatedDeltaProductConfig.model_type = 'gated_deltaproduct'` 位于 `configuration_gated_deltaproduct.py:13-15`；
- HF 注册位于 `fla/models/gated_deltaproduct/__init__.py:13-15`；
- `fla/models/__init__.py:19` 导入该模型包，并在 `fla/models/__init__.py:72-74` 放入 `__all__`；
- `fla/__init__.py:17`、`fla/__init__.py:51-52`、`fla/__init__.py:118-120` 也暴露了 layer 与模型类。

## 1.3 主流程拆解

用户最自然的触发方式是 HuggingFace 风格 API：

```python
from transformers import AutoModelForCausalLM
from fla.models import GatedDeltaProductConfig

config = GatedDeltaProductConfig(
    num_hidden_layers=4,
    hidden_size=256,
    num_heads=4,
    head_dim=64,
    num_householder=2,
)
model = AutoModelForCausalLM.from_config(config)
```

对应调用链是：

```text
GatedDeltaProductConfig(...)
  -> model_type = 'gated_deltaproduct'
  -> fla.models.gated_deltaproduct.__init__ 注册 AutoModelForCausalLM
  -> AutoModelForCausalLM.from_config(config)
  -> GatedDeltaProductForCausalLM(config)
  -> GatedDeltaProductModel(config)
  -> GatedDeltaProductBlock(config, layer_idx)
  -> GatedDeltaProduct(...)
```

配置类本身的默认值在 `configuration_gated_deltaproduct.py:17-48`：`attn_mode='chunk'`、`head_dim=256`、`num_heads=6`、`hidden_size=2048`、`expand_v=2.0`、`use_short_conv=True`、`use_output_gate=True` 等。DeltaProduct 专属字段在 `configuration_gated_deltaproduct.py:88-91` 写入实例：

```python
self.allow_neg_eigval = allow_neg_eigval
self.num_householder = num_householder
self.use_forget_gate = use_forget_gate
```

第一处真正改变模型行为的位置不是 op，而是 block 初始化：`modeling_gated_deltaproduct.py:50-76`。如果 `config.attn` 指定当前层是普通 attention 层，就构造 `Attention`；否则才构造 `GatedDeltaProduct`。

这意味着“开启 DeltaProduct”不是某个环境变量或 monkey patch，而是模型类型和每层 block 的构造选择。

## 1.4 关键细节与误区澄清

**误区一：README 叫 DeltaProduct，所以应该有 `DeltaProductConfig`。**

源码中没有这个主入口。实际 `model_type` 是 `'gated_deltaproduct'`（`configuration_gated_deltaproduct.py:14`），类名是 `GatedDeltaProductConfig`、`GatedDeltaProductModel`、`GatedDeltaProductForCausalLM`。README 只是模型表层面的名称（`README.md:93`）。用户如果搜索 `DeltaProductConfig` 很可能找不到真正入口。

**误区二：配置默认值和直接 layer 默认值一致。**

不一致。配置默认是 `use_forget_gate=False`、`allow_neg_eigval=False`、`num_householder=1`（`configuration_gated_deltaproduct.py:46-48`）；但直接实例化 layer 时默认是 `use_forget_gate=True`、`allow_neg_eigval=True`、`num_householder=2`（`fla/layers/gated_deltaproduct.py:49-51`）。因此 `GatedDeltaProduct()` 和 `AutoModelForCausalLM.from_config(GatedDeltaProductConfig())` 的默认语义不同。

**误区三：`attn` 字段只是注释，不影响主流程。**

它会真的改变 block 类型。`modeling_gated_deltaproduct.py:50-76` 明确：如果 `config.attn is not None and layer_idx in config.attn['layers']`，该层走普通 `Attention`，不走 DeltaProduct layer。配置校验和补默认值在 `configuration_gated_deltaproduct.py:93-103`。

## 1.5 本章小结

💡 小结

- DeltaProduct 在 FLA 的真实入口是 `GatedDeltaProduct*`，不是裸 `DeltaProduct*`。
- 第一个行为分叉发生在模型 block 初始化：是否走普通 `Attention`，还是走 `GatedDeltaProduct`。
- 配置默认值与直接 layer 默认值不一致，源码走读和测试设计都必须区分这两条入口。

# 二、模型层装配：把 DeltaProduct 嵌进 CausalLM 的残差骨架

## 2.1 设计哲学与核心问题

DeltaProduct op 只解决“给定 q/k/v/g/beta 如何算输出”。但一个可训练语言模型还需要 embedding、block、MLP、norm、lm head、loss、cache、generation。模型层装配解决的是**如何把状态更新算子变成标准 causal LM**的问题。

这一层的工程矛盾在于：FLA 想复用 HuggingFace 生态，又要把非 attention 的 recurrent state 接进 `past_key_values`。因此模型代码尽量保持 Transformer-like block 结构，但 attention 子层换成 `GatedDeltaProduct`。

## 2.2 源码入口与关键对象

```text
fla/models/gated_deltaproduct/modeling_gated_deltaproduct.py
  - GatedDeltaProductBlock：RMSNorm -> DeltaProduct/Attention -> RMSNorm -> GatedMLP
  - GatedDeltaProductModel：embedding + blocks + final norm
  - GatedDeltaProductForCausalLM：lm_head、loss、generate 兼容
  - GatedDeltaProductPreTrainedModel：HF PreTrainedModel 基类与初始化逻辑
```

关键行号：

- block 定义：`modeling_gated_deltaproduct.py:41-117`；
- block 中选择 `Attention` 或 `GatedDeltaProduct`：`modeling_gated_deltaproduct.py:49-76`；
- model 组装 embedding / layers / norm：`modeling_gated_deltaproduct.py:174-190`；
- CausalLM wrapper：`modeling_gated_deltaproduct.py:265-378`；
- 初始化逻辑 `_init_weights`：`modeling_gated_deltaproduct.py:130-171`。

## 2.3 主流程拆解

模型 forward 的主路径可以画成：

```text
GatedDeltaProductForCausalLM.forward(input_ids, labels, ...)
  -> self.model(...)
      -> embeddings(input_ids)
      -> for layer in self.layers:
           GatedDeltaProductBlock.forward(hidden_states)
             -> attn_norm(hidden_states)
             -> self.attn(...)     # DeltaProduct 或 hybrid Attention
             -> mlp_norm(...)
             -> GatedMLP(...)
      -> final norm
  -> lm_head(hidden_states) 或 fused_linear_cross_entropy
  -> CausalLMOutputWithPast
```

`GatedDeltaProductModel.forward` 首先处理输入合法性：不能同时传 `input_ids` 和 `inputs_embeds`，也不能都不传（`modeling_gated_deltaproduct.py:218-225`）。然后在 `modeling_gated_deltaproduct.py:233-244` 遍历每层 block。

CausalLM wrapper 的关键逻辑在 `modeling_gated_deltaproduct.py:333-366`：先取 `outputs = self.model(...)`，再决定是否显式计算 logits。若 `fuse_linear_cross_entropy=True` 且有 labels，则走 `FusedLinearCrossEntropyLoss`，避免先完整 materialize logits；否则走 `lm_head` 再 `CrossEntropyLoss` / `FusedCrossEntropyLoss`。

这说明 DeltaProduct 本身不直接处理 vocab logits。它只输出 hidden states，后续训练显存是否受 logits 影响，取决于 CausalLM wrapper 的 loss 配置。

## 2.4 关键细节与误区澄清

**误区一：`output_attentions=True` 会返回 DeltaProduct 的 attention map。**

不会。`GatedDeltaProductModel.forward` 在 `output_attentions` 为真时直接 warning 并置 `False`（`modeling_gated_deltaproduct.py:210-212`）。DeltaProduct 是 recurrent state 读写，不提供标准 attention 矩阵输出。

**误区二：`generate()` 在这里实现了特殊 decoding 算法。**

`GatedDeltaProductForCausalLM.generate` 基本转发给父类，只在捕获含 `past_key_values` 的 `AttributeError` 时改写报错信息（`modeling_gated_deltaproduct.py:297-310`）。真正的 cache 更新发生在 layer 的 forward 中，而不是 `generate()` 函数里。

**误区三：保存加载需要 DeltaProduct 自定义 state_dict。**

目标源码中没有发现自定义 `save_pretrained`、`from_pretrained`、`state_dict` 或 monkey patch。模型继承 `PreTrainedModel`，配置类为 `GatedDeltaProductConfig`（`modeling_gated_deltaproduct.py:119-126`），保存加载应主要依赖 HuggingFace 默认机制。`A_log` 和 `dt_bias` 是普通 `nn.Parameter`，会随 state dict 保存；`_no_weight_decay` 只影响优化器分组，不是保存逻辑。

## 2.5 本章小结

💡 小结

- 模型层把 DeltaProduct 包装成标准 causal LM：embedding、block、MLP、lm head 和 loss 都在这一层完成。
- DeltaProduct 不产生 attention map，`output_attentions` 会被关闭。
- 保存加载没有专门 patch；主要依赖 HuggingFace `PreTrainedModel` 默认路径。

# 三、Layer 前向：从 `[B,T,D]` 到 `T * num_householder` 的状态更新序列

## 3.1 设计哲学与核心问题

DeltaProduct 的核心不是“多投影几个 k/v”这么简单。它的关键工程动作是：对每个真实 token 生成 `num_householder` 组 k/v/beta，让状态在同一个 token 内连续更新多次；但 q 只在这个 token 的最后一个子步读出。

因此 layer forward 要解决三件事：

1. 把 hidden states 投影成 q/k/v/beta/g；
2. 把 k/v/beta 展开成长度 `T * num_householder`；
3. 在训练时走 chunk op，在短序列推理时走 fused recurrent 兼容路径。

## 3.2 源码入口与关键对象

```text
fla/layers/gated_deltaproduct.py
  - GatedDeltaProduct.__init__：定义投影、forget gate 参数、短卷积、输出门
  - GatedDeltaProduct.forward：主 shape 流、mask/unpad、cache、chunk/fused_recurrent 分流

fla/layers/utils.py
  - get_unpad_data / index_first_axis / pad_input：padding <-> varlen 转换
  - get_layer_cache / update_layer_cache：按 layer_idx 读取和写回 recurrent/conv state
```

关键行号：

- layer 参数与投影：`gated_deltaproduct.py:35-153`；
- forward 输入与 mode 决策：`gated_deltaproduct.py:164-185`；
- unpad：`gated_deltaproduct.py:188-192`；
- q/k/v/beta reshape：`gated_deltaproduct.py:220-231`；
- chunk 调用：`gated_deltaproduct.py:238-250`；
- fused recurrent 分支：`gated_deltaproduct.py:252-272`；
- cache 写回：`gated_deltaproduct.py:274-280`；
- pad 回原 shape：`gated_deltaproduct.py:289-290`。

## 3.3 主流程拆解

先看 layer 初始化。`GatedDeltaProduct.__init__` 中 q/k/v/beta 的投影维度已经埋下了展开逻辑：

```python
self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
self.k_proj = nn.Linear(hidden_size, self.key_dim * num_householder, bias=False)
self.v_proj = nn.Linear(hidden_size, self.value_dim * num_householder, bias=False)
self.b_proj = nn.Linear(hidden_size, self.num_v_heads * num_householder, bias=False)
```

这段位于 `fla/layers/gated_deltaproduct.py:97-100`。它说明 q 只有一组，而 k/v/beta 每个 token 有 `num_householder` 组。

forward 中真实 shape 流如下：

```text
hidden_states: [B, T, hidden_size]

q_proj -> q:    [B, T, H*K]
k_proj -> k:    [B, T, N*H*K]
v_proj -> v:    [B, T, N*HV*V]
b_proj -> beta: [B, T, N*HV]

rearrange:
q:    [B, T, H, K]
k:    [B, T*N, H, K]
v:    [B, T*N, HV, V]
beta: [B, T*N, HV]
```

其中 `N = num_householder`。对应源码是 `gated_deltaproduct.py:220-231`。

如果 `num_v_heads > num_heads`，q/k 会通过 `repeat` 扩展到 value heads 数量（`gated_deltaproduct.py:224-225`）。如果 `allow_neg_eigval=True`，`beta = sigmoid(...) * 2`，把 beta 范围从 `(0,1)` 扩到 `(0,2)`（`gated_deltaproduct.py:227-230`）。如果开启 forget gate，则：

```python
g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)
```

位于 `gated_deltaproduct.py:232-235`。注意 `g` 的 shape 仍是 `[B,T,H]`，不是 `[B,T*N,H]`；真正 interleave 会在 op 内部完成。

mode 分流是 layer forward 的另一个关键点：

```python
mode = 'fused_recurrent' if (q_len <= 64 and not self.training) else self.mode
if self.training:
    assert mode == 'chunk', "Only chunk mode is supported in training."
```

这在 `gated_deltaproduct.py:181-185`。因此训练主流程只能是 chunk；短序列 eval / decoding 会自动进入 `fused_recurrent`。

## 3.4 关键细节与误区澄清

**误区一：`fused_recurrent` 是训练主路径。**

不是。训练时强制 `mode == 'chunk'`（`gated_deltaproduct.py:183-185`）。`fused_recurrent` 主要用于非训练且 `q_len <= 64` 的短序列推理路径，或用户在 eval 下显式设置 mode 后使用。

**误区二：attention mask 是任意 attention bias。**

不是。`attention_mask` 只能是二维 `[batch_size, seq_len]` padding mask；源码 assert 明确拒绝 `[batch, seq, seq]` 任意 mask（`gated_deltaproduct.py:173-178`）。它的作用是 unpad/pad，而不是改变 pairwise attention。

**误区三：传了 `cu_seqlens` 就等于自动从 padded batch unpad。**

不完全是。如果 `cu_seqlens` 已经在 kwargs 中，layer 不会调用 `get_unpad_data`，而是假定上游输入已经是 flatten 后的 varlen 形态。模型测试也是这样做的：`input_ids.view(1, B*T)` 搭配 `cu_seqlens`（`tests/models/test_modeling_base.py:61-66`）。

另一个边界风险是：如果同时传 `attention_mask` 和显式 `cu_seqlens`，`gated_deltaproduct.py:188-192` 不会定义 `indices`，但结尾 `attention_mask is not None` 时仍会 `pad_input(..., indices, ...)`（`gated_deltaproduct.py:289-290`）。这条组合在源码中未见防护，属于潜在异常路径。

## 3.5 本章小结

💡 小结

- DeltaProduct layer 的核心 shape 设计是 q 保持真实时间 `T`，k/v/beta 展开为 `T * num_householder`。
- 训练只走 chunk；`fused_recurrent` 是推理/短序列兼容路径。
- `attention_mask` 在这里是 padding/unpad 工具，不是通用 attention mask。

# 四、Chunk Forward：为什么要把 DeltaProduct 写成专用 Triton 前向

## 4.1 设计哲学与核心问题

朴素 recurrent 写法最容易理解，但训练长序列不可接受。reference 写法把 q/g 插入到 `T * num_householder` 序列，然后直接调用已有 delta rule；这很适合验证正确性，却不一定是最省的 forward。

FLA 的主 forward 选择折中：

- 复用 delta rule 中成熟的 `A`、`w/u` 计算；
- 专门写 DeltaProduct 的 state kernel 和 output kernel；
- 避免把每个真实 token 的中间 readout 都完整当作最终输出处理。

这一层解决的是**把 recurrent state update 并行化，同时控制 interleaved 展开带来的额外开销**。

## 4.2 源码入口与关键对象

```text
fla/ops/gated_delta_product/chunk.py
  - chunk_gated_delta_product：公共 API，做 dtype/shape/varlen 校验
  - ChunkGatedDeltaProductFunction：自定义 autograd Function
  - chunk_gated_delta_product_fwd：forward pipeline

fla/ops/gated_delta_product/chunk_deltaproduct_h.py
  - chunk_gated_delta_product_fwd_h：生成 chunk state、v_new、final_state

fla/ops/gated_delta_product/chunk_deltaproduct_o.py
  - chunk_gated_delta_product_fwd_o：读 state 并叠加 intra-chunk contribution
```

关键行号：

- 公共 API：`chunk.py:221-335`；
- autograd forward：`chunk.py:110-163`；
- forward pipeline：`chunk.py:24-107`；
- h kernel wrapper：`chunk_deltaproduct_h.py:406-455`；
- output kernel wrapper：`chunk_deltaproduct_o.py:122-159`。

## 4.3 主流程拆解

`chunk_gated_delta_product` 的公共 API 首先规定输入契约：

```text
q:    [B, T, H, K]
k:    [B, T*num_householder, H, K]
v:    [B, T*num_householder, H, V]
beta: [B, T*num_householder, H]
g:    [B, T, H] 或 None
```

shape assert 位于 `chunk.py:300-307`。如果有 varlen，batch size 必须是 1，且 initial_state 数量要等于序列数（`chunk.py:308-318`）。默认 scale 是 `K ** -0.5`（`chunk.py:319-320`）。然后调用 `ChunkGatedDeltaProductFunction.apply`（`chunk.py:321-334`）。

真正 forward pipeline 在 `chunk_gated_delta_product_fwd`：

```text
Step 1: gate interleave + chunk cumsum
  g: [B,T,H]
  -> g_interleaved: [B,T*N,H]

Step 2: A = beta * K K^T
  A: [B,T*N,HV,64]

Step 3: solve_tril(A)
  得到 chunk 内三角解

Step 4: recompute_w_u_fwd
  复用 gated/non-gated delta rule 的 w/u 表示

Step 5: chunk_gated_delta_product_fwd_h
  生成 h: [B,NT,H,K,V]
  生成 v_new: like u
  生成 final_state: [Nseq,H,K,V] 或 None

Step 6: chunk_gated_delta_product_fwd_o
  对真实 q 做 readout，输出 o: [B,T,H,V]
```

对应源码是 `chunk.py:38-107`。

有 forget gate 时，`g` 的 interleave 很特别：只有每个真实 token 的第 0 个 householder 子步写入 g，其余子步是 0（`chunk.py:39-47`）。这与 reference 路径一致：`chunk_ref.py:41-45` 也构造 `g_new[:, :, 0] = g`。

`A` 的 shape 来自通用函数 `chunk_scaled_dot_kkt_fwd`：它返回 `[B, T, HV, BT]`，其中 `BT=64`（`fla/ops/common/chunk_scaled_dot_kkt.py:81-120`）。在 DeltaProduct 主流程里，这里的 `T` 已经是展开后的 `T * num_householder`。

h kernel wrapper 会分配：

```python
h = k.new_empty(B, NT, H, K, V)
final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
v_new = torch.empty_like(u) if save_new_value else None
```

位于 `chunk_deltaproduct_h.py:432-434`。这三个张量决定了 forward 的主要中间显存形态。

output kernel 的逻辑分两部分：先用 q 读历史 chunk state（`chunk_deltaproduct_o.py:80-88`），再循环 `num_householder` 叠加当前 chunk 内 k/v 的局部贡献（`chunk_deltaproduct_o.py:102-117`），最后乘 scale 并写回 o（`chunk_deltaproduct_o.py:117-119`）。

## 4.4 关键细节与误区澄清

**误区一：主 forward 只是简单调用 `chunk_gated_delta_rule`。**

不是。reference 确实这么做：`chunk_ref.py:45-68` 把 q/g 展开后调用 `chunk_gated_delta_rule` 或 `chunk_delta_rule`，最后取每组最后一个输出。但主 forward 是 `chunk.py:24-107` 的专用 pipeline，并进入 `chunk_deltaproduct_h.py` / `chunk_deltaproduct_o.py` 的 Triton kernel。

**误区二：`num_householder` 只影响投影参数量，不影响时间维计算。**

它直接把 k/v/beta 的时间维扩大到 `T * num_householder`，`chunk.py:302-304` 明确 assert 这一点；h kernel 还要求 expanded `T` 能被 `num_householder` 解释回真实时间（`chunk_deltaproduct_h.py:420-421`）。因此它同时影响参数量、激活、中间 A、v_new、backward 展开成本。

**误区三：op 文档示例可以直接复制。**

`chunk.py:275-288` 的示例存在不一致：它从 `fla.ops.gated_delta_rule` import `chunk_gated_delta_product`，且示例 k/v/beta 形状仍像 `[B,T,H,*]`，但真实 API 要求 `k/v/beta` 是 `T * num_householder`，并且 `num_householder` 是必需参数（`chunk.py:221-228`、`chunk.py:302-304`）。这是文档片段与源码不一致，应以源码为准。

## 4.5 本章小结

💡 小结

- 主 forward 不是 reference 展开路径，而是“复用通用 delta-rule 前处理 + 专用 h/o Triton kernel”。
- `num_householder` 会真实扩大 k/v/beta 的时间维，直接影响显存与计算量。
- public op 的 docstring 示例存在 API 不一致，复制使用会踩坑。

# 五、完整主路径串联：一次真实用户调用会经过哪些层

## 5.1 完整调用栈

以一次训练 forward + loss 为例：

```text
User:
  config = GatedDeltaProductConfig(...)
  model = AutoModelForCausalLM.from_config(config)
  loss = model(input_ids, labels=labels).loss

  │
  ├─ Step 1: HF 注册与配置
  │     └─ GatedDeltaProductConfig.model_type = 'gated_deltaproduct'
  │     └─ AutoModelForCausalLM.register(...)
  │
  ├─ Step 2: 模型初始化
  │     └─ GatedDeltaProductForCausalLM.__init__
  │     └─ GatedDeltaProductModel.__init__
  │     └─ GatedDeltaProductBlock.__init__
  │     └─ GatedDeltaProduct.__init__
  │
  ├─ Step 3: CausalLM forward
  │     └─ GatedDeltaProductForCausalLM.forward
  │     └─ self.model(...)
  │
  ├─ Step 4: 多层 block forward
  │     └─ GatedDeltaProductModel.forward
  │     └─ for layer in self.layers
  │
  ├─ Step 5: DeltaProduct layer forward
  │     └─ q/k/v/beta/g projection
  │     └─ reshape k/v/beta to T * num_householder
  │     └─ chunk_gated_delta_product(...)
  │
  ├─ Step 6: Triton op forward
  │     └─ ChunkGatedDeltaProductFunction.forward
  │     └─ chunk_gated_delta_product_fwd
  │     └─ fwd_h kernel + fwd_o kernel
  │
  └─ Step 7: LM head / loss
        └─ lm_head + CE / fused CE / fused linear CE
```

## 5.2 每一层做了什么

| 层 | 输入 | 输出 | 状态修改 | 通信 | 显存影响 | 执行频率 |
|---|---|---|---|---|---|---|
| Config | Python kwargs / JSON | config object | 写 config 属性 | 无 | 无 | 初始化 / load |
| Model init | config | modules | 创建参数 | 无 | 参数显存 | 初始化 |
| CausalLM forward | input_ids / labels | loss/logits | 无直接 recurrent state | 无 | logits 或 fused loss 决定峰值 | 每 step |
| Model forward | embeddings | hidden_states | `past_key_values` 可能转换 | 无 | hidden states | 每层每 step |
| Layer forward | `[B,T,D]` | `[B,T,D]` | cache 读写 | 无 | q/k/v/beta/g、conv state | 每层每 step |
| Chunk op | q/k/v/g/beta | o/final_state | autograd ctx 保存 | 无 | A/h/v_new/saved tensors | 每层每 step |
| Backward | do/dht | dq/dk/dv/db/dg | 无参数外状态 | 无 | 展开 q/do 与复用 backward | 每层每 backward |

## 5.3 哪些逻辑不在主路径

- `chunk_gated_delta_product_ref`（`chunk_ref.py:16-70`）：测试/reference 路径，不是 layer 默认 forward。
- `naive_recurrent_gated_delta_product`（`naive.py:11-43`）：语义参考，测试中用于构造对照，不是训练路径。
- `fused_recurrent_gated_delta_rule` 分支（`gated_deltaproduct.py:252-272`）：主要是 eval / 短序列推理路径；训练时被 assert 排除。
- `config.attn` 命中的 `Attention` 分支（`modeling_gated_deltaproduct.py:50-60`）：混合模型路径，一旦命中，当前层不执行 DeltaProduct。
- `fla/ops/cp/*` 通信路径：仓库有 context parallel，但 `chunk_gated_delta_product` 签名没有 `cp_context` / `group` / `rank` 参数（`chunk.py:221-234`），layer 也只取 `cu_seqlens`（`gated_deltaproduct.py:188`）。不能把 CP DPLR 测试误认为 DeltaProduct 自动支持 CP。

## 5.4 本章小结

💡 小结

- DeltaProduct 主路径从 HF config 进入，最终落到 `ChunkGatedDeltaProductFunction.apply`。
- reference、naive、fused recurrent、hybrid attention 都不是标准训练主路径。
- 当前 DeltaProduct 主路径没有分布式 process group 通信，也没有 monkey patch。

# 六、关键数据流 / 状态流 / shape 流程

## 6.1 Tensor shape 变化

固定长度训练时：

```text
输入:
  input_ids:      [B, T]
  hidden_states:  [B, T, hidden]

投影后:
  q:     [B, T, H*K]
  k:     [B, T, N*H*K]
  v:     [B, T, N*HV*V]
  beta:  [B, T, N*HV]
  g:     [B, T, HV] 或 None

重排后:
  q:     [B, T, H, K]
  k:     [B, T*N, H, K]
  v:     [B, T*N, HV, V]
  beta:  [B, T*N, HV]
  g:     [B, T, HV]

op 内部:
  g_interleaved: [B, T*N, HV]
  A:             [B, T*N, HV, 64]
  h:             [B, ceil(T/64), HV, K, V]
  v_new:         [B, T*N, HV, V]
  final_state:   [B, HV, K, V]  # output_final_state=True 时

输出:
  o:             [B, T, HV, V]
  after norm:    [B, T, HV, V]
  flatten:       [B, T, HV*V]
  o_proj:        [B, T, hidden]
```

其中 `N = num_householder`。`h` 的 `NT` 是按真实 `T` 的 chunk 数，而不是展开后的 `T*N`，这是专用 forward kernel 相比完全展开输出的一处优化。

varlen / unpad 路径有两种：

```text
attention_mask 触发:
  hidden_states: [B,T,D]
  get_unpad_data -> indices, cu_seqlens
  hidden_states: [1,total_nnz,D]
  op 输出:       [1,total_nnz,D]
  pad_input ->   [B,T,D]

显式 cu_seqlens 触发:
  上游直接给 input_ids: [1,total_tokens]
  cu_seqlens: [Nseq+1]
  layer 不再自行 unpad
```

第一种路径证据是 `gated_deltaproduct.py:188-192` 与 `layers/utils.py:79-103`；pad 回来在 `gated_deltaproduct.py:289-290` 与 `layers/utils.py:181-202`。第二种路径由模型测试证明：`tests/models/test_modeling_base.py:61-66` 直接把 input reshape 成 `[1,B*T]` 并传 `cu_seqlens`。

## 6.2 Rank / Mesh / Process Group 变化

标准 DeltaProduct 路径中没有 rank / mesh / process group 变化。

源码依据：

- `chunk_gated_delta_product` 参数列表只有 tensor、`num_householder`、`scale`、`initial_state`、`output_final_state`、`use_qk_l2norm_in_kernel`、`cu_seqlens`、`cu_seqlens_cpu`（`chunk.py:221-234`）；
- `GatedDeltaProduct.forward` 只读取 `kwargs.get('cu_seqlens')`（`gated_deltaproduct.py:188`）；
- 仓库的 CP 通信工具确实存在，如 `all_gather_into_tensor` / `all_reduce_sum` 位于 `fla/ops/cp/comm.py:19-60`，但不在 DeltaProduct op 签名中；
- CP README 说明 CP 设计面向 GDN / KDA 等 delta-rule recurrent models（`fla/ops/cp/README.md:1-3`），未说明 DeltaProduct layer 自动接入。

所以如果 world size = 8，这个特性不会自动形成：

```text
SP group 0: rank0, rank1, rank2, rank3
SP group 1: rank4, rank5, rank6, rank7
```

也不会在 DeltaProduct forward 内触发 all-gather / reduce-scatter。多卡训练若使用 DDP/FSDP，那是 PyTorch 外层并行负责参数/梯度同步；DeltaProduct 算子自身不感知这个通信维度。

## 6.3 状态切换

DeltaProduct 没有全局 context manager 或 monkey patch，但有两类状态：

### 6.3.1 Cache state

```text
进入 layer forward:
  last_state = get_layer_cache(self, past_key_values)

执行中:
  recurrent_state = last_state['recurrent_state'] if last_state else None
  conv_state_q/k/v = last_state['conv_state'] if last_state else None

退出前:
  update_layer_cache(
      recurrent_state=recurrent_state,
      conv_state=(conv_state_q, conv_state_k, conv_state_v),
      offset=q_len,
  )
```

源码在 `gated_deltaproduct.py:186-187`、`gated_deltaproduct.py:193-214`、`gated_deltaproduct.py:237`、`gated_deltaproduct.py:274-280`。cache helper 要求使用 cache 时必须有 `layer_idx`，否则抛 `ValueError`（`layers/utils.py:205-223`）。对应测试在 `tests/layers/test_layer_cache_layer_idx.py:176-179`。

这类状态是对象级 / cache 对象级，不是进程全局状态。

### 6.3.2 Autograd ctx state

`ChunkGatedDeltaProductFunction.forward` 保存 backward 需要的张量：

```python
ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g_interleaved, beta, A, initial_state, cu_seqlens, chunk_indices_dp)
ctx.scale = scale
ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
ctx.num_householder = num_householder
```

位于 `chunk.py:158-161`。这解释了训练显存为什么不仅是 forward 中间值：许多张量会被保留到 backward。

## 6.4 本章小结

💡 小结

- 关键 shape 变化是 `T -> T * num_householder`，但最终输出仍回到真实 `T`。
- DeltaProduct 当前没有 rank/group/mesh 内部通信，只有单进程/单 rank 内的 varlen 处理。
- 状态主要存在于 cache 对象和 autograd ctx；没有全局 monkey patch 状态。

# 七、核心机制深挖

## 7.1 多 Householder 展开：为什么不能只多投影一次

朴素实现最能说明算法语义。`naive_recurrent_gated_delta_product` 在 `naive.py:30-42` 做了三步：

```python
for i in range(T):
    if g is not None:
        h = h * g[:, i, :].exp()[..., None, None]
    for j in range(num_householder):
        k_ij = k[:, i*num_householder+j, :, :]
        v_ij = v[:, i*num_householder+j, :, :]
        beta_ij = beta[:, i*num_householder+j, :]
        h = h + (v_ij - (h * k_ij[..., None]).sum(-2)).unsqueeze(-2) * k_ij[..., None] * beta_ij[..., None, None]
    o[:, i] = (h * q[:, i, :, :][..., None]).sum(-2)
```

这段说明：`num_householder` 不是多个 value head，也不是多组输出门，而是**同一个 token 内连续执行多次状态更新**。q 的 readout 在所有子更新之后发生。因此主实现必须保留“每个真实 token 最后一个子步输出”的语义。

reference 路径选择把 q 插到每组最后一个子步：`q_new[:, :, -1] = q`（`chunk_ref.py:37-39`），g 插到每组第一个子步：`g_new[:, :, 0] = g`（`chunk_ref.py:41-44`）。这就是 DeltaProduct 与 delta rule 复用的关键桥。

隐藏假设也很明确：`k/v/beta` 的时间维必须严格等于 `T * num_householder`，否则 public op 直接 assert（`chunk.py:302-304`）。

## 7.2 Backward：前向专用，反向复用

forward 已经有专用 h/o kernel，但 backward 做了一个工程上很务实的选择：把 q 和 do 扩展回 `T * num_householder`，然后调用已有 delta-rule backward。

源码在 `chunk.py:173-217`：

```python
q_new = q.new_zeros(q.shape[0], q.shape[1], ctx.num_householder, q.shape[2], q.shape[3])
q_new[:, :, -1] = q
do_new = do.new_zeros(do.shape[0], do.shape[1], ctx.num_householder, do.shape[2], do.shape[3])
do_new[:, :, -1] = do
q = rearrange(q_new, 'b t n h d -> b (t n) h d')
do = rearrange(do_new, 'b t n h d -> b (t n) h d')
# call the gated deltanet kernel for now.
# TODO: optimize the backward pass like the forward pass.
```

有 g 时调用 `chunk_gated_delta_rule_bwd`（`chunk.py:181-197`），无 g 时调用 `chunk_delta_rule_bwd`（`chunk.py:198-211`）。最后再把 `dq` 从展开序列中取每组最后一个子步（`chunk.py:213`）。

这不是 bug，而是清晰的实现取舍：复用成熟 backward 降低实现风险，但带来额外显存和计算。源码 TODO 说明维护者也知道 backward 尚未像 forward 一样专门优化。

## 7.3 QK L2Norm：省的是上游 materialization，不是取消归一化

public op 有 `use_qk_l2norm_in_kernel` 参数，docstring 写明“within the kernel for saving GPU memory”（`chunk.py:258-260`）。如果为 true，`ChunkGatedDeltaProductFunction.forward` 会调用 `l2norm_fwd(q)` 与 `l2norm_fwd(k)`，并保存 `q_rstd/k_rstd` 供 backward 使用（`chunk.py:130-134`、`chunk.py:214-216`）。

layer 主路径固定传 `use_qk_l2norm_in_kernel=True`（`gated_deltaproduct.py:248-250`）。测试则覆盖了两种情况：外部 `F.normalize` 与 kernel 内 l2norm（`tests/ops/test_gated_delta_product.py:64-75`）。

容易误解的是：它不是“不做归一化”，而是把归一化放到 op 内部，避免上游显式 `F.normalize(q/k)` 产生额外中间张量。

## 7.4 Chunk size 与 K 限制：硬件友好但边界隐藏

chunk size 基本固定为 64：`chunk_local_cumsum(... chunk_size=64 ...)` 在 `chunk.py:43-47`，h wrapper 默认 `chunk_size=64`（`chunk_deltaproduct_h.py:413`），o wrapper 默认 `chunk_size=64`（`chunk_deltaproduct_o.py:130`）。

head dimension 也有限制：h forward assert `K <= 256`（`chunk_deltaproduct_h.py:431`），backward dhu assert 同样限制（`chunk_deltaproduct_h.py:476`）。但 public config 默认 `head_dim=256` 只是默认值，并没有在配置构造时拒绝更大值（`configuration_gated_deltaproduct.py:21`），layer 构造也只接收 `head_dim`（`gated_deltaproduct.py:39`）。

这意味着某些错误会在模型构造后、第一次 forward 才暴露。

## 7.5 本章小结

💡 小结

- DeltaProduct 的算法语义是“每个真实 token 多次更新 state，最后一次读出”。
- forward 专用优化，backward 复用展开后的 delta-rule backward，是当前最重要的性能折中。
- `use_qk_l2norm_in_kernel` 是归一化位置的工程优化，不是取消归一化。
- `K <= 256` 和 chunk size 64 是底层 kernel 的硬约束 / 强假设。

# 八、显存、性能与通信分析

## 8.1 显存收益范围

| 内容 | 是否节省 | 原因 |
|---|---:|---|
| 参数 | ❌ | k/v/beta 投影随 `num_householder` 扩大，参数量反而增加 |
| optimizer state | ❌ | 参数增加会带来对应 optimizer state；源码没有 optimizer sharding 逻辑 |
| 激活 q/k/v/beta | ❌ / ⚠️ | k/v/beta 时间维是 `T*N`，激活更大；qk l2norm in kernel 可少 materialize 外部 normalize |
| recurrent state `h` | ✅ 相对朴素全展开 | h 存 `[B,ceil(T/64),H,K,V]`，不是每个展开子步都存完整 state |
| logits | ✅ 可选 | `fuse_linear_cross_entropy=True` 时 CausalLM 可不先生成完整 logits（`modeling_gated_deltaproduct.py:348-363`） |
| final_state | ❌ | `output_final_state=True` 时以 fp32 存 `[N,H,K,V]`（`chunk_deltaproduct_h.py:433`） |
| backward 保存 | ❌ | autograd 保存 q/k/v/g/beta/A 等（`chunk.py:158`） |

真正显存大头来自：

- `A: [B, T*N, HV, 64]`，由 `chunk_scaled_dot_kkt_fwd` 分配（`chunk_scaled_dot_kkt.py:114-120`）；
- `h: [B, NT, H, K, V]`（`chunk_deltaproduct_h.py:432`）；
- `v_new = torch.empty_like(u)`（`chunk_deltaproduct_h.py:434`）；
- backward 中 `q_new/do_new` 的 zero-expanded 张量（`chunk.py:173-178`）；
- autograd ctx 保存的多个 forward 张量（`chunk.py:158`）。

因此，DeltaProduct 的显存收益不是“整体比普通线性 RNN 少”，而是相对朴素 recurrent / 完全展开 reference，在 forward chunk state 存储上更有组织；但 `num_householder` 增大仍会明显放大 k/v/beta、A、v_new 和 backward 成本。

## 8.2 通信开销

标准 DeltaProduct op 每 step / 每 layer 内部通信次数：

```text
all_gather:      0
all_to_all:      0
reduce_scatter:  0
broadcast:       0
barrier:         0
```

这不是因为仓库没有通信能力。`fla/ops/cp/comm.py:19-60` 定义了 `all_gather_into_tensor` 和 `all_reduce_sum`，CP README 也描述了 sequence context parallel（`fla/ops/cp/README.md:1-3`）。但 DeltaProduct op 和 layer 没有接入这些参数。

训练中的分布式通信如果存在，来自外层 DDP/FSDP/optimizer，而不是 DeltaProduct implementation 自身。FSDP / DDP 不感知 `num_householder` 这个展开维度；它只看到普通 PyTorch module 参数和 autograd 图。

## 8.3 性能取舍

这个实现牺牲了什么，换来了什么？

1. **用投影和展开换状态表达能力**：k/v/beta 乘上 `num_householder`，每 token 多次更新 state。
2. **用 chunk + Triton 换训练并行**：避免 Python recurrent loop，forward 写专用 h/o kernel。
3. **用 backward 复用换开发稳定性**：反向没有专用 DeltaProduct kernel，而是展开后调用 delta-rule backward；源码 TODO 直接承认这是待优化点（`chunk.py:179-180`）。
4. **用 fused loss 可选降低 logits 显存**：这属于 CausalLM wrapper，不是 DeltaProduct op 本体。
5. **不用通信换显存**：当前没有 CP/TP 内建通信，因此不会产生 rank 间同步开销，也不能自动获得序列并行显存收益。

## 8.4 本章小结

💡 小结

- DeltaProduct 不是“免费增强状态能力”：`num_householder` 会放大投影、激活和 backward 成本。
- 当前实现没有 rank 间通信，性能瓶颈主要在单 GPU kernel、访存和 autograd 保存。
- forward 比 reference 更专用；backward 是最值得 profile 和优化的环节。

# 九、配置项、边界条件与坑点

## 9.1 配置如何改变源码路径

| 配置项 / 参数 | 影响源码路径 | 行为变化 | 风险 / 坑点 |
|---|---|---|---|
| `model_type='gated_deltaproduct'` | `configuration...py:14` + 注册 `__init__.py:13-15` | AutoModel 映射到 GatedDeltaProduct 模型 | README 名称与类名不一致 |
| `attn_mode='chunk'` | `gated_deltaproduct.py:181-185` | 训练主路径；训练只支持 chunk | 训练设置 `fused_recurrent` 会 assert |
| `num_householder` | `gated_deltaproduct.py:98-100,221-231`；`chunk.py:302-304` | k/v/beta 时间维变 `T*N` | 未校验 `>=1`；generation 测试误用复数字段 |
| `use_forget_gate` | `gated_deltaproduct.py:102-121,232-235` | 增加 A_log/dt_bias/a_proj，op 传 g | config 默认 False，layer 默认 True |
| `allow_neg_eigval` | `gated_deltaproduct.py:227-230` | beta 从 sigmoid 扩为 `2*sigmoid` | config 默认 False，layer 默认 True |
| `use_short_conv` | `gated_deltaproduct.py:123-147,193-218` | q/k/v 先过 ShortConvolution | 关闭会 warning；性能可能下降 |
| `use_output_gate` | `gated_deltaproduct.py:148-153,282-288` | 输出经 `FusedRMSNormGated` | 关闭走普通 RMSNorm |
| `fuse_linear_cross_entropy` | `configuration...py:77-86`；`modeling...py:348-363` | labels 存在时可直接 hidden->loss | 与 `fuse_cross_entropy` 不能同时 True |
| `attn` | `configuration...py:93-103`；`modeling...py:50-76` | 指定层走普通 Attention | 命中的层不会执行 DeltaProduct |
| `cu_seqlens` | `gated_deltaproduct.py:188`；`chunk.py:308-318` | varlen flatten 路径 | batch 必须为 1；与 attention_mask 同传有潜在 indices 问题 |

## 9.2 硬边界

- **dtype**：public op 只 assert `q.dtype != torch.float32`（`chunk.py:300`），提示用 bfloat16；测试主要覆盖 fp16/bf16。k/v/beta dtype 一致性没有像某些其他 op 那样完整校验，混 dtype 可能晚失败。
- **head_dim**：底层 h forward/backward 限制 `K <= 256`（`chunk_deltaproduct_h.py:431`、`chunk_deltaproduct_h.py:476`），但 config/layer 构造不提前拒绝。
- **varlen batch**：传 `cu_seqlens` 时 q batch size 必须为 1（`chunk.py:308-313`）。
- **initial_state 数量**：varlen 时 `initial_state.shape[0] == len(cu_seqlens)-1`（`chunk.py:314-318`）。
- **attention_mask**：只支持二维 padding mask（`gated_deltaproduct.py:173-178`）。
- **cache**：使用 `past_key_values` 时 layer 必须有 `layer_idx`，否则 helper 抛错（`layers/utils.py:205-223`）。
- **num_v_heads**：构造函数只在 `num_v_heads > num_heads` 且不能整除时拒绝（`gated_deltaproduct.py:85-88`）。如果 `num_v_heads < num_heads`，后续 q/k/beta/v head 数可能在 op shape assert 才失败。

## 9.3 静默失效与测试坑

最典型的是 generation 测试字段名：

- 测试参数叫 `num_householders`，并写 `config.num_householders = num_householders`（`tests/models/test_modeling_gated_deltaproduct.py:49-70`）；
- 配置类字段是 `num_householder`（`configuration_gated_deltaproduct.py:48,90`）；
- block 实际读取 `config.num_householder`（`modeling_gated_deltaproduct.py:74`）。

这意味着 generation 测试很可能没有真正覆盖 `num_householder=2/3`，而是仍用默认单数值。这个判断基于源码行为推断，未运行 GPU 测试确认。

另一个坑是 naive reference：optimized output kernel 会 `b_o = b_o * scale`（`chunk_deltaproduct_o.py:117`），但 `naive.py:40-43` 的输出没有乘 scale。测试中虽然构造了 `torch_ref`，却在后面再次 assert `ref` vs `tri`，没有直接比较 `torch_ref`（`tests/ops/test_delta_product.py:169-191`、`tests/ops/test_gated_delta_product.py:192-216`）。因此 naive 路径更适合读语义，不应被误认为完整数值 oracle。

## 9.4 本章小结

💡 小结

- 配置项不是简单开关：`attn` 会绕过 DeltaProduct，`num_householder` 会改变时间维，`fuse_linear_cross_entropy` 影响 logits 显存。
- 多个约束在底层 kernel 才暴露，尤其是 `K <= 256`、dtype、`num_v_heads` 组合。
- 测试里存在 `num_householders` 拼写风险和 naive oracle 不完整风险。

# 十、测试、示例与覆盖缺口

## 10.1 已覆盖路径

| 测试 / 示例 | 覆盖的行为 | 说明 |
|---|---|---|
| `tests/ops/test_delta_product.py:18-93` | 无 forget gate 的固定长度 op 前后向 | 覆盖 `num_householder=1/2/3`、非 2 次幂 T/D、l2norm 开关 |
| `tests/ops/test_delta_product.py:96-191` | 无 forget gate varlen | 覆盖 `cu_seqlens`、多段序列、与 ref 对齐 |
| `tests/ops/test_gated_delta_product.py:20-103` | 带 forget gate 的固定长度 op 前后向 | 覆盖 g、dg、mask_p、gate_logit_normalizer |
| `tests/ops/test_gated_delta_product.py:106-216` | 带 forget gate varlen | 覆盖空序列边界 `[...1500,1500]` 与 mask gate |
| `tests/models/test_modeling_gated_deltaproduct.py:22-42` | 模型 forward/backward | 通过 base helper 测固定长度与 flatten varlen 一致性 |
| `tests/models/test_modeling_base.py:53-67` | 模型 shape、varlen flatten、backward | 对 `GatedDeltaProductConfig` 生效 |
| `tests/models/test_modeling_gated_deltaproduct.py:48-73` | generation / cache | 覆盖 use_cache chunk+token decode 与无 cache logits 对齐，但 `num_householders` 拼写可疑 |
| `tests/layers/test_layer_cache_layer_idx.py:176-179` | cache 需要 layer_idx | 覆盖缺失 layer_idx 抛错 |

这些测试的优点是：算子级对 ref 做了 forward/backward 对齐，且包含非 2 次幂长度、varlen、gate、l2norm。对于 Triton kernel 来说，这是必要的基础覆盖。

## 10.2 skipped test 与硬件覆盖

- Intel Alchemist 且 `D > 128` 时，gated op 测试 skip（`tests/ops/test_gated_delta_product.py:51-52`、`tests/ops/test_gated_delta_product.py:129-130`），skip reason 仍写 `chunk_gated_delta_rule`，名称上有历史遗留感。
- 模型 base helper 在 Intel Alchemist 上跳过 modeling forward/backward（`tests/models/test_modeling_base.py:28-31`）。
- 非 Hopper 且 `D == 128` 时跳过部分 modeling 测试以节省 CI（`tests/models/test_modeling_base.py:46-47`）。

因此普通 CI 不一定覆盖所有高维和硬件路径。

## 10.3 未覆盖风险

| 风险点 | 当前是否有测试 | 可能后果 |
|---|---|---|
| `num_householder` generation 真正为 2/3 | ⚠️ 字段拼写疑似无效 | cache decode 对多 householder 覆盖不足 |
| `head_dim > 256` 报错信息 | 未见专项测试 | 用户运行时才遇到低层 assert |
| mixed dtype q/k/v/beta | 未见专项测试 | Triton 晚失败或错误信息不清晰 |
| `num_v_heads < num_heads` | 未见专项测试 | 构造成功，forward shape assert 失败 |
| `attention_mask` + 显式 `cu_seqlens` 同传 | 未见专项测试 | `indices` 未定义风险 |
| save_pretrained/from_pretrained roundtrip | 未见 DeltaProduct 专项测试 | HF 默认路径未被特性级确认 |
| context parallel / 多机 | 没有直接 DeltaProduct CP 测试 | 不能证明序列并行支持 |
| naive reference 与 scale | 测试未直接比较 naive | naive 不能作为完整 oracle |
| 性能 / 显存收益 | 未见 benchmark 断言 | TODO backward 的实际代价需 profile |

## 10.4 本章小结

💡 小结

- 算子正确性覆盖较扎实，尤其是固定长度、varlen、gate、梯度。
- 模型级测试覆盖 forward/backward/generation，但 generation 的 `num_householder` 覆盖有拼写风险。
- 保存加载、多机 CP、异常配置、性能显存都不是当前测试的强覆盖区域。

# 十一、局限性与已知优化点

## 11.1 硬约束

1. **训练只能 chunk**：`self.training` 时 assert `mode == 'chunk'`（`gated_deltaproduct.py:183-185`）。
2. **fp32 输入不支持**：public op assert `q.dtype != torch.float32`（`chunk.py:300`）。
3. **head_dim 不超过 256**：Triton h forward/backward assert `K <= 256`（`chunk_deltaproduct_h.py:431`、`chunk_deltaproduct_h.py:476`）。
4. **varlen batch 必须 flatten 到 batch=1**：`chunk.py:308-313`。
5. **`num_householder` 需要正整数但未提前校验**：源码使用其做 shape、乘法、模运算和除法（如 `chunk_deltaproduct_h.py:114`、`chunk_deltaproduct_h.py:420`），但 public API 未显式拒绝 0/负数。

## 11.2 维护成本

- **命名不一致**：README 的 DeltaProduct、源码的 GatedDeltaProduct、测试里的 `num_householders` 复数字段，增加使用和维护成本。
- **forward/backward 不对称**：forward 专用 kernel，backward 复用 delta-rule，未来优化需要同时维护两套语义映射。
- **docstring 过期**：`chunk.py:275-288` 示例 import 和 shape 与真实 API 不一致。
- **默认值不一致**：config 默认和 layer 默认不同，容易导致直接 layer 测试与模型行为不一致。
- **底层约束上浮不足**：`K <= 256`、dtype、`num_v_heads` 等在构造时不充分校验。

## 11.3 性能瓶颈

- **backward 展开 q/do**：`chunk.py:173-178` 生成 `[B,T,N,H,D]` 零张量，再 rearrange 到 `[B,T*N,H,D]`。
- **A 张量随 `T*N` 增长**：`A = [B,T*N,HV,64]`（`chunk_scaled_dot_kkt.py:114-120`）。
- **v_new 与保存张量**：`v_new = torch.empty_like(u)`（`chunk_deltaproduct_h.py:434`），ctx 保存多个大张量（`chunk.py:158`）。
- **没有 CP overlap 或通信切分**：DeltaProduct 主路径没有 group 参数，不能把长序列天然切到多 rank。
- **chunk size 固定 64**：硬编码带来硬件友好性，但也限制了不同序列/维度上的调优空间。

## 11.4 已知优化点

源码中最明确的 TODO 是 backward：`chunk.py:179-180` 写着 “TODO: optimize the backward pass like the forward pass.” 这说明后续优化方向非常清晰：写 DeltaProduct 专用 backward，避免 q/do 零展开和完整 delta-rule backward 复用。

其他可行优化点包括：

- 在 config/layer 构造阶段提前校验 `head_dim <= 256`、`num_householder >= 1`、`num_v_heads` 关系；
- 修正文档示例和 generation 测试字段名；
- 为 save/load roundtrip、异常配置、mixed dtype 添加测试；
- 若目标是长上下文多机训练，设计 DeltaProduct 专用 CP transition/merge，而不是假设现有 CP DPLR 测试自然覆盖；
- 对 `A/h/v_new` 和 backward saved tensors 做显存 profile，评估 checkpointing、分块重算或更细粒度 kernel fusion。

## 11.5 本章小结

💡 小结

- 当前最大已知优化点是 backward 专用化，源码 TODO 已直接指出。
- 维护风险主要来自命名/default/docstring/测试字段不一致。
- 若要支持真正长上下文分布式训练，需要新增 DeltaProduct-aware CP，而不是依赖现有 op 参数。

# 小结与展望

FLA 的 DeltaProduct implementation 可以用四个关键词概括。

**关键词一：GatedDeltaProduct 命名空间。**  
项目文档叫 DeltaProduct，但工程入口是 `GatedDeltaProductConfig`、`GatedDeltaProduct`、`GatedDeltaProductForCausalLM`。这种命名反映了它在 FLA 中不是孤立 op，而是模型、layer、kernel 三层集成。

**关键词二：`T * num_householder` 展开。**  
每个真实 token 生成多组 k/v/beta，连续更新 recurrent state，q 只在最后子步读出。这是实现 DeltaProduct 状态表达能力的核心 shape 设计。

**关键词三：forward 专用、backward 复用。**  
forward 通过 `chunk_gated_delta_product_fwd_h` 和 `chunk_gated_delta_product_fwd_o` 写了专用 Triton 路径；backward 目前把 q/do 展开后复用 gated delta rule / delta rule backward。它换来了实现稳定性，也留下训练显存和性能优化空间。

**关键词四：单机 varlen，而非内建 CP。**  
当前实现支持 `cu_seqlens` / unpad 的变长 batch，但没有 `process_group`、`cp_context` 或 rank-aware 通信参数。它适合单机/单 rank op 级高效训练和常规 DDP/FSDP 外层并行，不适合直接宣称已有 DeltaProduct 序列并行能力。

这个实现适合的场景，是希望在 FLA / HuggingFace 生态中快速实验 DeltaProduct 风格线性 RNN，并接受 chunk 训练、Triton kernel、fp16/bf16 和 `head_dim <= 256` 等约束的研究/训练流程。它不适合的场景，是要求开箱即用多机 context parallel、严格 fp32、任意 head_dim、或对 backward 显存极限敏感的生产训练。

与完全朴素 recurrent 相比，它用 chunk/Triton 换来了训练可行性；与完全为 DeltaProduct 定制的前后向相比，它又用 backward 复用换来了较低实现复杂度。后续最值得继续走读的方向有三个：一是 gated delta rule backward 如何被复用；二是 FLA 的 CP 模块如何为 GDN/KDA 做跨 rank state 同步；三是 fused linear cross entropy 如何在 CausalLM 层减少 logits 显存。读完这些，才能完整理解 FLA 在“状态表达能力、训练吞吐、显存压力、工程维护成本”之间的整体取舍。
