# FLA 源码走读：MoM implementation 实现解析

在长序列建模里，“线性时间的记忆机制”通常只解决了一半问题：单个 recurrent / linear attention 核心可以避免 softmax attention 的二次复杂度，但模型仍然要面对“所有 token 都挤进同一个记忆通道”带来的表达瓶颈。`flash-linear-attention`（下文简称 FLA）里的 MoM（Mixture-of-Memories）实现，试图把 MoE 的路由思想放进线性记忆层：每个 token 只进入 top-k 个 memory，再由 Gated Delta Rule 这样的线性记忆算子处理。

本文不展开 MoM 论文里的外部理论，也不把源码写成文件索引。我们只沿着 FLA 当前源码回答一个工程问题：**这个 MoM implementation 到底怎样接入 HuggingFace 模型栈，怎样把 token 分发到多个 memory，哪里真的省、哪里反而新增开销，以及哪些配置/测试信号说明它仍偏实验性。**

# 前言

MoM 在 FLA 中出现的位置很明确：README 把它列为 2025 模型，并指向 `fla/layers/mom.py`（`README.md:41`, `README.md:98`）。但真正读源码会发现，它不是一个“独立 MoM Triton kernel”，而是三层组合：

1. HuggingFace 风格的 `MomConfig` / `MomModel` / `MomForCausalLM`；
2. 核心层 `MomAttention`，负责 router、token-to-memory 重排、cache；
3. 底层复用 `chunk_gated_delta_rule` / `fused_recurrent_gated_delta_rule`。

这背后的核心矛盾是：**MoM 想让不同 token 进入不同 memory，从而提升线性记忆的容量；但 GPU kernel 往往喜欢规整、连续、等长的张量。于是源码主线变成了“先把 token 按 memory 打散，再把它们整理成 Gated Delta Rule 能接受的 ragged/varlen 格式，最后再散射回原序列”。**

本文会按机制展开：

- 入口与配置：用户如何开启 MoM，哪些配置真正改变路径；
- 初始化：router、per-memory projection、shared memory、短卷积和状态如何建立；
- 前向主链：top-k routing、`transform`、Gated Delta Rule、`reconstruct` 如何串起来；
- cache / generation / loss：状态如何保存，aux loss 如何计算，哪里有已知风险；
- 分布式、显存、性能、测试：主路径是否通信，显存收益来自哪里，测试证明了什么又没证明什么。

本文不讲 Gated Delta Rule 数学推导，不讲 MoE 理论，也不讲 FSDP / CP 的通用原理；只讲 FLA 当前源码怎么实现 MoM。

主线文件不多：

| 文件 | 职责 |
|---|---|
| `fla/models/mom/configuration_mom.py` | 定义 `MomConfig`、MoM 配置字段和部分 schema 校验 |
| `fla/models/mom/__init__.py` | 注册 HuggingFace AutoConfig / AutoModel / AutoModelForCausalLM |
| `fla/models/mom/modeling_mom.py` | `MomBlock`、`MomModel`、`MomForCausalLM` 和 aux load-balance loss |
| `fla/layers/mom.py` | MoM 核心层：router、token 重排、GDN 调用、cache 更新 |
| `fla/layers/utils.py` | unpad/pad 与 cache layer_idx 辅助函数 |
| `fla/models/utils.py` | FLA cache 容器与 generation mixin |
| `fla/ops/gated_delta_rule/chunk.py` | 训练主路径使用的 chunk GDN 自定义 autograd op |
| `fla/ops/gated_delta_rule/fused_recurrent.py` | 短序列推理路径使用的 recurrent GDN op |
| `tests/models/test_modeling_mom.py` | MoM 模型级测试，但当前整体 skip / known bugs |
| `tests/layers/test_layer_cache_layer_idx.py` | 通用 cache layer_idx 保护测试，包含 `MomAttention` case |

💡 小结

- MoM 在 FLA 中是“模型壳 + layer 编排 + GDN 后端”的组合，而不是独立 kernel。
- 主工程矛盾不是“怎么写一个 router”，而是“如何把动态 routing 结果整理成 GPU 友好的线性记忆输入”。
- README 已声明 MoM，但测试文件显示实现仍有 known bugs，需要谨慎解读成熟度。

# 一、入口与配置归一化：用户如何真正开启 MoM

## 1.1 设计哲学与核心问题

一个开源框架里的新模型特性，首先要解决的不是 kernel，而是“用户怎么触发它”。FLA 选择的是 HuggingFace 兼容路径：通过 `model_type='mom'` 注册 AutoClass，同时保留直接导入 `MomAttention` 的 layer 级入口。

这层解决的是**接入问题**：让 MoM 可以像其他 FLA 模型一样被 `AutoModelForCausalLM.from_config(...)` 创建，也可以作为一个 attention layer 被内部模型栈调用。

如果没有这层，`MomAttention` 只能是一个孤立模块，不能进入标准 Causal LM 的 embedding、block、loss、generation、save/load 链路。

## 1.2 源码入口与关键对象

```text
fla/models/mom/configuration_mom.py
  - MomConfig：定义 model_type='mom' 与 MoM 配置字段

fla/models/mom/__init__.py
  - AutoConfig.register / AutoModel.register / AutoModelForCausalLM.register：HF AutoClass 入口

fla/models/mom/modeling_mom.py
  - MomBlock：根据 config.attn 选择普通 Attention 或 MomAttention
  - MomModel：embedding + 多层 MomBlock + final norm
  - MomForCausalLM：LM head、CE loss、aux load-balance loss

fla/layers/mom.py
  - MomAttention：真正改变 attention 行为的核心 layer
```

关键源码证据：

- `MomConfig.model_type = 'mom'` 位于 `fla/models/mom/configuration_mom.py:11-13`。
- AutoClass 注册位于 `fla/models/mom/__init__.py:13-15`。
- `fla.models` 导出 `MomConfig/MomForCausalLM/MomModel` 位于 `fla/models/__init__.py:34`, `fla/models/__init__.py:111-113`。
- 顶层 `fla` 导出 `MomAttention/MomForCausalLM/MomModel` 位于 `fla/__init__.py:28`, `fla/__init__.py:75-76`, `fla/__init__.py:149-151`。

## 1.3 主流程拆解

用户最自然的入口是：

```python
from fla.models import MomConfig, MomForCausalLM

config = MomConfig(model_type="mom")
model = MomForCausalLM(config)
```

真实初始化链路是：

```text
MomConfig(...)
  -> MomForCausalLM.__init__                         modeling_mom.py:353
    -> self.model = MomModel(config)                 modeling_mom.py:355
      -> embeddings                                  modeling_mom.py:258
      -> layers = [MomBlock(config, layer_idx)]      modeling_mom.py:259
        -> MomBlock.__init__                         modeling_mom.py:124
          -> if config.attn 命中: Attention(...)      modeling_mom.py:129-137
          -> else: MomAttention(...)                 modeling_mom.py:139-156
      -> post_init()                                 modeling_mom.py:264
    -> lm_head + aux loss 参数                       modeling_mom.py:356-363
```

第一个真正改变行为的函数是 `MomBlock.__init__`：它决定一个 block 的 attention 是普通 softmax `Attention`，还是 MoM 的 `MomAttention`。默认 `config.attn is None`，所以走 `MomAttention` 分支。

配置流里最重要的字段在 `MomConfig.__init__`：

- `attn_mode`、`num_heads`、`head_dim`、`expand_v`、`use_output_gate`、`use_short_conv`：`configuration_mom.py:17-24`；
- `num_memories`、`topk`、`capacity`、`use_layer_wise_balance`、`aux_loss_scale`、`shared_mem`、`single_kv_proj`、`mom_backend`：`configuration_mom.py:38-45`；
- `fuse_cross_entropy` 等训练 loss 相关配置：`configuration_mom.py:46-48`。

这些字段被写入 `self` 的位置在 `configuration_mom.py:52-83`。随后 `MomBlock` 把其中一部分传给 `MomAttention`：`num_memories/topk/capacity/shared_mem/single_kv_proj` 分别在 `modeling_mom.py:151-155` 传入。

## 1.4 关键细节与误区澄清

这里有几个容易误解的点。

**误区一：`mom_backend` 看起来是后端插件机制。**

源码里它不是开放插件。`MomConfig` 只允许 `mom_backend in ['gated_deltanet']`，否则直接 `NotImplementedError`（`configuration_mom.py:85-86`）。`MomBlock` 内也只实现了 `gated_deltanet` 分支（`modeling_mom.py:139-158`）。所以当前 MoM 后端就是 Gated Delta Rule。

**误区二：`config.attn` 可以无缝混入普通 Attention。**

配置校验确实支持 `attn` dict，要求 `layers` 和 `num_heads`（`configuration_mom.py:88-96`），`MomBlock.__init__` 也会在命中层使用普通 `Attention`（`modeling_mom.py:129-137`）。但 `MomBlock.forward` 固定按四元组解包：

```python
hidden_states, attentions, past_key_values, router_logits = self.attn(...)
```

位置在 `modeling_mom.py:180`。普通 `Attention.forward` 只返回三元组 `o, attentions, past_key_values`（`fla/layers/attn.py:181`）。因此 hybrid attention 配置是“看起来支持、实际 forward 可能解包失败”的兼容路径，而不是可靠主路径。

**误区三：`from fla import MomConfig` 可以像 `from fla.models import MomConfig` 一样工作。**

顶层 `fla/__init__.py` 导出了 `MomForCausalLM` 和 `MomModel`，也导出了 `MomAttention`，但没有导出 `MomConfig`（对照 `fla/__init__.py:38-91`, `fla/__init__.py:149-151`）。正确入口是 `from fla.models import MomConfig`。

## 1.5 本章小结

💡 小结

- MoM 的用户入口是 HF AutoClass 注册 + `fla.models` 导出，真正进入 MoM 的节点是 `MomBlock.__init__`。
- `mom_backend` 当前不是可扩展后端机制，只支持 `gated_deltanet`。
- `config.attn` hybrid attention 是高风险兼容路径：初始化能走到普通 Attention，但 forward 解包与返回值不匹配。

# 二、模块初始化：Memory 是怎样被“造出来”的

## 2.1 设计哲学与核心问题

MoM 的核心不是简单多放几个线性层，而是要让每个 token 动态选择 memory。初始化阶段要同时准备三类对象：

1. router：决定 token 去哪些 memory；
2. memory-specific projections：每个 memory 如何把 token 投影成 Q/K/V/beta/decay；
3. recurrent/conv state 容器：推理缓存时要把每个 memory 的状态续上。

这层解决的是**状态与参数组织问题**。如果所有 memory 共享一套投影，表达力会下降；如果所有 memory 都独立投影，参数和 Python 编排开销又会上升。FLA 在 `shared_mem` 和 `single_kv_proj` 两个开关之间给了不同折中。

## 2.2 源码入口与关键对象

```text
fla/layers/mom.py
  - MomAttention.__init__：初始化 router、projection、A_log/dt_bias、ShortConvolution、输出门控
  - MomAttention._initialize_weights：layer 内部 Linear 权重初始化

fla/models/mom/modeling_mom.py
  - MomPreTrainedModel._init_weights：HF 模型级初始化逻辑
  - MomModel.__init__ / MomForCausalLM.__init__：post_init 与 LM head 初始化
```

关键字段：

- `self.gate`：router，`mom.py:334-335`；
- `self.k_proj/v_proj/b_proj/a_proj`：每 memory 一组投影，`mom.py:341-357`；
- `self.shared_k/shared_v/shared_b/shared_a`：共享 memory 分支或 single-kv 分支，`mom.py:336-340`, `mom.py:358-362`；
- `self.A_log` / `self.dt_bias`：GDN decay 参数，`mom.py:364-381`；
- `q_conv1d/k_conv1d/v_conv1d`：短卷积，`mom.py:383-402`；
- `g_proj/o_norm/o_proj`：输出门控与投影，`mom.py:408-413`。

## 2.3 主流程拆解

`MomAttention.__init__` 先保存 MoM 结构参数：

```text
num_memories/topk/capacity/shared_mem/single_kv_proj  -> mom.py:306-310
mode/hidden_size/expand_v/use_output_gate/use_short_conv -> mom.py:312-320
key_dim/value_dim/head_v_dim                          -> mom.py:325-328
```

随后创建 router：

```python
self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
self.gate = nn.Linear(self.hidden_size, self.num_memories, bias=False)
```

位置在 `fla/layers/mom.py:334-335`。这意味着 router logits 的 shape 是 `[batch, seq, num_memories]`。

memory 投影有两条路径：

```text
single_kv_proj=True:
  shared_k/shared_v/shared_b/shared_a                mom.py:336-340

single_kv_proj=False:
  k_proj/v_proj/b_proj/a_proj = ModuleList[num_memories]  mom.py:341-357
  if shared_mem=True:
    再额外创建 shared_k/shared_v/shared_b/shared_a       mom.py:358-362
```

默认 `MomConfig` 是 `single_kv_proj=False`, `shared_mem=True`（`configuration_mom.py:43-44`），所以模型默认既有每个 memory 独立投影，又有一条额外 shared memory 分支。

GDN 的 decay 参数也在初始化时建立：

- `A_log = log(uniform(0,16))`，并标记 `_no_weight_decay`：`mom.py:364-366`；
- `dt_bias` 用 `dt_min=0.001`、`dt_max=0.1` 的随机值做 softplus inverse：`mom.py:367-381`。

短卷积是强制性能路径。`use_short_conv=True` 时创建 Q/K/V 三个 `ShortConvolution`（`mom.py:383-402`）；如果关闭，代码不是 warning 后降级，而是 `raise UserWarning(...)`（`mom.py:403-407`）。

最后，输出门控：

```text
use_output_gate=True:
  g_proj + FusedRMSNormGated                         mom.py:408-410
else:
  RMSNorm                                            mom.py:411-412

o_proj                                               mom.py:413
```

## 2.4 关键细节与误区澄清

**误区一：`use_short_conv=False` 是普通可选项。**

配置里确实有 `use_short_conv: bool = True`（`configuration_mom.py:24`），但 layer 里关闭它会 `raise UserWarning`（`mom.py:403-407`）。这是异常路径，不是普通降级路径。文章读者如果只看 config，会误以为可以无痛关闭 short conv；源码并不支持。

**误区二：`capacity` 会在初始化后约束每个 memory 的 token 数。**

`capacity` 被保存为 `self.capacity`（`mom.py:300`, `mom.py:308`），也从 `MomConfig` 传入（`modeling_mom.py:153`）。但后续 `transform()` 中 capacity 相关的 `capacity_len`、padding/truncation 逻辑是注释（`mom.py:193-198`）。初始化阶段保存了字段，不代表主流程消费了字段。

**误区三：裸 `MomAttention` 默认行为等同 `MomConfig` 建出来的模型。**

二者默认不同。`MomConfig` 默认 `expand_v=1.0`、`num_memories=4`、`shared_mem=True`（`configuration_mom.py:22`, `configuration_mom.py:38`, `configuration_mom.py:43`）；而裸 `MomAttention` 默认 `expand_v=2`、`num_memories=8`、`shared_mem=False`（`mom.py:290`, `mom.py:298`, `mom.py:301`）。测试中直接构造 layer 时要特别注意这一点。

## 2.5 本章小结

💡 小结

- MoM 初始化同时建立 router、per-memory projection、可选 shared memory、GDN decay 参数和 short conv。
- 默认 `MomConfig(shared_mem=True)` 会多跑一条 shared memory 分支，这不是裸 `MomAttention` 的默认行为。
- `capacity` 和 `use_short_conv` 是两个最容易被配置名误导的字段：前者当前未消费，后者关闭会抛异常。

# 三、Forward 主链：从 token 到 memory，再回到 token

## 3.1 设计哲学与核心问题

MoM forward 最核心的工程问题是：router 的输出是动态的、每个 batch 每个 memory token 数都不同；但 Gated Delta Rule 期望输入是 `[B, T, H, D]` 或 varlen 的 `[1, total_tokens, H, D]` 形式。

因此 `MomAttention.forward` 的主线不是“算 attention”，而是四步变换：

```text
原始 token 序列
  -> top-k routing
  -> transform: 按 memory 分组、排序、gather、pad/unpad
  -> Gated Delta Rule 计算
  -> reconstruct: scatter 回原序列并按 routing weight 混合
```

这层解决的是**动态路由与静态 kernel 接口之间的张量布局矛盾**。

## 3.2 源码入口与关键对象

```text
fla/layers/mom.py
  - MomAttention.forward：MoM 主 forward
  - transform：token -> memory layout
  - _upad_input：memory layout -> ragged/varlen layout
  - reconstruct：memory output -> original token layout

fla/ops/gated_delta_rule/chunk.py
  - chunk_gated_delta_rule：训练主路径，有自定义 backward

fla/ops/gated_delta_rule/fused_recurrent.py
  - fused_recurrent_gated_delta_rule：短序列 eval 路径，backward 未实现
```

## 3.3 主流程拆解

### 3.3.1 输入、mask 与模式选择

`MomAttention.forward` 先处理 mask：如果 `attention_mask` 存在，会转成 bool，并断言必须是 `[batch_size, seq_len]` 的 2D padding mask，不允许任意 `[batch, seq, seq]` mask（`mom.py:434-440`）。

如果上游传入 `cu_seqlens`，MoM 先调用 `cu2pad` 把 varlen 输入转回 padded batch（`mom.py:442-445`），因为 router 和 transform 需要 `[B,S,H]` 形式。`cu2pad` 的实现位于 `mom.py:744-757`。

模式选择很关键：

```python
mode = 'fused_recurrent' if (hidden_states.shape[1] <= 64 and not self.training) else self.mode
if self.training:
    assert mode == 'chunk', "Only chunk mode is supported in training."
```

位置在 `mom.py:446-449`。这说明训练永远只能走 chunk；短序列 eval 会自动切到 fused recurrent。

### 3.3.2 Router：每个 token 选 top-k memory

核心代码在 `mom.py:453-467`：

```python
router_logits = self.gate(hidden_states)          # [B, S, M]
scores = F.softmax(router_logits, dim=2, dtype=torch.float)
routing_weights, selected_memories = torch.topk(scores, self.topk, dim=-1)
routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
routing_weights_full = torch.zeros(..., M).scatter(-1, selected_memories, routing_weights)
routing_mask = routing_weights_full.bool().int()
```

这里有两个输出：

- `selected_memories: [B,S,topk]`，决定每个 token 进入哪些 memory；
- `routing_weights: [B,S,topk]`，最终 reconstruct 时用来加权混合。

### 3.3.3 `transform`：把 token 按 memory 排队

`transform` 的 docstring 已经给了 shape 合同：输入 `x: (batch, seq, hidden)`，输出 `transformed_x: (num_memories, batch, max_len, hidden)`（`mom.py:120-144`）。

关键步骤：

1. top-k 展开：`x.repeat_interleave(topk, dim=1)`，把 `[B,S,H]` 变成 `[B,S*topk,H]`（`mom.py:145-153`）；
2. padding token 的 route 置零（`mom.py:155-159`）；
3. 用 `batch_indices * (memories_flat.max() + 1) + memories_flat` 构造排序 key，然后 `argsort`（`mom.py:163-175`）；
4. 统计每个 `(batch, memory)` 的 token 数，取最大值 `max_len`（`mom.py:177-188`）；
5. `torch.gather` 后 reshape 成 `[M,B,max_len,H]`（`mom.py:190-202`）。

这里的 `max_len` 是实际路由后最拥挤 memory 的长度。由于当前没有 capacity 裁剪，热点 memory 会直接决定整块 padded buffer 的长度。

### 3.3.4 Projection 与 varlen 化

进入 memory layout 后，先投影 Q/K/V/beta/g：

- `q = self.q_proj(hidden_states)`（`mom.py:477`）；
- `single_kv_proj=True` 时共享投影（`mom.py:478-482`）；
- 默认逐 memory 调用 `k_proj/v_proj/b_proj/a_proj` 并 `torch.stack`（`mom.py:483-488`）。

然后合并 memory 和 batch 维度：

```python
q, k, v, g, beta, mask_2 = (rearrange(x, 'e b l ... -> (e b) l ...') for x in (...))
```

位置在 `mom.py:490`。接着 `_upad_input` 去掉空 token，构造 `cu_seqlens`（`mom.py:491-493`）。底层 GDN 的 varlen API 要求 batch 维为 1，并通过 `cu_seqlens` 表示多段序列，这一点在 `chunk_gated_delta_rule` 文档和校验中也能看到：`chunk.py:441-449`, `chunk.py:478-488`。

### 3.3.5 ShortConvolution 与 GDN

短卷积对 Q/K/V 三路分别执行，且会处理 cache state（`mom.py:495-570`）。在 eval 且某些 routed sequence 长度小于 `conv_size` 时，还会先 `pad_for_conv` 再 `unpad_after_conv`（`mom.py:502-507`, `mom.py:568-569`, `mom.py:759-799`）。

GDN 调用分两条：

训练/长序列 chunk：

```python
chunk_gated_delta_rule(
    q=cu_q, k=cu_k, v=cu_v, g=cu_g, beta=cu_beta,
    initial_state=recurrent_state[0],
    output_final_state=use_cache,
    use_qk_l2norm_in_kernel=True,
    cu_seqlens=cu_seqlens,
)
```

位置在 `mom.py:578-589`。

短序列 eval recurrent：

```python
fused_recurrent_gated_delta_rule(...)
```

位置在 `mom.py:598-614`。

底层 GDN 的 shape 合同是：`q/k: [B,T,H,K]`，`v: [B,T,HV,V]`，`final_state: [N,HV,K,V]`（`chunk.py:371-421`, `fused_recurrent.py:331-381`）。

### 3.3.6 `reconstruct`：scatter 回原序列

GDN 输出后，MoM 做反向布局恢复：

```text
o.squeeze(0)
  -> pad_input(o, indices_q, batch_size*num_memories, max_len)       mom.py:623-624
  -> rearrange('(e b) l h d -> e b l (h d)')                         mom.py:625
  -> reconstruct(...)                                                mom.py:626-627
  -> rearrange('b l (h d) -> b l h d')                               mom.py:628
```

`reconstruct` 内部先 `scatter_add_` 回 `[B*S*topk,D]`，再用 `sorted_indices.argsort()` 恢复原 token 顺序，最后乘 routing weights 并对 top-k 求和（`mom.py:255-277`）。

这一步是 MoM 的“collect”阶段：它把 memory 维度的输出重新合成为原始 `[B,S,H]` 语义。

## 3.4 关键细节与误区澄清

**误区一：MoM 有自己的专用 Triton 算子。**

主路径没有 `mom` 专属 op 目录。`fla/layers/mom.py` 直接从 `fla.ops.gated_delta_rule` 导入 `chunk_gated_delta_rule` 和 `fused_recurrent_gated_delta_rule`（`mom.py:18-20`），调用点在 `mom.py:578-614`。MoM 的独特性主要在 router + layout 编排，而不是底层 recurrence kernel。

**误区二：`capacity` 会防止 memory 过载。**

`transform` 实际使用 `max_len = batch_memory_tokens.max()`（`mom.py:182`），并将所有 `(batch,memory)` pad 到这个 max。capacity 相关代码是注释（`mom.py:193-198`）。所以当前实现没有 token dropping，也没有 capacity clipping。

**误区三：短序列 fused recurrent 可以用于训练。**

MoM 训练时 assert 只能 chunk（`mom.py:447-449`）。底层 `FusedRecurrentFunction.backward` 直接 `raise NotImplementedError`（`fused_recurrent.py:302-309`）。所以 fused recurrent 是推理路径，不是训练路径。

## 3.5 本章小结

💡 小结

- MoM forward 的主线是 dispatch/collect：top-k dispatch 到 memory，GDN 计算后 scatter collect 回 token。
- 真正难点在动态 routing 后的排序、gather、unpad、pad、scatter，而不是单个矩阵乘。
- 当前没有 capacity 限制；routing 不均衡时，`max_len` 会把显存和时间开销重新拉高。

# 四、完整主路径串联：一次真实 Causal LM 调用发生了什么

## 4.1 设计哲学与核心问题

前面拆了入口、初始化、forward；现在把它们合成一次用户调用。完整主路径要回答：每一层接收什么、输出什么、是否通信、是否改状态、是否每 step 执行。

## 4.2 完整调用栈

```text
User:
  model = MomForCausalLM(MomConfig(...))
  out = model(input_ids, attention_mask=..., labels=...)

  │
  ├─ Step 1: 配置与注册
  │     ├─ MomConfig(model_type='mom')                         configuration_mom.py:11-13
  │     └─ AutoClass 注册                                      fla/models/mom/__init__.py:13-15
  │
  ├─ Step 2: 模型初始化
  │     ├─ MomForCausalLM.__init__                             modeling_mom.py:353-363
  │     ├─ MomModel.__init__                                   modeling_mom.py:253-264
  │     └─ MomBlock.__init__: Attention 或 MomAttention         modeling_mom.py:124-158
  │
  ├─ Step 3: Causal LM forward
  │     ├─ MomForCausalLM.forward -> self.model(...)            modeling_mom.py:418-428
  │     └─ logits/loss/aux_loss                                 modeling_mom.py:430-464
  │
  ├─ Step 4: Decoder stack
  │     ├─ MomModel.forward embedding                           modeling_mom.py:292-300
  │     ├─ Cache.from_legacy_cache                              modeling_mom.py:302-303
  │     └─ for layer in self.layers                             modeling_mom.py:309-324
  │
  ├─ Step 5: Block
  │     ├─ attn_norm                                            modeling_mom.py:177-180
  │     ├─ self.attn(...)                                       modeling_mom.py:180-187
  │     └─ MLP residual                                         modeling_mom.py:188-194
  │
  └─ Step 6: MomAttention
        ├─ mask / cu_seqlens / mode                            mom.py:434-449
        ├─ router top-k                                        mom.py:453-467
        ├─ transform + projection + unpad                       mom.py:473-493
        ├─ short conv                                           mom.py:495-570
        ├─ chunk/fused recurrent GDN                            mom.py:578-614
        ├─ reconstruct                                          mom.py:623-628
        ├─ optional shared_o                                    mom.py:630-633
        ├─ update_layer_cache                                   mom.py:635-641
        └─ output gate + o_proj                                 mom.py:643-655
```

## 4.3 每一层做了什么

- **配置层**：只在初始化前执行。写入 `num_memories/topk/shared_mem/single_kv_proj/...`，并校验 `mom_backend` 与 `attn` schema（`configuration_mom.py:85-96`）。不触发通信，不分配 GPU 激活。

- **模型初始化层**：创建 embedding、blocks、norm、lm head；`MomAttention.__init__` 创建 router 与 memory 参数。只执行一次。可能影响 checkpoint 结构，因为 `num_memories/shared_mem/single_kv_proj` 改变参数名和 shape。

- **`MomForCausalLM.forward`**：每 step 执行。调用 `self.model` 得到 hidden states，再根据 `fuse_cross_entropy` 决定是否创建 logits。训练时如果 `fuse_cross_entropy=True`，`fuse_linear_and_cross_entropy = self.config.fuse_cross_entropy and self.training`，logits 可为 `None`，loss 直接由 hidden states + `lm_head.weight` 计算（`modeling_mom.py:430-453`）。

- **`MomModel.forward`**：每 step 执行。负责 embedding、cache 格式兼容、逐层调用，收集 `router_logits`（`modeling_mom.py:302-340`）。没有分布式通信。

- **`MomBlock.forward`**：每层每 step 执行。做 prenorm、attention、MLP residual（`modeling_mom.py:168-198`）。状态修改发生在 attention 内。

- **`MomAttention.forward`**：每层每 step 执行，是显存和性能核心。它创建 routing logits、排序索引、gather/scatter buffer，调用 GDN kernel，写入 recurrent/conv cache。没有显式 all-gather/all-to-all/reduce-scatter。

- **loss 层**：如果 `labels is not None`，计算 CE loss 和 load-balancing aux loss。aux loss 使用所有层的 `router_logits`（`modeling_mom.py:455-464`）。

## 4.4 哪些逻辑不在主路径

- **CP / process group**：底层 `chunk_gated_delta_rule` 有 `cp_context` 参数（`chunk.py:399-402`, `chunk.py:470-476`），但 MoM 调用时没有传（`mom.py:579-589`）。所以 MoM 主路径不是 context-parallel aware。

- **save/load 自定义逻辑**：`MomPreTrainedModel` 继承 `PreTrainedModel`（`modeling_mom.py:201-204`），没有 MoM 专属 `save_pretrained`、`state_dict`、load hook。保存加载走 HF 标准机制。

- **monkey patch**：除了 `AutoConfig.register` / `AutoModel.register` 这种全局 registry 写入（`fla/models/mom/__init__.py:13-15`），未见 MoM 对 HF 类或函数做 monkey patch。

- **模型级测试主路径**：`tests/models/test_modeling_mom.py` 的 forward/backward 测试整体 skip（`test_modeling_mom.py:19`），generation 测试也 `Known bugs in mom`（`test_modeling_mom.py:63`）。它们不是当前可用性证据。

## 4.5 本章小结

💡 小结

- 一次 MoM Causal LM 调用的主路径非常集中：`MomForCausalLM -> MomModel -> MomBlock -> MomAttention`。
- 每 step 的新增开销主要在 `MomAttention.forward` 的 routing、layout transform、GDN、reconstruct。
- 分布式通信、save/load hook、monkey patch 都不是 MoM 当前主路径的一部分。

# 五、关键数据流 / 状态流 / shape 流程

## 5.1 设计哲学与核心问题

MoM 最容易读错的地方是 shape。它不是简单把 `[B,S,H]` 送进一个 attention kernel；中间会先展开 top-k，再按 memory 分组，再展平为 varlen，再恢复。

这一章只回答一个问题：**一个 token 在 MoM forward 中的 shape 身份如何变化？**

## 5.2 Tensor shape 变化

设：

```text
B = batch_size
S = seq_len
M = num_memories
K = topk
D = hidden_size
H = num_heads
Hd = head_dim
Vd = head_v_dim = int(head_dim * expand_v)
```

主 shape 流：

```text
原始输入:
  hidden_states: [B, S, D]

Router:
  router_logits:      [B, S, M]
  selected_memories:  [B, S, K]
  routing_weights:    [B, S, K]
  routing_mask:       [B, S, M]

Top-k 展开:
  x.repeat_interleave(K): [B, S*K, D]

按 memory 分组:
  transformed_x: [M, B, max_len, D]
  mask_2:        [M, B, max_len]

合并 memory 和 batch:
  q/k/v/g/beta before unpad: [(M*B), max_len, ...]

_unpad_input 后:
  cu_q/cu_k/cu_v: [1, total_active_tokens, H, Hd/Vd]
  cu_seqlens:     [N_active + 1]

Gated Delta Rule 输出:
  o: [1, total_active_tokens, H, Vd]

pad 回 memory layout:
  o: [M, B, max_len, H*Vd]

reconstruct 后:
  o: [B, S, H*Vd]

output gate + o_proj:
  final o: [B, S, D]
```

源码依据：

- `transform` 输入/输出 docstring：`mom.py:120-144`；
- top-k 展开：`mom.py:145-153`；
- `[M,B,max_len,D]` gather：`mom.py:190-202`；
- `rearrange('e b l ... -> (e b) l ...')`：`mom.py:490`；
- GDN q/k/v reshape：`mom.py:574`；
- reconstruct：`mom.py:623-628`, `mom.py:255-277`。

这一步真正可能节省的是每个 memory 内部的 effective sequence length：如果 routing 均匀，单个 memory 看到的长度约为 `S*K/M`。但注意 token 已经因 top-k 被复制，且最终 padded 到 `max_len`。如果 routing 不均衡，`max_len` 接近 `S*K` 的局部峰值，收益会明显下降。

## 5.3 Rank / Mesh / Process Group 变化

MoM 当前主路径没有 rank / mesh / process group 切换。全局搜索和调用点显示：

- `MomAttention.forward` 没有 `process_group` / `device_mesh` / `all_to_all` / `all_gather`；
- 调 `chunk_gated_delta_rule` 时没有传 `cp_context`（`mom.py:579-589`）；
- 底层 op 支持 `cp_context`，但那是 GDN 的能力，不是 MoM 接入的能力（`chunk.py:399-402`, `chunk.py:470-476`）。

因此一次 forward 中的“组”不是分布式通信组，而是本地张量里的 memory group：

```text
本地 batch 内:
  memory 0: 收到若干 token
  memory 1: 收到若干 token
  ...
  memory M-1: 收到若干 token

进 GDN 前:
  把 (memory, batch) 当作 varlen sequence 集合
  用 cu_seqlens 描述每个有效 memory-batch 段
```

如果用户期待 MoM 自动做 expert parallel / memory parallel，需要明确：当前源码没有这条路径。

## 5.4 状态切换

MoM 的状态不是全局变量，而是 per-layer cache：

```text
进入 MomAttention.forward:
  last_state = get_layer_cache(self, past_key_values)          mom.py:450

若无 cache:
  recurrent_state = [None for _ in range(1 + self.shared_mem)] mom.py:576-577

执行中:
  chunk/fused recurrent 返回 recurrent_state_                mom.py:579-614
  handle_recurrent_state 映射回完整 memory state               mom.py:590-621, 823-837

退出前:
  update_layer_cache(... recurrent_state, conv_state, offset)  mom.py:635-641
```

cache helper 在 `fla/layers/utils.py`：

- `get_layer_cache` 会要求传 cache 时 module 有 `layer_idx`（`layers/utils.py:205-216`）；
- `update_layer_cache` 调用 `past_key_values.update(layer_idx=layer_idx, **kwargs)`（`layers/utils.py:219-223`）。

通用 cache state 字段包括 `recurrent_state/attn_state/conv_state/ffn_state`（`fla/models/utils.py:60-66`），其中 recurrent 和 conv state 分别在 update 中写入（`fla/models/utils.py:68-69`, `fla/models/utils.py:95-97`）。

但这里有一个高风险点：MoM 更新 cache 时传的是 `offset=q.shape[2]`（`mom.py:635-641`）。此时 `q` 已在 `mom.py:490` 被重排为 `(e*b, l, key_dim)`，所以 `q.shape[2]` 更像 key_dim，而不是当前 token 数。对比同类 `GatedDeltaNet` 使用 `offset=q_len`（`gated_deltanet.py:298-303`）。这可能解释 generation 测试为何被标为 known bugs；这是基于源码行为的推断，未在本文中运行 GPU generation 复现。

## 5.5 本章小结

💡 小结

- MoM 的 shape 主线是 `[B,S,D] -> [M,B,max_len,D] -> varlen GDN -> [B,S,D]`。
- 当前没有 rank/process group 维度，只有本地 memory 分组；底层 CP 能力未被 MoM 主路径启用。
- cache 是 per-layer 状态，但 MoM 的 `offset=q.shape[2]` 很可能把 key_dim 当成 token 增量，是 generation 风险点。

# 六、核心机制深挖：路由、GDN 后端与状态缓存

## 6.1 Router：表达力来自 top-k，代价也来自 top-k

### 设计哲学与核心问题

Router 要解决的是“所有 token 共用同一个记忆状态”带来的容量问题。每个 token 选 top-k memory，理论上可以把不同模式的数据送到不同 recurrent memory。

### 源码实现

router 在 `MomAttention.forward` 中每 step 执行：`mom.py:453-467`。它先 softmax 全部 memory，再 top-k 离散选择。后续 `transform` 中大量索引构造放在 `torch.no_grad()` 下（`mom.py:163-188`），但 `routing_weights` 在 `reconstruct` 里参与加权（`mom.py:274-277`）。

因此可以作一个源码层面的判断：memory 选择本身不可微，top-k 权重路径可对被选中 memory 反传；训练时还有 aux load-balancing loss 使用所有 router logits（`modeling_mom.py:40-119`, `modeling_mom.py:455-464`）。

### 隐藏假设与风险

- 假设 `1 <= topk <= num_memories`。但 `MomConfig` 没有校验，非法值要到 `torch.topk` 才报错（`configuration_mom.py:39`, `mom.py:456`）。
- 假设 routing 不会极端不均衡。否则 `max_len = batch_memory_tokens.max()`（`mom.py:182`）会放大显存。

💡 小结

- Router 是 MoM 表达力来源，也是 top-k 复制、排序、scatter 的开销来源。
- 当前没有 topk 配置校验，也没有 capacity 限流，routing 不均衡会直接反映到 `max_len`。

## 6.2 GDN 后端：复用成熟 kernel，而不是重写 MoM kernel

### 设计哲学与核心问题

MoM 要的是 memory mixture；单个 memory 内部仍然需要高性能线性 recurrence。FLA 选择复用 Gated Delta Rule：MoM 负责把 token 整理成多个 varlen sequence，GDN 负责真正计算。

### 源码实现

- 导入：`fla/layers/mom.py:18-20`；
- chunk 调用：`mom.py:578-589`；
- fused recurrent 调用：`mom.py:598-614`。

训练主路径使用 `ChunkGatedDeltaRuleFunction`，它在 forward 中保存 q/k/v/g/beta/A/initial_state/cu_seqlens 等（`chunk.py:251-309`），backward 调 `chunk_gated_delta_rule_bwd` 返回梯度（`chunk.py:311-349`）。

fused recurrent 路径的 backward 未实现：`FusedRecurrentFunction.backward` 直接抛 `NotImplementedError`（`fused_recurrent.py:302-309`）。这和 MoM 训练强制 chunk 的 assert 是配套设计（`mom.py:446-449`）。

### 隐藏假设与风险

- MoM 传 `use_qk_l2norm_in_kernel=True`（`mom.py:587`, `mom.py:612`），底层 chunk op 会在 forward 中调用 `l2norm_fwd` 并保存 rstd（`chunk.py:275-278`）。这有助于减少 Python 侧显式 norm 编排，但不是零成本。
- GDN varlen API 要求 `q.shape[0] == 1` when `cu_seqlens is not None`（`chunk.py:478-483`），所以 MoM 必须把 memory/batch flatten 后再 unpad。

💡 小结

- MoM 的“算子创新”主要在调度和布局，核心 recurrence 复用 Gated Delta Rule。
- chunk 是训练路径；fused recurrent 是短序列推理路径，不能反传。
- `use_qk_l2norm_in_kernel=True` 把 q/k 归一化塞进 kernel，但会保存额外 rstd。

## 6.3 Cache 与 generation：状态续接是最脆弱的部分

### 设计哲学与核心问题

线性记忆模型推理时最重要的是 recurrent state 的正确续接。MoM 更复杂：一个 token 的 memory 路由是动态的，cache 里不仅要存 GDN recurrent state，还要存 short conv state，而且 shared memory 分支还多一份 state。

### 源码实现

- `last_state = get_layer_cache(self, past_key_values)`：`mom.py:450`；
- recurrent state 默认长度为 `1 + self.shared_mem`：`mom.py:576-577`；
- `prepare_recurrent_state` 选择 active memories（`mom.py:801-821`）；
- `handle_recurrent_state` 将 new state 写回原布局（`mom.py:823-837`）；
- `update_layer_cache` 写入 recurrent/conv state（`mom.py:635-641`）。

`MomForCausalLM.generate` 只做异常包装：如果 generation 策略操作 `past_key_values` 引起 AttributeError，会抛更清晰错误（`modeling_mom.py:383-397`）。它不是修复 cache 的逻辑。

### 隐藏假设与风险

- 裸 `MomAttention` 如果传 `past_key_values` 但没有 `layer_idx`，会被 `require_cache_layer_idx` 拦下（`layers/utils.py:205-208`）。测试覆盖了这个保护，包含 `MomAttention` case（`tests/layers/test_layer_cache_layer_idx.py:120-122`, `tests/layers/test_layer_cache_layer_idx.py:176-179`）。
- `offset=q.shape[2]` 风险如前所述，可能污染 seen tokens。
- MoM generation 测试明确 skip：`pytest.skip("Known bugs in mom")`（`tests/models/test_modeling_mom.py:63`）。

💡 小结

- MoM cache 同时维护 recurrent state 和 conv state，`shared_mem=True` 时 state 数量更多。
- cache 保护测试只证明“缺 layer_idx 会报错”，不证明 generation 正确。
- 当前 generation known bugs 与 cache offset 可疑点高度相关。

# 七、显存、性能与通信分析

## 7.1 设计哲学与核心问题

MoM 的收益不是“所有东西都省显存”。它用 top-k routing 降低每个 memory 的有效长度，但为了达成这个目标，引入了 token 复制、排序、gather/scatter、padded buffer 和可选 shared branch。

## 7.2 显存收益范围

| 内容 | 是否节省 | 原因 |
|---|---:|---|
| softmax attention KV 矩阵 | ✅ | 主路径不构造 `[B,H,S,S]` attention，使用 GDN 线性记忆 |
| 每个 memory 的序列长度 | 视 routing 而定 | 均衡时约为 `S*topk/num_memories`；不均衡时由 `max_len` 决定 |
| logits | ✅/❌ | 训练且 `fuse_cross_entropy=True` 时 logits 可不物化（`modeling_mom.py:430-453`）；推理或非 fused loss 仍需 logits |
| 参数 | ❌/视配置 | 默认每 memory 独立 K/V/b/a projection，还可能有 shared branch；`single_kv_proj=True` 才减少参数 |
| optimizer state | ❌ | 没有 optimizer state sharding 逻辑；参数越多 optimizer state 越多 |
| recurrent final state | ❌/新增 | `use_cache=True` 时要保存 `[N,HV,K,V]` 状态（GDN 文档 `chunk.py:388-421`） |
| routing 中间 buffer | ❌ | `repeat_interleave`、`argsort`、`gather`、`scatter_add_` 都会产生额外 buffer |
| shared memory 分支 | ❌ | 默认 `shared_mem=True` 会额外跑 shared GDN 分支（`mom.py:630-633`, `mom.py:685-736`） |

真正的大头取决于场景：

- 长序列训练中，避免 softmax attention 的 `[S,S]` 显存是主要收益；
- MoM 自身新增的 `[M,B,max_len,D]` buffer 和 scatter buffer 是主要代价；
- 如果 routing 不均衡或 `topk` 较大，收益会被 top-k 复制和 max padding 抵消；
- 如果 `shared_mem=True`，还要额外计算一条共享 GDN 路径。

## 7.3 通信开销

MoM 当前主路径没有显式分布式通信：

| 通信类型 | MoM 主路径是否触发 | 源码依据 |
|---|---:|---|
| all-gather | ❌ | `fla/layers/mom.py` 无相关调用 |
| all-to-all | ❌ | 无 expert parallel / token exchange |
| reduce-scatter | ❌ | 无梯度通信逻辑 |
| broadcast | ❌ | 无 rank0 load/broadcast 特殊逻辑 |
| barrier | ❌ | MoM 层无 barrier |
| CP context | ❌（主路径未接） | 底层 `chunk_gated_delta_rule` 支持 `cp_context`，但 MoM 调用未传（`mom.py:579-589`, `chunk.py:399-402`） |

所以 MoM 的“通信”主要是单卡/单进程内的 tensor 数据搬运：sort、gather、pad、scatter，而不是跨 rank 通信。

## 7.4 性能取舍

主要性能开销来自：

- `F.softmax` + `torch.topk`（`mom.py:453-457`）；
- `repeat_interleave(topk)`（`mom.py:145-153`）；
- `combined.argsort()`（`mom.py:172-173`）；
- `torch.gather`（`mom.py:190-191`）；
- 每 memory Python list comprehension 投影并 `torch.stack`（`mom.py:483-488`）；
- short conv 三路调用（`mom.py:515-560`）；
- `scatter_add_` reconstruct（`mom.py:265-277`）；
- varlen/pad/state remap 中的 Python for-loop（`mom.py:744-799`, `mom.py:811-837`）。

换来的收益是：每个 memory 内部可用 GDN 线性时间处理较短子序列，并避免 softmax attention 的二次复杂度。换句话说，这是典型的**用路由和数据重排开销，换线性记忆容量与长序列可扩展性**。

## 7.5 本章小结

💡 小结

- MoM 不是无条件省显存；它省的是 softmax attention 和均衡 routing 下的 per-memory 长度。
- 新增开销集中在 top-k 复制、排序、gather/scatter、short conv、shared branch。
- 当前没有跨 rank 通信，底层 CP 能力没有接到 MoM 主路径。

# 八、配置项、边界条件与坑点

## 8.1 设计哲学与核心问题

配置项不是表格里的“参数说明”，而是源码路径选择器。MoM 的几个配置会改变模块结构、forward 分支、loss 行为；也有一些字段存在但当前没有主流程消费。

## 8.2 配置如何改变源码路径

| 配置项 | 影响源码路径 | 行为变化 | 风险 / 坑点 |
|---|---|---|---|
| `mom_backend` | `configuration_mom.py:85-86`, `modeling_mom.py:139-158` | 只允许 `gated_deltanet` | 不是可插拔后端 |
| `attn` | `configuration_mom.py:88-96`, `modeling_mom.py:129-137` | 指定层走普通 `Attention` | forward 固定四元组解包，普通 Attention 返回三元组，可能报错 |
| `attn_mode` | `mom.py:446-449`, `mom.py:578-614` | chunk 或 fused recurrent | 训练强制 chunk；短序列 eval 自动 fused recurrent |
| `num_memories` | `mom.py:334-357`, `mom.py:453-467` | memory 数量与 projection 数量 | 改变参数结构，checkpoint 互载风险 |
| `topk` | `mom.py:456`, `mom.py:145-153` | 每 token 路由到 K 个 memory | 无校验；`topk > num_memories` 或 `topk=0` 到 forward 才炸 |
| `capacity` | `configuration_mom.py:40,73`, `mom.py:300,308` | 当前未实际限流 | capacity 相关逻辑是注释（`mom.py:193-198`） |
| `shared_mem` | `mom.py:358-362`, `mom.py:630-633` | 额外 shared GDN 分支 | 默认模型开启，额外算力/显存；改变 state 长度 |
| `single_kv_proj` | `mom.py:336-340`, `mom.py:478-482` | 共享 K/V/b/a projection | 减参数但不消除 routing 重排开销 |
| `use_short_conv` | `mom.py:383-407` | 创建三路 ShortConvolution | False 会抛 `UserWarning` 异常 |
| `use_output_gate` | `mom.py:408-412`, `mom.py:643-649` | gated RMSNorm 或普通 RMSNorm | 影响输出投影前归一化路径 |
| `fuse_cross_entropy` | `modeling_mom.py:430-453` | 训练时可融合 lm_head + CE | 只影响 loss/logits 物化，不影响 MoM attention |
| `aux_loss_scale` | `modeling_mom.py:455-464` | 负载均衡 loss 权重 | 全 padding mask 可能 NaN（`modeling_mom.py:101-115`） |
| `use_layer_wise_balance` | `configuration_mom.py:41,74` | 当前未见消费 | 字段存在但未在 forward/loss 中使用 |

## 8.3 边界条件

- `topk` 必须在 `[1, num_memories]`，但源码未在 config 阶段校验；
- `expand_v` 会用 `int(head_dim * expand_v)` 截断（`mom.py:326-328`），不像 `GatedDeltaNet` 有整数一致性校验（`gated_deltanet.py:126-140`）；
- `attention_mask` 只支持 2D padding mask（`mom.py:434-440`）；
- `output_attentions=True` 会被 `MomModel.forward` warning 并设为 False（`modeling_mom.py:284-286`）；
- `MomAttention` 裸层使用 cache 时必须有 `layer_idx`（`layers/utils.py:205-216`）；
- 保存/加载没有自定义迁移逻辑，不同 `num_memories/shared_mem/single_kv_proj` 的 checkpoint 互载会出现参数缺失或 shape mismatch。

## 8.4 本章小结

💡 小结

- MoM 配置里真正改变结构的是 `num_memories/topk/shared_mem/single_kv_proj/attn_mode`。
- `capacity` 和 `use_layer_wise_balance` 是当前最典型的“字段存在但主流程未消费”。
- 若要生产使用，最先应补 config 校验：`topk`、`expand_v`、`use_short_conv`、hybrid attention 返回值。

# 九、测试、示例与覆盖缺口

## 9.1 设计哲学与核心问题

源码走读不能把 README 里的“Add MoM implementation”当成可用性证明。测试告诉我们当前项目维护者认为哪些路径可靠，哪些路径仍 known buggy。

## 9.2 已覆盖路径

| 测试 / 示例 | 覆盖的行为 | 说明 |
|---|---|---|
| `README.md:41`, `README.md:98` | 文档索引 | 声明 MoM 已加入，并链接 layer |
| `tests/layers/test_layer_cache_layer_idx.py:120-122` | 把 `MomAttention` 纳入通用 cache layer_idx case | 只覆盖“无 layer_idx 但传 cache 应报错” |
| `tests/layers/test_layer_cache_layer_idx.py:176-179` | 缺 layer_idx 抛 ValueError | 证明 cache 保护生效 |
| `tests/models/test_modeling_mom.py:31-40` | 原计划模型 forward/backward | 但被 skip |
| `tests/models/test_modeling_mom.py:55-64` | 原计划 generation | 但内部 skip known bugs |

模型级测试的真实状态：

- forward/backward：`@pytest.mark.skip(reason="Bug not fixed yet")`（`test_modeling_mom.py:19`）；
- generation：`pytest.skip("Known bugs in mom")`（`test_modeling_mom.py:63`）。

因此当前测试只证明一个很窄的点：`MomAttention` 接入通用 cache helper 时，缺 `layer_idx` 会被拒绝。它没有证明 MoM forward/backward/generation 正确。

## 9.3 未覆盖风险

| 风险点 | 当前是否有测试 | 可能后果 |
|---|---:|---|
| router top-k 正确性 | 未见专测 | token 路由错误、权重混合错误 |
| `transform` / `reconstruct` round-trip | 未见专测 | shape 恢复错位、scatter 权重错误 |
| `capacity` 语义 | 未见专测 | 用户以为限流，实际热点 memory 膨胀 |
| `shared_mem=True` | 未见专测 | 默认模型路径额外分支不可靠 |
| `single_kv_proj=True` | 未见专测 | 参数共享分支 shape/性能风险 |
| `cu_seqlens` varlen | 模型测试 skip | varlen 与 padded 输出一致性无保护 |
| chunk vs fused recurrent 一致性 | 未见 MoM 专测 | eval 短序列与长序列行为不一致 |
| generation/cache | 明确 known bugs | cache state/offset/conv state 续接错误 |
| hybrid `attn` | 未见专测 | 初始化可过，forward 解包失败 |
| 全 padding mask aux loss | 未见专测 | loss NaN |
| 性能/显存收益 | 未见 benchmark | 无法证明 routing 均衡下收益或热点退化 |

## 9.4 本章小结

💡 小结

- README 是存在性信号，不是成熟度信号。
- MoM 模型级 forward/backward 和 generation 测试当前都被跳过。
- 最需要补的是 routing/transform/reconstruct、默认 `shared_mem=True`、cache generation、hybrid attention 与 config 边界测试。

# 十、局限性与已知优化点

## 10.1 设计哲学与核心问题

MoM 的实现方向清晰，但源码呈现的是一个实验性较强的版本。局限性不是泛泛的“还可优化”，而是具体落在配置校验、routing 负载、cache、测试关闭、Python 编排上。

## 10.2 硬约束

- 只支持 `mom_backend='gated_deltanet'`（`configuration_mom.py:85-86`）；
- 训练只支持 chunk（`mom.py:447-449`）；
- fused recurrent backward 未实现（`fused_recurrent.py:302-309`）；
- attention mask 只能是 2D padding mask（`mom.py:434-440`）；
- `use_short_conv=False` 无法正常构造（`mom.py:403-407`）；
- `topk`、`expand_v` 缺少 config 阶段校验；
- `MomAttention` 使用 cache 必须有 `layer_idx`（`layers/utils.py:205-216`）。

## 10.3 维护成本

- AutoClass 注册是全局 registry 修改，import 后影响 HF AutoModel 解析（`fla/models/mom/__init__.py:13-15`）；
- 参数结构依赖 `num_memories/shared_mem/single_kv_proj`，checkpoint 兼容性需要靠 config 保持一致；
- `MomAttention` 里混合了 routing、varlen、short conv、state remap、shared branch、输出门控，单函数维护负担较高；
- hybrid attention 初始化路径与 forward 返回值不一致，是典型“配置支持但测试未覆盖”的维护风险。

## 10.4 性能瓶颈

- `argsort` 对 `B*S*topk` 排序，长序列开销明显（`mom.py:172-173`）；
- `repeat_interleave(topk)` 直接复制 hidden（`mom.py:145-153`）；
- 每 memory projection 使用 Python list comprehension + `torch.stack`（`mom.py:483-488`）；
- `cu2pad`、`pad_for_conv`、`unpad_after_conv`、state remap 多处 Python for-loop（`mom.py:744-837`）；
- `shared_mem=True` 默认额外跑一条 GDN 分支（`mom.py:630-633`, `mom.py:685-736`）；
- `capacity` 未生效导致热点 memory 无法被限制。

## 10.5 已知优化点

源码中最直接的 TODO/未启用信号是 capacity 相关注释：`transform` 中 `capacity_len`、left pad、truncation 逻辑都被注释（`mom.py:193-198`）。围绕它可以形成几个优化方向：

1. **真正实现 capacity / token dropping / block capacity**：避免热点 memory 决定 `max_len`；
2. **替换全局 argsort**：用更专门的 grouped top-k / segmented gather 降低排序成本；
3. **融合 transform + projection**：减少 `[M,B,max_len,D]` 中间 buffer；
4. **修复 cache offset**：改为原始 `seq_len` / `q_len`，并补 generation 回归；
5. **给 `single_kv_proj` 和 `shared_mem` 补专测**：确保不同结构 checkpoint 和 forward 都可用；
6. **接入底层 CP context（若目标是分布式长序列）**：当前 GDN op 有 `cp_context` 参数，但 MoM 没传。

## 10.6 本章小结

💡 小结

- MoM 的硬约束集中在 backend、训练模式、mask、cache、配置边界。
- 性能瓶颈主要是 routing 布局编排，而不是 GDN kernel 本身。
- 最值得优先修的是测试关闭、cache offset、capacity 未生效和 hybrid attention 返回值不匹配。

# 十一、小结与展望

FLA 的 MoM implementation 可以用几个关键词概括。

**关键词一：HF 模型壳。**  
`MomConfig`、AutoClass 注册、`MomModel`、`MomForCausalLM` 让 MoM 进入标准 Causal LM 链路。它适合通过 `fla.models` 或 HuggingFace config 创建，而不是只作为孤立 layer 存在。

**关键词二：dispatch/collect 对称设计。**  
`MomAttention.forward` 先用 router 把 token dispatch 到 top-k memory，再用 `transform` 整理成 GDN 可处理的 varlen 输入，最后用 `reconstruct` collect 回原序列。这是 MoM 源码的主心骨。

**关键词三：Gated Delta Rule 复用。**  
MoM 没有重写 recurrence kernel，而是复用 `chunk_gated_delta_rule` 与 `fused_recurrent_gated_delta_rule`。训练用 chunk，短序列 eval 用 fused recurrent；后者 backward 未实现。

**关键词四：线性记忆收益与路由开销交换。**  
MoM 用 routing、排序、gather/scatter、短卷积和可选 shared branch 的开销，换取每个 memory 更短的线性序列和更强的记忆容量。均衡 routing 下收益更明显；热点 memory 或大 top-k 下收益会被削弱。

**关键词五：实验性实现。**  
`capacity` 未生效、`use_layer_wise_balance` 未消费、generation 测试 known bugs、模型 forward/backward 测试 skip、cache offset 可疑、hybrid attention 路径会解包失败。这些都说明它更适合研究和二次开发，而不是直接视为生产稳定模块。

这个实现适合：

- 想研究 mixture-of-memory 如何接入线性 attention / GDN 的读者；
- 想基于 FLA 快速实验 MoM 结构的研究者；
- 能接受补测试、修 cache、做性能 profiling 的工程团队。

它不适合：

- 直接上生产 generation；
- 依赖 capacity 做严格负载控制；
- 期待自动 expert parallel / context parallel 通信；
- 期待所有 HuggingFace attention mask / output_attentions / hybrid attention 组合都可用。

与替代方案相比，MoM 的取舍很清楚：相比普通 Gated DeltaNet，它增加了 memory mixture 的表达力，但也增加了路由和布局开销；相比 MoE，它没有跨 rank expert dispatch，也没有真正 capacity 限流；相比 softmax attention，它避免二次 attention 矩阵，但要承担 recurrent state、routing 均衡和 cache 复杂度。

后续值得继续走读的方向有三个：第一，底层 `fla/ops/gated_delta_rule` 的 chunk backward 与 CP context；第二，`ShortConvolution` 在 varlen/cache 下的状态语义；第三，MoM 若要接入分布式长序列训练，需要怎样把 memory routing 与 CP / SP / expert parallel 的通信边界打通。
