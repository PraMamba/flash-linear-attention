# Flash Linear Attention 源码走读：NSA implementations 实现解析

在长序列训练里，全量 softmax attention 的问题并不只是公式里的 $O(T^2)$。真正落到工程上，它会变成几类更具体的压力：score matrix 不能显式落地、KV 访问模式不够友好、稀疏选择如果做得太动态又很难被 GPU kernel 高效执行。`flash-linear-attention`（下文简称 FLA）在 2025-02 的 README changelog 中加入了 NSA，并把它指向 `fla/ops/nsa` kernel 目录（`README.md:54`）；模型表中对应论文是 *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*（`README.md:91`）。

本文不展开 NSA 论文推导，也不讲 Triton 编程细节，而是沿着 FLA 源码看一个更工程化的问题：**这个项目如何把 NSA 从一个稀疏注意力思想，接成 HuggingFace 风格的 CausalLM、可训练的 PyTorch autograd 算子，以及可测试的 Triton kernel？**

# 前言

## 业务 / 工程背景

NSA 在 FLA 中出现的位置很明确：它既是一个底层算子（`parallel_nsa`），也是一个可复用 attention layer（`NativeSparseAttention`），最后还被包装成 HuggingFace 风格的 `NSAForCausalLM`。也就是说，用户既可以直接测 kernel，也可以通过 `AutoModelForCausalLM.from_config(NSAConfig(...))` 走模型主路径。

它要解决的是长序列注意力的训练 / 推理效率问题，但这个实现不是简单把 dense attention 换成一个 sparse mask。源码里真正落地的是三路注意力：

1. compressed attention：对 K/V 做 block mean pooling 后在压缩序列上注意力；
2. selected block attention：根据压缩注意力的重要性选 block，再对原始 K/V 的选中块做 causal attention；
3. sliding-window attention：用 FlashAttention 做局部窗口注意力。

三路输出再由模型层学习出的 gate 融合。

## 核心矛盾

NSA 的核心工程矛盾可以概括成三句话：

1. 只做局部窗口会丢掉远距离信息；只做全局注意力又回到 $T^2$ 显存 / 访存压力。
2. 先压缩再选块能降低候选空间，但 selection 本身也要可训练、可反传，并且不能退化成 PyTorch 级别的低效 gather 循环。
3. 模型层要像普通 CausalLM 一样接入 Transformers，但底层 kernel 对 shape、head 分组、FlashAttention 依赖、varlen 输入都有硬约束。

## 本文主线

本文按机制展开，而不是按文件列表展开：

- 第一章看配置与入口：用户如何真正开启 NSA。
- 第二章看 attention layer：三路 gate 为什么在 QKV 投影旁边生成。
- 第三章看 `parallel_nsa`：压缩、选块、滑窗如何合成一次注意力。
- 第四章看 Triton 与 autograd：训练为什么需要自定义 forward/backward。
- 第五章把完整主路径串起来。
- 第六章梳理 shape、状态、cache、varlen 与 rank 语义。
- 第七章分析显存、性能和通信。
- 第八至十章讨论配置坑点、测试覆盖、局限与优化点。

## 不展开的内容

本文不讲 NSA 论文数学推导，不讲 FlashAttention 原理，不讲 Triton 语法，也不讲 DDP/FSDP 原理。我们只看 FLA 这份源码如何实现 NSA，以及哪些路径看似支持但实际有坑。

## 核心文件表

| 文件 | 职责 |
|---|---|
| `fla/models/nsa/configuration_nsa.py` | 定义 `NSAConfig`，包括 block、window、head、loss fuse 等配置 |
| `fla/models/nsa/__init__.py` | 注册 `AutoConfig` / `AutoModel` / `AutoModelForCausalLM` |
| `fla/models/nsa/modeling_nsa.py` | 定义 `NSABlock`、`NSAModel`、`NSAForCausalLM` 与 loss/cache 外壳 |
| `fla/layers/nsa.py` | `NativeSparseAttention`：QKV/gate 投影、RoPE、cache、调用 `parallel_nsa` |
| `fla/ops/nsa/parallel.py` | NSA 主算子：top-k block selection、selected attention、三路融合与反向 |
| `fla/ops/nsa/compression.py` | compressed attention 的 Triton forward/backward 与 autograd wrapper |
| `fla/ops/nsa/naive.py` | selected block attention 的 naive 参考实现，用于测试对齐 |
| `tests/ops/test_nsa.py` / `tests/models/test_modeling_nsa.py` | 算子与模型路径测试 |

# 一、配置与入口：让 NSA 成为一个可加载的 CausalLM

## 1.1 设计哲学与核心问题

NSA 首先要解决的不是 kernel，而是入口问题：用户如何不用手写一堆 layer 拼装，就能让 Transformers 的 Auto 类识别它？如果没有这一层，`parallel_nsa` 只能作为孤立算子存在；有了配置和注册，它才进入 `from_config`、`save_pretrained`、LM loss、generation mixin 这些通用模型生态。

FLA 的做法是典型的 HuggingFace 模型接入：`NSAConfig` 描述模型形态，`fla.models.nsa.__init__` 做 Auto 注册，`NSAForCausalLM` 包装 base model 与 LM head。

## 1.2 源码入口与关键对象

```text
fla/models/nsa/configuration_nsa.py
  - NSAConfig：定义 model_type='nsa' 和 NSA 相关配置

fla/models/nsa/__init__.py
  - AutoConfig.register / AutoModel.register / AutoModelForCausalLM.register：HF Auto 入口

fla/models/nsa/modeling_nsa.py
  - NSABlock：把 norm、NativeSparseAttention、MLP 串成 Transformer block
  - NSAModel：embedding + block stack + final norm
  - NSAForCausalLM：lm_head、loss、logits_to_keep、generation guard
```

配置类在 `fla/models/nsa/configuration_nsa.py:13-16` 定义 `model_type = 'nsa'`，默认参数包括 `hidden_size=2048`、`num_heads=64`、`num_kv_heads=4`、`head_dim=32`、`block_size=64`、`block_counts=16`、`window_size=512` 等（`configuration_nsa.py:18-48`），并写入实例属性（`configuration_nsa.py:50-76`）。HF Auto 注册在 `fla/models/nsa/__init__.py:8-15` 完成。

## 1.3 主流程拆解

用户最自然的模型入口是：

```python
from transformers import AutoModelForCausalLM
from fla.models import NSAConfig

config = NSAConfig(...)
model = AutoModelForCausalLM.from_config(config)
```

真实调用链可以简化成：

```text
User
  -> AutoModelForCausalLM.from_config(NSAConfig)
    -> NSAForCausalLM.__init__
      -> NSAModel.__init__
        -> NSABlock(config, layer_idx) * num_hidden_layers
          -> NativeSparseAttention(... block_size, block_counts, window_size ...)
```

`NSAModel.__init__` 创建 embedding、`NSABlock` 列表和 final norm，然后调用 `post_init()`（`fla/models/nsa/modeling_nsa.py:161-172`）。`NSABlock.__init__` 则把 config 中的 NSA 参数传给 `NativeSparseAttention`：`block_size`、`block_counts`、`window_size`、`rope_theta`、`max_position_embeddings` 都在这里传递（`modeling_nsa.py:48-61`）。这就是第一个真正改变模型行为的位置：从普通 Transformer block 切换到 NSA attention layer。

`NSAForCausalLM.forward` 的外层逻辑仍然是标准 CausalLM：先调用 `self.model(...)`（`modeling_nsa.py:315-325`），再根据 `labels` 与 fuse 配置决定是否计算 logits 和 loss（`modeling_nsa.py:330-348`）。如果开启 `fuse_linear_cross_entropy` 且传入 `labels`，它可以直接用 hidden states 和 `lm_head.weight` 计算 fused linear CE，避免显式保留完整 logits；这个开关在配置中还带有精度风险 warning（`configuration_nsa.py:82-87`）。

## 1.4 关键细节与误区澄清

这里有一个容易误解的点：README 只在 changelog 和模型表里提到 NSA（`README.md:54`, `README.md:91`），并没有提供独立 CLI、Trainer 配置样例或 NSA 专用示例。源码里也没有发现 NSA 专属的命令行入口。**真正入口是 Python API / HF Auto 注册，而不是某个训练脚本开关。**

第二个误区是：`NSAConfig` 看起来像完整 schema validation，但它主要是保存配置字段，只显式校验了 `fuse_cross_entropy` 与 `fuse_linear_cross_entropy` 不能同时为真（`configuration_nsa.py:78-87`）。更多 shape 约束是在底层 `parallel_nsa` 中用 assert 或运行时逻辑检查，例如 GQA group 约束和 key dimension 约束。

💡 小结

- NSA 的用户级入口是 `NSAConfig` + HF Auto 注册，而不是 CLI 开关。
- 配置层只做少量互斥校验，真正的形状约束延后到算子层。
- `NSABlock -> NativeSparseAttention` 是模型行为切换到 NSA 的第一处主路径。

# 二、NativeSparseAttention：三路 gate 为什么从 QKV 投影旁边长出来

## 2.1 设计哲学与核心问题

如果只看 `parallel_nsa`，很容易以为 NSA 是一个单独的 sparse attention kernel。但模型层告诉我们：FLA 的 NSA 是“三路注意力 + 学习门控”的组合。这个设计要解决的是表达能力和稀疏效率之间的冲突：压缩分支负责全局粗粒度信息，selected 分支负责少量远距离精细块，sliding-window 分支负责局部邻域；gate 则让模型按 token/head 自己决定三路权重。

因此 `NativeSparseAttention` 的职责不是写 kernel，而是把 hidden states 变成 `q/k/v` 和三路 gate，并处理 RoPE、cache 与输出投影。

## 2.2 源码入口与关键对象

```text
fla/layers/nsa.py
  - NativeSparseAttention.__init__：创建 q/k/v/g/o projection 和 RotaryEmbedding
  - NativeSparseAttention.forward：生成 q/k/v/g，处理 RoPE/cache，调用 parallel_nsa
```

初始化阶段，`NativeSparseAttention` 创建五个关键投影：`q_proj`、`k_proj`、`v_proj`、`g_proj`、`o_proj`（`fla/layers/nsa.py:63-67`）。其中 `g_proj` 的输出维度是 `num_heads * 3`，这正对应 compressed / selected / sliding 三路 gate。

## 2.3 主流程拆解

`NativeSparseAttention.forward` 的核心路径可以写成：

```text
hidden_states: [B, T, hidden]
  -> q_proj/k_proj/v_proj
       q: [B, T, num_heads, head_dim]
       k/v: [B, T, num_kv_heads, head_dim]
  -> g_proj + sigmoid + unbind
       g_cmp/g_slc/g_swa: [B, T, num_heads]
  -> RoPE(q, k)
  -> optional Cache.update(k, v)
  -> parallel_nsa(q, k, v, gates, block/window config)
  -> reshape [B, T, num_heads*head_dim]
  -> o_proj -> [B, T, hidden]
```

源码中 Q/K/V/G 的生成在 `fla/layers/nsa.py:89-93`：

```python
q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=3)
g_cmp, g_slc, g_swa = g.sigmoid().unbind(-1)
```

这段代码说明 gate 是逐 token、逐 query head 的，而不是整个 layer 共用一个标量。RoPE 在 `fla/layers/nsa.py:107-109` 应用到 q/k；随后如果传入 `past_key_values`，会更新缓存并可能把 k/v 替换为缓存后的历史 K/V（`fla/layers/nsa.py:111-123`）。最终调用 `parallel_nsa` 的位置在 `fla/layers/nsa.py:124-135`。

## 2.4 关键细节与误区澄清

第一个误区：`attention_mask` 被函数签名接收，就代表 padding mask 被 NSA 主流程应用了。源码并不是这样。`NativeSparseAttention.forward` 只检查 `attention_mask` 是二维 padding mask（`fla/layers/nsa.py:80-85`），但调用 `parallel_nsa` 时没有把 `attention_mask` 传下去（`fla/layers/nsa.py:124-135`）。除 cache 场景下用于调整 RoPE offset 的一小段逻辑外（`fla/layers/nsa.py:102-105`），标准 prefill 路径并没有把 padding mask 转成 unpad / `cu_seqlens`。因此，**有 padding 的 batch 会把 padding token 当作真实 token 参与 NSA 计算**，除非调用方自己提供已经 flatten 的 varlen 输入和 `cu_seqlens`。

第二个误区：`output_attentions=True` 可以拿到注意力权重。`NSAModel.forward` 会 warning 并强制关闭 attentions（`fla/models/nsa/modeling_nsa.py:192-195`）。更细一点看，直接调用 `NativeSparseAttention.forward(output_attentions=True)` 还可能遇到 `attentions` 未定义的问题，因为 `attentions = None` 只在 `not output_attentions` 分支赋值（`fla/layers/nsa.py:139-142`）。

💡 小结

- `NativeSparseAttention` 的关键不是 sparse kernel，而是生成三路 gate 并把模型层接到 kernel 层。
- Q 使用 `num_heads`，K/V 使用 `num_kv_heads`，这让 NSA 默认走 GQA 风格。
- `attention_mask` 和 `output_attentions` 是最容易误读的接口：前者没有真正参与 prefill attention，后者当前不支持。

# 三、parallel_nsa：压缩、选块、滑窗如何合成一次注意力

## 3.1 设计哲学与核心问题

到了 `parallel_nsa`，工程问题变成：如何在不构造完整 $T \times T$ attention matrix 的情况下，仍然让每个 query 有全局粗粒度视野、局部精细视野和少量远距离精细块？

FLA 的实现顺序很直接：先对 K/V 做 mean pooling 形成压缩序列；如果有 compressed gate，就用压缩注意力产生 compressed 输出与 LSE，再基于压缩注意力分数做 top-k block selection；随后在原始 K/V 上只对选中的 block 做 selected attention；最后可选地叠加 sliding-window FlashAttention。

## 3.2 源码入口与关键对象

```text
fla/ops/nsa/parallel.py
  - parallel_nsa：公开主入口，负责三路分支调度与 gate 融合
  - parallel_nsa_topk：根据 compressed K 与 LSE 生成 block_indices
  - ParallelNSAFunction：selected attention 的自定义 autograd wrapper

fla/ops/nsa/compression.py
  - parallel_nsa_compression：compressed attention 分支的自定义 autograd wrapper

fla/ops/utils/pooling.py
  - mean_pooling：把 K/V 按 block_size 压缩成 chunk 表示
```

`parallel_nsa` 的 docstring 明确了 shape 契约：q 是 `[B, T, HQ, K]`，k/v 是 `[B, T, H, K/V]`，输出是 `[B, T, HQ, V]`（`fla/ops/nsa/parallel.py:795-833`）。这里 `HQ` 是 query heads，`H` 是 KV heads，`G = HQ // H` 是每个 KV head 对应的 query head 组。

## 3.3 主流程拆解

核心代码路径在 `fla/ops/nsa/parallel.py:834-890`，可简化为：

```text
parallel_nsa(q, k, v, g_cmp, g_slc, g_swa, block_counts, block_size, window_size)
  -> validate block_counts / scale / varlen batch / GQA group
  -> k_cmp = mean_pooling(k, block_size)
     v_cmp = mean_pooling(v, block_size)
  -> if g_cmp is not None:
       o_cmp, lse_cmp = parallel_nsa_compression(q, k_cmp, v_cmp)
       block_indices = parallel_nsa_topk(q, k_cmp, lse_cmp)
  -> o_slc = ParallelNSAFunction.apply(q, k, v, block_indices, ...)
  -> o = o_slc * g_slc
  -> o += o_cmp * g_cmp
  -> if window_size > 0:
       o_swa = flash_attn_func / flash_attn_varlen_func(... window_size ...)
       o += o_swa * g_swa
  -> return o
```

几个 shape 变化值得单独画出来：

```text
原始 K/V:
  k, v:      [B, T, H, D]

Mean pooling 后:
  k_cmp:     [B, ceil(T / block_size), H, D]
  v_cmp:     [B, ceil(T / block_size), H, D]

Top-k selection 后:
  block_indices: [B, T, H, S]
  S = next_power_of_2(block_counts) 或 tensor block_counts 的最大值向上取 2 次幂

Selected attention 输出:
  o_slc: [B, T, HQ, D]

三路 gate:
  g_cmp/g_slc/g_swa: [B, T, HQ]
  unsqueeze 后与 [B, T, HQ, D] 相乘
```

压缩 K/V 的生成在 `parallel_nsa.py:844`；`mean_pooling` 的 varlen 限制在 `fla/ops/utils/pooling.py:198-215`，如果传入 `cu_seqlens`，batch 必须是 1。compressed 分支调用 `parallel_nsa_compression` 在 `parallel_nsa.py:846-854`；top-k selection 在 `parallel_nsa.py:857-865`；selected attention 在 `parallel_nsa.py:866`；gate 融合在 `parallel_nsa.py:867-870` 和 `parallel_nsa.py:889`；sliding-window FlashAttention 在 `parallel_nsa.py:871-889`。

## 3.4 关键细节与误区澄清

一个非常关键的误区是：传入 `block_indices` 就一定会使用这些 indices。源码明确相反：如果 `g_cmp is not None`，`parallel_nsa` 会先计算 compressed attention，再自己调用 `parallel_nsa_topk` 生成 `block_indices`；如果用户同时传了 `block_indices`，会 warning 说它将被忽略（`fla/ops/nsa/parallel.py:846-857`）。模型层总是传入 `g_cmp`，所以标准 `NativeSparseAttention` 主路径的 block selection 是内部生成的。

第二个误区是：`block_indices=None` 看起来是合法默认值。它只在 `g_cmp` 存在时才安全；如果直接调用 `parallel_nsa(q,k,v,g_cmp=None,block_indices=None)`，代码不会提前报错，而是把 `None` 传给 `ParallelNSAFunction.apply`（`parallel_nsa.py:788`, `parallel_nsa.py:866`），后续会在 kernel 或 wrapper 中失败。类似地，`window_size > 0` 时会使用 `g_swa.unsqueeze(-1)`（`parallel_nsa.py:889`），直接算子调用者也必须提供 `g_swa`。

第三个细节是：`parallel_nsa` 无论是否使用 compressed 分支，都会先执行 `mean_pooling(k/v)`（`parallel_nsa.py:844`）。标准模型路径下 `g_cmp` 总是存在，所以这是必要的；但直接调用算子且只想用外部 `block_indices` 时，这一步可能成为额外开销。

💡 小结

- `parallel_nsa` 不是一个单一 sparse kernel，而是 compressed / selected / sliding-window 三路调度器。
- 标准模型路径中，`block_indices` 由 compressed 分支自动生成；外部传入会被忽略。
- 直接调用底层算子时，签名里的可选参数并不都安全，需要自己保证 gates 和 block indices 的组合有效。

# 四、Triton selection 与自定义反向：训练为什么不靠 PyTorch 稀疏索引

## 4.1 设计哲学与核心问题

如果用 PyTorch 直接 gather 每个 query 的若干 block，再对每个 token 做 softmax，逻辑会很清楚，但性能很难接受：每个 query/head 的 block 集合不同，Python 循环和动态索引会把 GPU 吞吐打散。FLA 的选择是把 top-k selection、selected attention forward 和 backward 都写成 Triton kernel，并用自定义 autograd wrapper 把它们接回 PyTorch。

这一层解决的是训练性能和反向语义问题：forward 只看选中块，backward 也必须只对被选中且 causal 合法的位置累积梯度。

## 4.2 源码入口与关键对象

```text
fla/ops/nsa/parallel.py
  - parallel_nsa_kernel_topk：Triton top-k block selection
  - parallel_nsa_fwd_kernel：selected block attention forward
  - parallel_nsa_bwd_kernel_dq：计算 dQ
  - parallel_nsa_kernel_mask：为 dK/dV 构造 block mask
  - parallel_nsa_bwd_kernel_dkv：计算 dK/dV
  - ParallelNSAFunction：保存张量并连接 forward/backward

fla/ops/nsa/utils.py
  - _bitonic_merge：Triton bitonic merge，用于 selection 中的排序/合并

fla/ops/nsa/naive.py
  - naive_nsa：测试参考，不是主训练路径
```

## 4.3 主流程拆解

Top-k selection 的入口是 `parallel_nsa_topk`（`fla/ops/nsa/parallel.py:500-545`）。它先根据 `block_counts` 得到 `S`，并向上取 2 次幂（`parallel.py:512-514`），然后要求 `block_size >= 2 * S`（`parallel.py:518`），最后分配 `[B, T, H, S]` 的 `block_indices`（`parallel.py:520`）并启动 top-k kernel。

Triton top-k kernel 内部有两个值得注意的规则：

- 它先在 compressed K 上计算重要性分数；
- 注释和逻辑说明 “the 1st and the last 2 blocks are always selected”（`parallel.py:142-143`），最后把选中的 block id 写回 `block_indices`（`parallel.py:165-166`）。

Selected attention forward 则在 `parallel_nsa_fwd_kernel` 中完成（`parallel.py:182-265`）。每个 program 负责一个 token、一个 value tile、一个 KV head；它遍历 `NS` 个 selected block（`parallel.py:238-240`），对每个 block 内 token 做 causal mask（`parallel.py:249`），用在线 softmax 累积输出与 LSE（`parallel.py:251-265`）。

反向由 `ParallelNSAFunction` 接住：forward 保存 `q, k, v, o, lse`，以及 `block_indices`、`block_counts`、`cu_seqlens`、`block_size`、`scale` 等上下文（`parallel.py:739-756`）；backward 调用 `parallel_nsa_bwd`（`parallel.py:763-778`）。`parallel_nsa_bwd` 先用 `parallel_attn_bwd_preprocess` 生成 softmax 反向需要的 `delta`（`parallel.py:651`），再分别计算：

```text
dQ:
  parallel_nsa_bwd_kernel_dq
  对每个 query 遍历自己选中的 blocks

dK/dV:
  parallel_nsa_block_mask -> [B, T, H, NS]
  parallel_nsa_bwd_kernel_dkv
  对每个 K/V block 反查哪些 query 选中了它
```

这不是简单“forward 反过来跑一遍”。dQ 天然按 query 遍历；dK/dV 如果也按 query 写，会产生大量原子累加或冲突，所以源码先构造 block mask（`parallel.py:690`），再按 K/V block 维度累积（`parallel.py:695-721`）。

## 4.4 关键细节与误区澄清

第一个误区：`naive_nsa` 是主流程的一部分。它不是。`naive_nsa` 在 `fla/ops/nsa/naive.py:12-93` 实现了一个易读的参考版本，会把 K/V 和 block_indices 按 GQA group repeat 到 query heads（`naive.py:55-58`），逐 token gather block 并 softmax（`naive.py:76-91`）。但主模型路径调用的是 `parallel_nsa`，测试才用 `naive_nsa` 对齐 forward/backward（`tests/ops/test_nsa.py:61-76`）。

第二个误区：NSA 的“通信”来自分布式 all-gather / all-to-all。至少在 NSA 相关源码中，没有发现 `torch.distributed` collectives、process group、device mesh 或 rank mapping。这里的复杂性主要是单 GPU kernel 内部的访存、block selection 和自定义反向；分布式训练框架如果包住这个模型，看到的是普通 PyTorch module。

第三个细节：varlen 反向有一个值得警惕的实现风险。`parallel_nsa_kernel_mask` 比较 `b_i * BS <= i_t`（`parallel.py:286-293`），但它没有接收 `cu_seqlens` 或 token local index；在 packed varlen 输入里，`i_t` 可能是全局 flattened token 位置，而 selected block index 在其他 kernel 中按序列局部位置解释。这一点测试目前没有用“故意包含未来 block 的 varlen block_indices”去覆盖。

💡 小结

- NSA selected 分支用 Triton kernel 和自定义 autograd 保证训练路径可控。
- dQ 与 dK/dV 的反向遍历方向不同，`block_mask` 是避免写冲突的重要中间结构。
- `naive_nsa` 是参考测试路径，不是生产主路径；NSA 本身也没有实现分布式通信组逻辑。

# 五、完整主路径串联：一次真实 forward 到底经过哪些层

## 5.1 完整调用栈

以 `AutoModelForCausalLM.from_config(NSAConfig()).forward(input_ids, labels=...)` 为例，主路径如下：

```text
User: AutoModelForCausalLM.from_config(NSAConfig)
  │
  ├─ Step 1: HF 注册与模型构造
  │     └─ fla/models/nsa/__init__.py: Auto* register
  │     └─ NSAForCausalLM.__init__ -> NSAModel.__init__
  │
  ├─ Step 2: Block 初始化
  │     └─ NSAModel 创建 NSABlock 列表
  │     └─ NSABlock 创建 NativeSparseAttention + MLP
  │
  ├─ Step 3: 前向进入 CausalLM wrapper
  │     └─ NSAForCausalLM.forward 调用 self.model(...)
  │
  ├─ Step 4: Base model forward
  │     └─ embedding(input_ids)
  │     └─ for layer in self.layers: NSABlock.forward
  │
  ├─ Step 5: NSA attention layer
  │     └─ q/k/v/g projection + RoPE + optional cache
  │     └─ parallel_nsa(q, k, v, g_cmp, g_slc, g_swa)
  │
  ├─ Step 6: NSA ops
  │     └─ mean_pooling(k/v)
  │     └─ compressed attention + top-k block selection
  │     └─ selected block attention
  │     └─ optional sliding-window FlashAttention
  │     └─ gate 融合
  │
  └─ Step 7: LM head / loss
        └─ logits 或 fused linear cross entropy
```

## 5.2 每一层做了什么

模型构造层只执行一次：`NSAForCausalLM.__init__` 创建 `NSAModel` 和 `lm_head`（`fla/models/nsa/modeling_nsa.py:251-259`）；`NSAModel.__init__` 创建 embedding、layers、final norm（`modeling_nsa.py:161-172`）。这些步骤影响参数量和初始化显存，但不在每个 step 重复执行。

每个 forward step 中，`NSAModel.forward` 首先把 `input_ids` 查成 `inputs_embeds`（`modeling_nsa.py:200-208`），然后逐层调用 block（`modeling_nsa.py:215-226`）。`NSABlock.forward` 里 attention 前后都有 residual/norm/MLP：先 `attn_norm`，再 `self.attn(...)`（`modeling_nsa.py:80-89`），然后 MLP norm 和 MLP（`modeling_nsa.py:90-99`）。

`NativeSparseAttention.forward` 是每层每步真正进入 NSA 的地方：它产生 q/k/v/g、应用 RoPE、更新 cache，调用 `parallel_nsa`（`fla/layers/nsa.py:89-135`）。这一层不触发跨 rank 通信，但会触发多个 GPU kernel：pooling、compressed attention、top-k、selected attention、FlashAttention window。

`NSAForCausalLM.forward` 最后决定是否显式生成 logits。若 `labels is None` 或没有启用 `fuse_linear_cross_entropy`，会计算 `lm_head(hidden_states)`（`modeling_nsa.py:330-331`）；若 labels 存在且启用 fused linear CE，则直接用 `criterion(hidden_states, labels, lm_head.weight, lm_head.bias)`（`modeling_nsa.py:344-345`），这是输出 logits 显存的关键分叉。

## 5.3 哪些逻辑不在主路径

- `naive_nsa` 不在模型主路径，只在 `tests/ops/test_nsa.py` 中作为参考实现使用。
- `tests/models/test_modeling_nsa.py` 里存在 generation 测试（`test_modeling_nsa.py:45-62`），但共享测试工具把 `NSAConfig` 放在 `GENERATION_UNSUPPORTED`（`tests/models/test_modeling_utils.py:29-34`），因此实际会 skip（`tests/models/test_modeling_base.py:91-92`）。不要把它误解成 cache/generation 已有测试保护。
- NSA 相关文件中没有发现自定义 `save_pretrained`、`state_dict`、`load_state_dict` 或 checkpoint merge 逻辑；保存加载依赖 `PreTrainedModel` 的标准行为。
- NSA 相关文件中没有发现 monkey patch / replace / bind 之类的全局替换逻辑。Auto 注册是显式注册，不是 monkey patch。
- NSA 相关文件中没有发现 all-gather / all-to-all / reduce-scatter / broadcast / process group 逻辑；它不是一个序列并行通信特性。

💡 小结

- NSA 主路径从 HF Auto 入口一路串到 `parallel_nsa`，中间没有额外 Trainer/CLI 层。
- 每层 forward 会触发三路注意力相关 kernel；初始化和 loss fuse 是独立阶段。
- naive、generation test、自定义 save/load、分布式通信、monkey patch 都不是标准 NSA 主路径。

# 六、关键数据流、状态流与 shape 约束

## 6.1 Tensor shape 变化

标准非 varlen 输入下，shape 流程如下：

```text
输入:
  input_ids:      [B, T]
  hidden_states:  [B, T, hidden]

投影后:
  q:              [B, T, HQ, K]
  k:              [B, T, H, K]
  v:              [B, T, H, V]
  g_cmp/slc/swa:  [B, T, HQ]

压缩后:
  k_cmp/v_cmp:    [B, ceil(T / block_size), H, K/V]

选择后:
  block_indices:  [B, T, H, S]

三路输出:
  o_cmp:          [B, T, HQ, V]
  o_slc:          [B, T, HQ, V]
  o_swa:          [B, T, HQ, V]

融合与输出投影:
  o:              [B, T, HQ, V]
  reshape:        [B, T, HQ * V]
  o_proj:         [B, T, hidden]
```

`parallel_nsa` 对 GQA group 有硬断言：`q.shape[2] % (k.shape[2] * 16) == 0`（`fla/ops/nsa/parallel.py:842`）。docstring 说 query/KV head ratio 必须是 power of 2 且 >=16（`parallel.py:799-802`），但实现只检查“是否为 16 的倍数”，没有检查 power-of-two。这是文档和实现不完全一致的地方。

## 6.2 Varlen 输入：batch 维度为什么必须变成 1

varlen 模式使用 `cu_seqlens`，契约与 FlashAttention 类似：多个样本被 flatten 到一个 batch 里。`parallel_nsa` 明确要求如果 `cu_seqlens is not None`，`q.shape[0]` 必须等于 1，否则报错（`parallel.py:837-841`）。`mean_pooling` 也有同样的 batch=1 限制（`fla/ops/utils/pooling.py:206-211`）。

因此 varlen shape 应该是：

```text
原始多样本:
  sequence lengths: [T1, T2, ...]

flatten 后:
  q/k/v:       [1, sum(Ti), H, D]
  cu_seqlens:  [0, T1, T1+T2, ...]

kernel 内:
  token_indices 把 flattened token 映射回 (sequence_id, local_token_id)
```

`ParallelNSAFunction.forward` 的注释解释了 `token_indices` 如何从 `cu_seqlens=[0,2,6]` 变成 `[[0,0],[0,1],[1,0]...]`（`fla/ops/nsa/parallel.py:733-738`）；compression wrapper 也有同样注释（`fla/ops/nsa/compression.py:488-493`）。

## 6.3 状态切换：config、cache、autograd ctx

NSA 没有全局 context manager 或进程级状态切换，主要状态有三类：

```text
初始化状态:
  NSAConfig 字段 -> NSABlock -> NativeSparseAttention 成员变量

运行时 cache 状态:
  past_key_values.get_seq_length(layer_idx)
  past_key_values.update(attn_state=(k, v), cache_kwargs={'window_size': ...})

autograd 状态:
  ParallelNSAFunction.ctx 保存 q/k/v/o/lse/block_indices/scale/cu_seqlens
  ParallelNSACompressionFunction.ctx 保存 q/k/v/o/lse/scale/cu_seqlens
```

cache 的共享实现位于 `fla/models/utils.py`。`FLALayer.update` 会在 `attn_state` 存在时按 `window_size` 截断或 rolling 更新历史 K/V（`fla/models/utils.py:75-93`），并用 `_seen_tokens` 记录已见 token 数（`models/utils.py:121-132`）。NSA layer 在 `fla/layers/nsa.py:97-123` 使用这个状态。

这里有一个重要风险：decode/cache 路径看似暴露，但 selected kernel 假设 q/k 的时间维一致。`parallel_nsa_fwd` 从 `k.shape` 推导 `T` 并分配输出 `[B, T, HQ, V]`（`fla/ops/nsa/parallel.py:558-574`），而 decode 时 q 可能只有当前 token、k/v 是缓存后的长序列；`NativeSparseAttention` 又把输出 reshape 回当前 `seq_len`（`fla/layers/nsa.py:136`）。这就是为什么测试工具把 `NSAConfig` 列为 generation unsupported（`tests/models/test_modeling_utils.py:29-34`）。

## 6.4 Rank / Mesh / Process Group 变化

NSA 相关源码没有构造 device mesh，也没有 process group 切换。可以把它的 rank 语义写成：

```text
world_size = N
每个 rank 上：
  如果外部 DDP/FSDP 启动，每个 rank 持有自己的模型副本或参数分片
  NSA layer 内部只看本 rank 的 q/k/v tensor
  parallel_nsa 不知道 global rank / dp rank / tp rank / sp rank

NSA 相关通信:
  all_gather:      未发现
  all_to_all:      未发现
  reduce_scatter:  未发现
  broadcast:       未发现
  barrier:         未发现
```

唯一和切分工具相关的字段是 `NSAPreTrainedModel._no_split_modules = ['NSABlock']`（`fla/models/nsa/modeling_nsa.py:104-110`），它告诉上层工具不要把 `NSABlock` 拆开，但这不是 NSA 自己实现的通信。

💡 小结

- NSA 的核心 shape 是 `[B,T,HQ,D]` 对 `[B,T,H,D]` 的 GQA selected/compressed/window 组合。
- varlen 输入必须 flatten 到 batch=1，并通过 `cu_seqlens` 恢复局部 token 位置。
- cache 路径当前是高风险接口：状态更新存在，但 selected kernel 的 q/k 长度假设并不适合增量 decode。
- NSA 本身没有 rank/mesh/process group 状态。

# 七、显存、通信与性能：省在哪里，又把成本转移到了哪里

## 7.1 显存收益范围

| 内容 | 是否直接节省 | 原因 |
|---|---:|---|
| 参数 | ❌ | Q/K/V/G/O、MLP、embedding 都是普通参数；NSA 不做参数 sharding |
| optimizer state | ❌ | 没有 NSA 专属 optimizer state 压缩逻辑 |
| dense attention score | ✅ | selected/window/compressed 分支避免显式构造完整 `[B,HQ,T,T]` score |
| attention 输出激活 | 部分 | 输出仍是 `[B,T,HQ,D]`，但中间 score / value 聚合范围减少 |
| logits | 取决于配置 | `fuse_linear_cross_entropy=True` 且有 labels 时可不显式生成完整 logits（`modeling_nsa.py:330-345`） |
| input batch | ❌ | `input_ids` / hidden states 不因 NSA 自动压缩；varlen 需要调用方 flatten |
| backward buffer | ❌ / 额外增加 | 自定义反向保存 q/k/v/o/lse，并分配 `block_mask`、临时 dq/dk/dv |

真正的显存收益主要来自“不落地完整 attention matrix”和“每个 query 只访问 selected blocks / window / compressed chunks”。但它并不意味着所有激活都线性化：compressed attention 仍会随 `T * ceil(T/block_size)` 增长；selected 分支的成本约随 `T * S * block_size` 增长；sliding-window 分支随 `T * window_size` 增长。

反向还有额外显存峰值。`ParallelNSAFunction.forward` 保存 `q,k,v,o,lse`（`fla/ops/nsa/parallel.py:750-756`）；`parallel_nsa_bwd` 分配 `dq` 临时张量（`parallel.py:653`）、`block_mask`（`parallel.py:690`）、`dk/dv`（`parallel.py:691-692`）。其中 `block_mask` 是 `[B,T,H,ceil(T/block_size)]` 量级，长序列下不可忽略。

## 7.2 通信开销

从分布式通信角度，NSA 没有 all-gather、all-to-all、reduce-scatter、broadcast 或 barrier。每层新增的是 GPU kernel 与内存访问开销，而不是 rank 间通信开销。

每层 forward 的主要 kernel / 后端路径大致是：

```text
mean_pooling(k)
mean_pooling(v)
parallel_nsa_compression_fwd_kernel      # 如果 g_cmp 存在，模型路径总是存在
parallel_nsa_kernel_topk                 # 生成 block_indices
parallel_nsa_fwd_kernel                  # selected attention
flash_attn_func / flash_attn_varlen_func # window_size > 0 时
若训练反向:
  compression backward kernels
  selected backward dQ kernel
  block mask kernel
  selected backward dK/dV kernel
```

这意味着 NSA 是“用更多结构化 kernel 和额外访存，换掉 full attention score 的巨大矩阵”。它不需要跨 GPU 通信，但每层内部的 kernel launch、compressed selection、FlashAttention window 都会增加调度和访存复杂度。

## 7.3 性能取舍

NSA 的性能取舍可以拆成三点：

1. **用稀疏块访问换 full attention score。** selected 分支只看 `S * block_size` 个 token，但 selection 需要先在压缩空间上计算重要性。
2. **用 compressed branch 保留全局信息。** 这降低了全局注意力成本，但不是免费：`parallel_nsa_compression` 有自己的 forward/backward kernel（`fla/ops/nsa/compression.py:30-322`）。
3. **用 FlashAttention 处理局部窗口。** 默认 `window_size=512`（`configuration_nsa.py:28`），所以默认模型路径会调用 flash-attn（`parallel.py:871-889`）。如果没有安装 flash-attn，import 只发出 `ImportWarning` 并把 `flash_attn_func` 设为 `None`（`parallel.py:22-30`），运行到窗口分支时才会失败，这个默认依赖需要特别注意。

硬件上，selected/compressed forward 会根据 Hopper 与非 Hopper shared memory 限制选择 `BK/BV` 上限，并断言 key dimension 不能超过 256（`parallel.py:562-570`, `fla/ops/nsa/compression.py:338-346`）。这说明实现是强硬件约束的，不是任意 head_dim 都能跑。

💡 小结

- NSA 节省的是 attention score / 部分中间激活，不节省参数、optimizer state 或输入 batch。
- NSA 没有 rank 间通信；新增的是每层多个 GPU kernel、压缩选择和访存开销。
- 默认 `window_size=512` 让 FlashAttention 成为默认运行依赖，而不是可有可无的装饰。

# 八、配置项、边界条件与坑点：配置如何改变源码路径

## 8.1 关键配置如何改变行为

| 配置 / 参数 | 影响源码路径 | 行为变化 | 风险 / 坑点 |
|---|---|---|---|
| `block_size` | `configuration_nsa.py:26` -> `parallel_nsa.py:819-820` | 控制 mean pooling 粒度和 selected block 大小 | top-k 要求 `block_size >= 2*S`（`parallel.py:518`） |
| `block_counts` | `configuration_nsa.py:27` -> `parallel_nsa_topk` | 控制每个 query/head 选择多少 block | tensor 时取 max 后 next_power_of_2；过大直接触发 assert |
| `window_size` | `configuration_nsa.py:28` -> `parallel_nsa.py:871-889` | 大于 0 启用 sliding-window FlashAttention | 默认 512；没装 flash-attn 时运行期失败风险 |
| `num_heads / num_kv_heads` | `configuration_nsa.py:22-24` -> `layers/nsa.py:45-53` | 决定 HQ/H 的 GQA 组大小 | `HQ` 必须是 `H*16` 的倍数（`parallel.py:842`） |
| `head_dim` | `configuration_nsa.py:24` -> projection / Triton tile | 决定 Q/K/V 每 head 维度 | key dimension >256 会 assert（`parallel.py:570`） |
| `fuse_linear_cross_entropy` | `configuration_nsa.py:45` -> `modeling_nsa.py:330-345` | labels 存在时可跳过显式 logits | 配置 warning 提醒可能有精度风险（`configuration_nsa.py:82-87`） |
| `use_cache` | `configuration_nsa.py:37` -> `modeling_nsa.py:197-211` | eval 默认可能创建 Cache | NSA generation/cache 被测试工具标为 unsupported |
| `cu_seqlens` | `layers/nsa.py:95` -> `parallel_nsa.py:837-841` | 进入 varlen packed 模式 | batch 必须为 1，调用方要自己 flatten |

## 8.2 最小可用配置与默认行为

模型层最小入口可以是 `NSAConfig()`，默认 `num_heads=64,num_kv_heads=4`，GQA group 正好是 16，符合底层 assert。默认 `block_counts=16,block_size=64`，也满足 `block_size >= 2*S`。但默认 `window_size=512` 会走 FlashAttention window 分支，因此运行环境需要安装 flash-attn。

如果只想验证 selected branch 或没有 flash-attn，可以把 `window_size=0`。如果直接调用底层 `parallel_nsa` 且不提供 gates，就必须显式提供 `block_indices`；否则 `block_indices=None` 会晚些失败。

## 8.3 静默失效与不兼容组合

- `attention_mask`：只做二维 shape 校验，不参与 prefill NSA 计算；这是“看似支持，实际未消费”的典型接口。
- `output_attentions`：模型层强制关闭；直接 layer 调用还可能未定义变量。
- `block_indices + g_cmp`：如果 `g_cmp` 存在，外部 `block_indices` 被忽略。
- `head ratio`：docstring 说 power-of-two，但实现只检查 16 的倍数；这会造成配置理解不一致。
- `num_logits_to_keep`：`NSAForCausalLM.forward` 用 `@deprecate_kwarg` 把旧参数迁移到 `logits_to_keep`（`modeling_nsa.py:294-307`）。
- `head_first`：`naive_nsa` 中已移除，传入会触发 `DeprecationWarning`（`fla/ops/nsa/naive.py:47-50`），但这只影响测试参考函数，不影响主模型路径。

💡 小结

- 默认配置在 shape 上基本自洽，但默认依赖 FlashAttention。
- 很多限制不是 config 初始化时报错，而是在底层算子里 assert 或运行到某分支才暴露。
- `attention_mask`、`output_attentions`、`block_indices` 是最容易误判“已经支持”的三个接口。

# 九、测试、示例与覆盖缺口

## 9.1 已覆盖路径

NSA 的测试覆盖主要集中在两个层面。

算子层，`tests/ops/test_nsa.py::test_parallel` 构造 q/k/v/do 和随机 block_indices，用 `naive_nsa` 作为 reference，对齐 output 与 dq/dk/dv（`tests/ops/test_nsa.py:34-76`）。参数覆盖了非整齐序列长度 `T=63/111`、不同 D，包括 `D=100` 这种非 2 次幂维度（`test_nsa.py:24-31`）。

varlen 算子层，`test_parallel_varlen` 使用 `cu_seqlens`，把输入做成 `[1,total_tokens,H,D]`，同样对齐 naive 与 Triton 的 forward/backward（`tests/ops/test_nsa.py:79-154`）。

模型层，`tests/models/test_modeling_nsa.py::test_modeling` 调用共享的 `run_test_model_forward_backward`，覆盖 bf16、`use_l2warp`、不同 D（`tests/models/test_modeling_nsa.py:19-39`）。共享测试还会比较 fixed batch 与 flatten varlen 输出是否一致（`tests/models/test_modeling_base.py:53-68`）。

我在当前环境做了轻量验证：`python -m py_compile` 通过 NSA 源码与测试文件；`pytest --collect-only -q tests/ops/test_nsa.py tests/models/test_modeling_nsa.py` 收集到 12 个测试。没有运行完整 GPU/Triton 测试，因此不声称运行期全部通过。

## 9.2 未覆盖风险

| 风险点 | 当前是否有测试 | 可能后果 |
|---|---:|---|
| cache / generation 主路径 | ❌，NSA 被列为 generation unsupported | 用户使用 `use_cache=True` 时可能遇到 shape 错误 |
| attention_mask padding 正确性 | ❌ | padding token 参与注意力，batched padded 输出错误 |
| 默认 flash-attn 缺失 | ❌ | 默认 `window_size=512` 运行期失败且错误信息不清晰 |
| custom save/load | ❌ | 依赖 HF 标准路径，但无 NSA 专属 smoke test |
| 多机 / 多 rank | ❌ | NSA 本身无通信逻辑；外部并行兼容性未由 NSA 测试证明 |
| 性能 / 显存收益 | ❌ | 没有 NSA benchmark 或显存断言 |
| varlen future block edge case | 覆盖不足 | backward block mask 可能和 forward 语义不一致 |
| `output_attentions=True` layer 直调 | ❌ | 可能触发 `UnboundLocalError` |

示例与文档方面，`examples/`、`benchmarks/`、`evals/` 下未发现 NSA 专用示例；README 只有 changelog 和模型表链接（`README.md:54`, `README.md:91`）。因此，读者不应把当前仓库理解为已经提供完整 NSA 教程或 benchmark 套件。

💡 小结

- 测试最强的是算子 forward/backward 与 varlen parity。
- 用户级路径（padding mask、cache/generation、save/load、flash-attn 缺失）覆盖明显不足。
- README 提供发现入口，但没有 NSA 专用教程、性能报告或训练配置样例。

# 十、局限性与已知优化点

## 10.1 硬约束

从源码可以确认的硬约束包括：

- `block_counts` 不能为空（`fla/ops/nsa/parallel.py:834`）。
- varlen 模式要求 batch size 为 1（`parallel.py:837-841`）。
- Query heads 必须满足 `HQ % (H * 16) == 0`（`parallel.py:842`）。
- selected/compressed key dimension 不能超过 256（`parallel.py:570`, `fla/ops/nsa/compression.py:346`）。
- top-k selection 要求 `block_size >= 2*S`（`parallel.py:518`）。
- `window_size > 0` 依赖 flash-attn 分支（`parallel.py:871-889`）。
- attention mask 只允许二维 padding mask，并且没有真正下传给 `parallel_nsa`（`fla/layers/nsa.py:80-85`, `layers/nsa.py:124-135`）。

## 10.2 维护成本

NSA 的维护成本主要来自三类边界：

1. **上游依赖边界。** sliding-window 分支依赖 flash-attn，但 import failure 只发 warning（`parallel.py:22-30`），默认配置却会走该分支。
2. **接口承诺边界。** `use_cache=True`、`attention_mask`、`output_attentions` 都在签名中出现，但实际支持程度不一致。
3. **文档与实现边界。** head ratio 的 docstring 和 assert 不完全一致；测试文件开头还有一个没有解释的 `# FIXME`（`tests/ops/test_nsa.py:20`）。

此外，`ParallelNSAFunction` 和 `ParallelNSACompressionFunction` 都保存了较多 forward 张量用于 backward；这对 correctness 友好，但未来如果要做 activation checkpointing 或更激进的重算，需要小心处理 LSE、block_indices、block_mask 的一致性。

## 10.3 性能瓶颈

明显的性能瓶颈有：

- compressed branch 本身仍然需要对压缩序列做随 `T * ceil(T/block_size)` 增长的计算；
- top-k selection 需要遍历历史 compressed blocks，并做 bitonic merge；
- selected backward 的 dK/dV 需要构造 `[B,T,H,NS]` block mask；
- varlen sliding-window 分支把 `max_seqlen` 设成 `q.shape[1]`（`parallel.py:872-879`），对 packed batch 来说这是 total tokens，而不是每条序列的最大长度，可能带来 FlashAttention workspace 浪费；
- 直接算子路径即使不用 compressed branch，也会先 mean-pool k/v（`parallel.py:844`）。

## 10.4 已知优化点

可以从源码行为推导出几类优化方向：

- 对 `window_size > 0` 增加显式 flash-attn 可用性检查，并在错误信息中说明如何关闭窗口分支。
- 在 `parallel_nsa` 入口提前校验 gate / block_indices 组合，避免晚到 kernel 才失败。
- 修正 varlen FlashAttention 的 `max_seqlen`，用 `prepare_lens(cu_seqlens).max()` 而不是 total packed length。
- 为 varlen backward block mask 引入 local token index，避免全局 flattened index 与局部 block id 混用。
- 对 cache/generation 要么显式 `NotImplementedError`，要么实现 q_len != kv_len 的 decode kernel 路径。
- 增加 `save_pretrained/from_pretrained`、padding mask、flash-attn 缺失、性能/显存 benchmark 的测试或示例。

💡 小结

- NSA 的硬约束集中在 head 分组、block selection、head_dim、varlen batch 和 flash-attn 依赖。
- 维护风险不是 monkey patch，而是“签名看似支持，但主路径没有完整实现”的接口边界。
- 后续优化最值得优先处理的是默认依赖检查、cache 明确语义、varlen 性能与测试覆盖。

# 小结与展望

FLA 的 NSA implementations 可以用几个关键词概括。

**关键词一：HF 兼容外壳。**  
`NSAConfig`、Auto 注册、`NSAForCausalLM` 让 NSA 不是孤立 kernel，而是能进入 Transformers 模型生态的 CausalLM。保存加载目前也依赖这层标准 `PreTrainedModel` 机制，而不是 NSA 自定义 checkpoint 逻辑。

**关键词二：三分支门控。**  
源码中的 NSA 不是单一路径 sparse attention，而是 compressed / selected / sliding-window 三路输出加 gate 融合。`g_proj -> sigmoid -> g_cmp/g_slc/g_swa` 是理解模型层的关键。

**关键词三：压缩驱动选块。**  
标准模型路径下，外部不提供 block selection；`parallel_nsa` 会先在 mean-pooled K/V 上做 compressed attention，再用 LSE/importance 生成 selected block indices。这让 selection 和模型输入相关，但也带来了额外 kernel 与访存成本。

**关键词四：自定义反向。**  
selected attention 的训练不是靠 PyTorch 稀疏 gather 拼出来，而是由 Triton forward/backward 和 autograd wrapper 负责。dQ 与 dK/dV 的遍历方向不同，`block_mask` 是反向实现的重要中间状态。

**关键词五：算子复杂度换显存，而不是分布式通信换显存。**  
NSA 避免 full attention score，适合长序列中希望保留局部与少量全局信息的场景；但它不节省参数和 optimizer state，也没有内置 rank 间通信。对于需要序列并行、checkpoint merge、rank0-only loading 的场景，当前 NSA 源码没有提供专门支持。

这个实现适合：研究和训练 NSA 风格长序列模型、验证 Triton sparse attention kernel、在有 flash-attn/Triton GPU 环境下探索 block sparse + compressed + local window 的组合。它不适合：直接拿来做完全可靠的 padding-mask batch 推理、cache-based generation、多机序列并行，或在没有 flash-attn 的环境中使用默认配置。

与 dense FlashAttention 相比，NSA 用 selection 和多分支 kernel 换取更低的长序列 attention 中间量；与纯局部窗口相比，它用 compressed branch 和 selected remote blocks 补充全局信息；与真正的分布式 sequence parallel 相比，它更像单 rank 内的稀疏算子优化。后续值得继续走读的方向，是把 NSA 与 FLA 其他模型的 varlen/cache 处理做横向对比，并进一步验证它在真实训练配置中的显存曲线和 kernel 时间占比。
