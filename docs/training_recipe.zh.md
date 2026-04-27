# T6 / T7 Training Recipe（v1.6.2）

> 语言：中文 | English: [training_recipe.md](training_recipe.md) (TODO)

**目的**：把 T6（teacher-trace SFT）和 T7（self-distill）的全部训练细节归档，方便复现。包括 loss 数学定义、超参、parallelism 配置、踩过的坑。

---

## 0. Base 模型

```
GSAI-ML/LLaDA-8B-Instruct        (8B params, bf16)
本地路径：checkpoints/llada-instruct/
mask_token_id = 126336 (<|mdm_mask|>)
vocab_size ≈ 126,464
hidden_size = 4096
num_layers = 32
```

LLaDA 是 absorbing-state masked diffusion LM（"MDLM" 类）。前向是任意位置的 mask → 模型预测全部位置的 logits → 用置信度选择 commit 哪些位置。

---

## 1. Loss 函数（MDLM masked-diffusion loss + answer-only mask）

### 1.1 Forward noise

对每个 batch sample，独立采时间 t ∈ Uniform(ε, 1−ε)（ε = 1e-5），按 t 的概率 mask 每个位置：

```
x_t[i, j] = mask_token_id   with prob t[i]
x_t[i, j] = x_0[i, j]       with prob 1 − t[i]
```

代码：`LLaDAWrapper.noise_input` (`src/dllm_reason/models/llada.py:173`)

```python
def noise_input(self, x_0, t):
    sigma = t[:, None].expand_as(x_0)              # (B, L)
    mask = torch.rand_like(sigma.float()) < sigma  # Bernoulli per position
    return torch.where(mask, self.mask_token_id, x_0)
```

### 1.2 Per-token NLL

模型输入 x_t 输出 logits ∈ ℝ^(B, L, V)：

```
log_probs[i, j, v] = log_softmax(logits[i, j, :])[v]
nll[i, j]          = −log_probs[i, j, x_0[i, j]]
```

### 1.3 Loss masking（关键）

只在 **被 noise mask 掉 AND 不是 prompt 区 AND attention_mask=1** 的位置上算 loss：

```
is_masked[i, j]   = (x_t[i, j] == mask_token_id)              # noise 操作
                  AND ~prompt_mask[i, j]                       # 答案区
                  AND attention_mask[i, j]                     # 非 padding
```

每条 sample 内 normalize（除以本条 sample 的 masked 位置数），再 batch 平均：

```
masked_nll[i] = sum_j (nll[i, j] × is_masked[i, j])
num_masked[i] = max(sum_j is_masked[i, j], 1)                  # 防 div by 0
loss          = mean_i (masked_nll[i] / num_masked[i])
```

代码：`Finetuner._compute_finetune_loss` (`src/dllm_reason/training/finetune.py:66`)

### 1.4 Loss 数学

每条 sample 实际优化的是：

```
L = 𝔼_{t∼U(ε, 1−ε)}  𝔼_{x_t∼q(·|x_0, t)}  [
        −1 / |M(t)|   ·   Σ_{j∈M(t)}  log p_θ(x_0[j] | x_t)
    ]
```

其中 M(t) = (位置在答案区) ∩ (此 sample 此时刻被 mask 掉的位置)。**等价于 absorbing-state diffusion 的 Bayes 重构 loss 的简化版**（不再 weight 1/t，因为 per-sample average 已经做掉时间归一）。

### 1.5 prompt_mask 来源

数据预处理时 `build_jsonl_dataset` 把每条 (question, answer) 编码成单个 token 序列：

```
[<|user|>question<|assistant|>answer<eos>]
 └──── prompt_mask=True ─────┘└── False ──┘
```

prompt_mask 的 True 区不参与 loss → **模型只学"在 prompt 后预测 answer"**。

---

## 2. 数据 pipeline

### 2.1 T6（teacher-trace SFT）

**源**：gsm8k train（2000 prompts）→ Qwen3-8B 推理 → 过滤 `is_correct(extracted_answer, gt)` 通过 → 清洗后 ~1500 条。

**生成命令**（`scripts/validate/t6_teacher_trace.py`）：

```bash
torchrun --nproc_per_node=8 scripts/validate/t6_teacher_trace.py \
    --teacher_ckpt checkpoints/Qwen__Qwen3-8B \
    --gsm8k_split train \
    --max_prompts 2000 \
    --temperature 0.7 \
    --retries 5 \
    --collect_all_valid True
```

**输出**：`runs/validation/t6_teacher_trace_<ts>/t6_sft.jsonl`

格式：
```json
{"question": "...", "answer": "Step 1: ... \\boxed{72}", "gt": "72"}
```

train/val split：`val_frac=0.1` → 1350 train / 149 val（数字以日志为准）。

### 2.2 T7（self-distill）

**源**：gsm8k train 全集 → T6 best ckpt（默认 step_336）在 T>0 下采样 → `is_correct` 过滤正解 → pick 策略选 1 条作为 SFT target。

**生成命令**（`scripts/validate/t7_gen_correct_samples.py`）：

```bash
torchrun --nproc_per_node=8 t7_gen_correct_samples.py \
    --model runs/training/v161_t6_ablate/hf_step_336 \
    --scope_path runs/validation/gsm8k_train_prompts.json \
    --temperatures 0.7,1.0 \
    --n_samples 8 \
    --gen_length 192 --block_length 32 \
    --pick first \
    --prompt_batch 4 \
    --prompt_shard <s>/8
```

**输出**：`runs/validation/t7_gen_<ts>/t7_sft.jsonl`，cover_rate 通常 ~95%（1918/2000）。

**pick 策略**：
- `first`（默认）：按迭代顺序 (T 从小→大, sample 0..7) 第一个正解 → 偏 greedy 风格
- `shortest` ❌：易选 truncation/lucky 样本（T7 v1 失败）
- `longest`：偏长推理
- `random`：固定 seed=42

**数据示例**（T7 v1，`pick=shortest` 故意展示问题）：
```json
{"question": "Natalia sold clips...", "answer": " 24+48 = 72. \\boxed{72}",
 "selection": "shortest", "temperature": 0.7, "n_candidates": 16}
```

---

## 3. SFT 超参（T6 default）

| 项 | 值 | 说明 |
|---|---|---|
| init_ckpt | `GSAI-ML/LLaDA-8B-Instruct` | T6；T7 改为 T6 best ckpt |
| **max_steps** | **336**（≈ 2 epoch on 1350 data） | T6 实测 sweet spot；超过 4 epoch 灾难性遗忘 |
| batch_size | 1 | per-rank micro-batch |
| grad_accum_steps | 16 | gradient accumulation |
| world_size | 8 | torchrun --nproc_per_node=8 |
| **effective batch** | **1 × 16 × 8 = 128 samples / opt-step** | |
| lr | 2e-5 | 经典 SFT lr |
| warmup_steps | 100 | linear warmup |
| max_seq_len | 768 | prompt(~150) + trace(~500) + headroom |
| max_grad_norm | 1.0 | 梯度裁剪 |
| seed | 42 | torch.manual_seed |
| dtype | bf16 | weights / activations / softmax 全 bf16 |

### 3.1 T7 差异

| 项 | T6 | T7 v2 |
|---|---|---|
| init_ckpt | base LLaDA | T6 step_336 |
| jsonl size | ~1350 | ~1918 |
| max_steps | 336 | **480**（保持 2 epoch on 1918）|
| 其余 | 同 | 同 |

### 3.2 LoRA Variant（可选）

```bash
--use_lora --lora_r 16 --lora_alpha 32 --lora_dropout 0.05
--lora_target q_proj,k_proj,v_proj,o_proj
--lora_merge_on_save True   # 保存时 merge 进 base，下游 v16_eval 不用 peft 依赖
--parallel ddp              # LoRA optim state 极小，FSDP overhead 不值
```

trainable params ≈ 10M / 8B（≈ 0.12%）。

---

## 4. Optimizer + Scheduler

### 4.1 Optimizer

```python
torch.optim.AdamW(
    params=model.parameters(),
    lr=2e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
)
```

**注意**：fp32 moments（FSDP MixedPrecision 自动 cast；optimizer state 在 fp32）。8B model AdamW moments ≈ 64 GB（如不 shard 则单卡爆 80GB；FSDP 切 8 路 → 8 GB/rank）。

LoRA 模式下：`Finetuner.__init__` 默认 `model.parameters()` 包含 frozen base，浪费 fp32 moments。`t6t7_train.py` 在 LoRA path 检测后**重建** optimizer 只 track `requires_grad=True` 参数（同时**重建** scheduler，否则 scheduler 挂在旧 optim 上 lr 永不更新）。

### 4.2 Scheduler

```python
torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=1000,        # 第一周期长度（steps）
    T_mult=2,        # 每个周期翻倍
)
```

每次 `optimizer.step()` 后调用 `scheduler.step()`（注意：是 optim step 不是 backward step；grad_accum 后才会触发）。

实际有效 lr 曲线：
```
step  0:    2e-5    (cosine 起点)
step  500:  ~1.5e-5
step  1000: ~0      (第一周期结束)
step  1001: 2e-5    (重启)
step  3000: ~0      (第二周期结束，T_mult=2)
...
```

T6 训 336 步只到周期 1/3，整体偏 cosine 下降的左半段，lr 平均约 1.6e-5。

---

## 5. Parallelism（FSDP 默认）

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision, ShardingStrategy,
    StateDictType, FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

mp = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

model._llada = FSDP(
    model._llada,
    sharding_strategy=ShardingStrategy.FULL_SHARD,    # weights/grads/optim 全切
    mixed_precision=mp,
    auto_wrap_policy=functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=int(1e7),                       # 子 module ≥ 10M 参数才单独 wrap
    ),
    device_id=torch.cuda.current_device(),
    use_orig_params=True,                              # LoRA + FSDP 必需
    sync_module_states=True,                           # rank 0 → broadcast init
    limit_all_gathers=True,                            # 限并发 all-gather
)

FSDP.set_state_dict_type(
    model._llada,
    StateDictType.FULL_STATE_DICT,
    FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
)
```

**FSDP 只包 `model._llada`**（HF 内层），**不包 `LLaDAWrapper` 外层**。这样 `noise_input / mask_token_id / device / tokenizer` 等自定义属性直接可用，不需要 `TransparentDDP` 那种属性兜底。

### 5.1 显存账（A100-80GB, 8 rank, bf16）

```
weights sharded:        16 GB / 8 = 2 GB / rank
gradients sharded:      16 GB / 8 = 2 GB / rank
AdamW fp32 moments:     64 GB / 8 = 8 GB / rank
activations + workspace: ~15-20 GB / rank
total per rank:         ~30 GB    (fits 80GB w/ headroom)
```

DDP 全参不可行（不 shard optim，单卡 ~96 GB > 80 GB）。

### 5.2 DDP（only LoRA）

LoRA 下 optim state 极小（~10M params × fp32 ≈ 40 MB），DDP 单卡显存够：

```
weights bf16:          16 GB
LoRA grads bf16:       40 MB
AdamW fp32:            ~80 MB
activations:           ~15-20 GB
total:                 ~32 GB    (fits)
```

---

## 6. Checkpointing

### 6.1 Per-step

```
--save_every 500          # 每 N steps 存 step_<N>.pt
--keep_last_n 2           # 滚动清理：只留最近 N 个 step ckpt
```

`step_<N>.pt` 格式（`Finetuner.save_checkpoint`）：
```python
torch.save({
    "model_state_dict":     ...  (FSDP-gathered FULL_STATE_DICT, rank 0 only)
    "optimizer_state_dict": ...  (rank 0 saves; resume 暂禁用 — see B4)
    "scheduler_state_dict": ...
    "global_step":          int
    "best_val_loss":        float
}, path)
```

约 96 GB / 个（model 32 GB fp32 + optim 64 GB fp32 + 元数据）。所以 `keep_last_n=2` 必开。

### 6.2 HF export

每次训练结束**自动**导出可服务的 HF 格式：

```
runs/training/v161_t6/hf/
├── config.json
├── model-00001-of-00006.safetensors
...
├── tokenizer.json
└── modeling_llada.py        (从 init_ckpt 拷贝；trust_remote_code 用)
```

**CRITICAL**：`FSDP.summon_full_params(rank0_only=True)` 是**集体通信**，所有 rank 必须进 context。**只 rank 0 写盘**，其他 rank 在 context 退出时 release shard。早期 v1.6.1 直接 `if not is_main: return` 会**死锁**（rank 0 等不到其他 rank）。

### 6.3 多 ckpt 一次训练（`--hf_export_at_steps`）

训练**确定性**：θ 在 step N 不依赖是不是 fresh start。所以扫 epoch 消融**只需要 1 次训练**到 max(epochs)，途中在指定 step 导出 HF ckpt：

```bash
--hf_export_at_steps "84,168,336,672"
# 训练到 672 步，途中在 84/168/336 也导出 hf_step_<N>/
```

实现：`Finetuner.step_hook` 钩子在每 step 后调用。`t6t7_train.py` 注册 hook 检测到 step 在 `export_steps` set 里就调 `export_hf(save_dir / f"hf_step_{step}")`。

---

## 7. Eval 配置（canonical）

T6/T7 训完后默认 eval 在 **canonical config**：

```bash
python scripts/validate/v16_eval.py \
    --ckpts baseline=GSAI-ML/LLaDA-8B-Instruct \
            t6=runs/training/v161_t6/hf \
    --out_dir runs/validation/v161_eval_<ts> \
    --gen_length 128 \
    --block_length 32 \
    --temperature 0
```

**所有 scope_fail / scope_ok 都是按这套配置定义的**（baseline 在这里答错 = fail）。改任一参数 = scope 重定义。

| 项 | 值 | 影响 |
|---|---|---|
| gen_length | 128 | answer 区长度（影响 ceiling 5%-15%）|
| block_length | 32 | block-wise 解码块大小 |
| temperature | 0 | greedy argmax |
| remasking | low_confidence | sampler 内硬编码 |
| steps | gen_length（128）| diffusion forward 总数 |

---

## 8. 已知坑（v1.6.1 audit + 后续）

### 8.1 已修

| ID | 症状 | 修法 |
|---|---|---|
| B1 | FSDP HF export 非 rank0 提前 return → 死锁 | 全 rank 进 collective，rank 0 写盘 + barrier |
| B2 | LoRA optim 重建后 scheduler 挂旧 optim | 同时重建 scheduler |
| B3 | Finetuner save_checkpoint 缺 rank-0 guard | save_checkpoint 集体安全（all-gather + rank 0 写）|
| B4 | FSDP resume 静默 OOM | refuse FSDP resume，报清楚错（fresh 重训）|
| B5 | LoRA target 默认 Llama 名，可能不匹配 LLaDA | catch ValueError 列出 actual Linear names |
| B6 | val_loss 各 rank 分歧 | 加 all_reduce(AVG) |
| S1 | fp32 加载（32 GB peak） | `torch_dtype=bfloat16` 直接加载 |

详见 `docs/archive/issues/minors.zh.md`。

### 8.2 未修（已知限制）

| 项 | 影响 | 缓解 |
|---|---|---|
| FSDP resume | 不能从 step_N.pt 续训 | 每次 fresh，用 `--hf_export_at_steps` 多 ckpt 共享 1 次训练 |
| `attention_mask` 传给 LLaDA forward | 极少数 LLaDA 远程代码版本不接受；fallback 到无 mask | h3_passN / generate_batched_multi 启动时一次性 probe |
| Position-id shift in batched eval | padding 在 prompt 之间会改 RoPE 相对位置 | h3_passN 按 prompt 长度排序，每 chunk 内长度接近，padding 最小化 |

---

## 9. 完整命令

### T6（默认 epoch=2）

```bash
torchrun --standalone --nproc_per_node=8 \
    scripts/validate/t6t7_train.py \
        --jsonl_path runs/validation/t6_teacher_trace_<ts>/t6_sft.jsonl \
        --run_name v161_t6 \
        --init_ckpt GSAI-ML/LLaDA-8B-Instruct \
        --max_steps 336 \
        --batch_size 1 --grad_accum_steps 16 \
        --lr 2e-5 \
        --warmup_steps 100 \
        --max_seq_len 768 \
        --parallel fsdp \
        --hf_export_at_steps "84,168,336,672"   # 消融用
```

### T7 v2（一键流水线）

```bash
bash scripts/t7_pipeline.sh
# 默认参数：
#   gen_ckpt = base_ckpt = T6 step_336
#   pick = first
#   max_train = 480 (= 2 epoch on 1918)
#   prompt_batch = 4 (gen 阶段 OOM-safe at gen_length=192)
```

或显式：

```bash
bash scripts/t7_pipeline.sh \
    --base_ckpt runs/training/v161_t6_ablate/hf_step_336 \
    --gen_ckpt  runs/training/v161_t6_ablate/hf_step_336 \
    --temperatures 0.7,1.0 \
    --n_samples 8 \
    --pick first \
    --max_train 480
```

### LoRA Variant

```bash
torchrun --standalone --nproc_per_node=8 \
    scripts/validate/t6t7_train.py \
        --jsonl_path ... \
        --init_ckpt GSAI-ML/LLaDA-8B-Instruct \
        --max_steps 336 \
        --use_lora --lora_r 16 --lora_alpha 32 \
        --parallel ddp \
        --hf_export_at_steps "84,168,336,672"
```

---

## 10. 时间预算（8×A100）

| 任务 | 时长 |
|---|---|
| 单次 T6 SFT（336 steps）| ~10 min |
| 单次 T7 v2 SFT（480 steps）| ~15 min |
| 单次 LoRA SFT | ~8 min |
| HF export | ~2 min |
| Canonical eval（baseline + ckpt）| ~10 min |
| decode_ablate（3 cells × full scope, 8 GPU sharded + autotune）| ~5-10 min |
| T7 gen 阶段（gsm8k train 2000 prompts × 16 samples）| ~1.5h |
| T7 pipeline 一键（gen + SFT + eval）| ~2.5-3h |

---

## 11. 监控

```bash
# 训练 loss 曲线
grep -E 'loss=|val_loss' runs/ablate_logs/<train>.log | tail -50

# GPU 利用率（应 70-90%）
watch -n 5 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader'

# 当前正在跑的 ckpt
ls runs/training/v161_t6/step_*.pt | tail -3

# 看 best.pt 大小是否合理（约 32 GB model + 64 GB optim 的存档）
ls -lh runs/training/v161_t6/best.pt
```

---

## 12. Sanity checks

跑完一个 SFT 后，打开 `train_meta.json` 看是否符合预期：

```json
{
  "cli_args": {"max_steps": 336, "lr": 2e-5, "parallel": "fsdp", ...},
  "finetune_config": {"loss_on_answer_only": true, ...},
  "train_size": 1350, "val_size": 149, "world_size": 8,
  "started_at": "2026-04-..."
}
```

跑 canonical eval 看 T6 数字是否落在 ablation 已知范围（28% fail / 91% ok @ step_336）。**严重偏离要 audit log 找 bug**。

---

## 13. 文档导引

| 概念 | 主文档 |
|---|---|
| Loss / FSDP / pipeline | 本文 |
| 假设登记 + verdict | `docs/archive/hypotheses.zh.md` |
| 术语（Ceiling-5 / hardset / FAIL18 等）| `docs/archive/definitions_hard_sets.zh.md` |
| A 轴探索（v1.5 时期推理实验）| `docs/archive/finding_a_axis_exploration.zh.md` |
| T6 24-ckpt 消融 | `docs/archive/finding_t6_training_ceiling.zh.md` |
| P2 decode_ablate + SC | `docs/archive/finding_p2_decode_frontier.zh.md` |
| 整条逻辑链 | `docs/archive/logic_chain_a_axis_to_p2.zh.md` |
| 跨版本 bug 日志 | `docs/archive/issues/minors.zh.md` |
