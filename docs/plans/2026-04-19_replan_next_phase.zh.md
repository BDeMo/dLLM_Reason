# Plan: 2026-04-19 Replan —— SS 搁浅后的下阶段规划

**日期**：2026-04-19（首版，骨架待讨论填充）
**状态**：**active** ⭐
**前置上下文**：[2026-04-19_ss_run_postmortem.zh.md](2026-04-19_ss_run_postmortem.zh.md) —— SS 5D full space 不可行（real-prompt O(seq²) 让预算爆掉 30×）
**继任**：—

---

## 现状盘点

### 已有的**不需要重跑**

| Artifact | 状态 | 备注 |
|---|---|---|
| A1-A6 实验数据（n=60 fail）| ✓ | `runs/validation/a*_*`  |
| H3 n=60 pass@N 数据 | ✓ | `h3_passN_20260415_133254` |
| P6 crossref（权威）| ✓ | `runs/validation/p6_h3_crossref/` |
| 全方法 union 55/60 = 91.67% | ✓ | 定量 claim |
| Ceiling 5 = `{4, 5, 14, 41, 42}` | ✓ | 定量 claim |
| A6-only `{19, 51}` + idx=48 | ✓ | write-space > diversity 证据 |
| Paper framing 敲定 | ✓ | canvas-constrained reasoning |
| Ablation index + Setting | ✓ | 索引 |
| SS 工具链代码（strategy_search / run_ss_shards / ss_analyze / ss_to_sft）| ✓ | 设计 OK，搜索空间要缩 |

### 已有的**部分可用的**

| Artifact | 状态 | 备注 |
|---|---|---|
| SS run `strategy_search_20260417_042620` | 15/109 完成 | 可做初步 template_position 倾向分析，但不足以 paper claim |
| SS oracle prior 归档 | 待实测验证 | 中位 55/60 仍期望成立 |

### **缺口**（paper claim 需要补的）

| 需要 | 当前状态 | 优先级 |
|---|---|---|
| **template_position novelty**（inpaint_novel_set）| 未验证 | **HIGH**（paper 核心 claim） |
| Per-prompt 最优 strategy 分布 | 未验证 | HIGH |
| Oracle rate > 91.67% or = 91.67% | 未验证 | MEDIUM |
| SFT distill baseline（policy head）| 未跑 | MEDIUM（paper 可做 future work） |
| 每 shard GPU 速度差异根因 | 未调查 | LOW |

---

## 核心权衡

**Paper 想 claim 什么**决定搜索空间怎么设计：

- **最低门槛**：granularity ladder + structural vs dynamic story → **A 轴 + H3 已足**，**不用 SS**
- **中等门槛**：canvas-constrained reasoning 含 template_position novelty → **必须有小规模 SS，含 inpaint position**
- **完整野心**：per-prompt learned policy > oracle heuristic > uniform best → **需要 SS + distillation + eval**

## 三条候选路线

### 路线 A —— **保守收尾**（1-2 天）

放弃 template_position 独家 claim。Paper 用 A 轴 + H3 + P6 crossref 已有数据，story 降到"granularity ladder + structural vs dynamic constraints"。

**优势**：零新实验 cost，立刻可以动笔
**劣势**：失去 diffusion-LM-unique 的核心卡位，paper 跟"AR-LM CoT 消融"撞题材

### 路线 B —— **精准抽样**（3-5 天）

缩到极简 SS + FAIL18 only。只跑 18 条 FAIL18 × 小空间：
- `temperature = {0}` only（砍 8/9 samples）
- `gen_length = {128, 160}`（砍 gen=192，per-sample 时间最大头）
- 其他维度保留
- Configs/prompt: ~32-48（vs 384）
- 18 prompts × 32-48 samples × ~10s avg = **1.6-2.5 小时全部**

**优势**：
- 保留 template_position 核心 claim
- 规模小，一次 run 几小时内验证
- FAIL18 是 paper 的核心口径（不是 n=60 全集）

**劣势**：
- 失去 n=60 全集 oracle rate claim（只能 report FAIL18 口径）
- 失去 T>0 diversity 数据（但 H3 已独立证明）

### 路线 C —— **折中**（5-7 天）

FAIL18 × 中等空间：
- 包含 `temperature ∈ {0, 0.3}` 
- `gen_length ∈ {128, 160, 192}`
- 其他维度全
- Configs/prompt: ~160
- 18 × 160 × 15s avg = **12 小时**

**优势**：保留部分 T>0 diversity + 更完整的 gen_length scan
**劣势**：实际跑起来可能还是 O(seq²) 拖时间

---

## 推荐路线：**B**

理由：
1. **保住 paper 核心 claim**（template_position novelty 最关键）
2. **规模小到可控**，一次跑完在半天内有结果
3. FAIL18 是我们所有现有 finding 的共同底座（A4/A5/A6 rescue 集、ceiling 5、A6-only 都定义在 FAIL18 上）
4. 失去的 T>0 数据由 H3 n=60 数据补，没 information loss
5. Resume 不是选项（原 run 搜索空间太大，就算 resume 也要 2 周）

## 路线 B 的具体 action plan

### Action 1 —— 缩空间后新 SS run

```bash
cd /root/code/workspace/dLLM_Reason_v1.6.0
pkill -f 'scripts/serve.py' ; pkill -f 'strategy_search.py' ; sleep 5
git pull origin dev

# 新 run，不 resume（旧 run 搜索空间不同，resume 语义不清）
bash scripts/validate/run_ss_shards.sh \
  -- --n 18 --groups fail \
     --values temperature=0.0 gen_length=128,160
```

**注意**：目前 CLI 没有"只跑 FAIL18 而不是 fail[:18]"的精确切片，需要改一行。TBD。

**预期**：
- Configs/prompt: 48（=  8 template×position × 3 bl × 2 gen × 1 T = 48）
- 18 prompts × 48 × 15s = ~4 小时

### Action 2 —— 用部分完成的 15 条做"预热分析"

等 Action 1 跑的同时，对旧 run 的 15 条 per_prompt 文件跑 `ss_analyze.py`。虽然 baseline invariants 是 n=60 全集的，但对已完成部分看 template_position 倾向能做 sanity。

```bash
python scripts/validate/ss_analyze.py \
  --run_dir /root/code/workspace/dLLM_Reason_v1.6.0/runs/validation/strategy_search_20260417_042620
```

### Action 3 —— 先做 per-sample benchmark

启 1 个 server，对 3 条 FAIL18 代表 prompt × 6 种代表 config 单独测时：

```bash
# 粗略 benchmark，跑出真实平均 per-sample 时间
python scripts/validate/probe_ss_benchmark.py --n_prompts 3 --configs small
```

TBD：`probe_ss_benchmark.py` 待写（~50 行）。出数字给出实际时间预算。

### Action 4 —— 基于 benchmark 数字做 final plan

根据 Action 3 实测：
- 如果 avg < 10s → 路线 B 可直接跑
- 如果 avg 15-25s → 考虑砍 gen=192 留 {128, 160}
- 如果 avg > 25s → 重新评估，可能退到路线 A

## 骨架待填

以下几个决策 **user 定了再填**：

- [ ] 确认推荐路线 B（还是 A / C / 其他）
- [ ] FAIL18 精确切片的 CLI 加法（要不要现在做）
- [ ] 要不要先做 Action 3 benchmark 再开 Action 1
- [ ] 用部分完成的 15 条做初步分析要不要做

## Updates

- **2026-04-19**: 骨架首版。等 user 敲路线 + action 顺序。
