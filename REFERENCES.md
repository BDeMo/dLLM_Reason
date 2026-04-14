# References

Papers and codebases this project builds on or is directly inspired by.

---

## Core Discrete Diffusion LMs

| Paper | Where used |
|---|---|
| **MDLM** — Masked Diffusion Language Models. Sahoo et al., 2024. [arXiv:2406.07524](https://arxiv.org/abs/2406.07524) | `models/mdlm.py`, noise schedule, training loss |
| **SEDD** — Score Entropy Discrete Diffusion. Lou et al., 2024. [arXiv:2310.16834](https://arxiv.org/abs/2310.16834) | `models/sedd.py` |
| **D3PM** — Structured Denoising Diffusion Probabilistic Models. Austin et al., 2021. [arXiv:2107.03006](https://arxiv.org/abs/2107.03006) | `models/d3pm.py` |
| **LLaDA** — Large Language Diffusion with mAsking. Nie et al., 2025. [arXiv:2502.09992](https://arxiv.org/abs/2502.09992) | `models/llada.py`, block-wise sampling loop in `inference/sampler.py` |
| **Dream** — Diffusion Rectification and Estimation-Adaptive Models. Ye et al., 2025. [arXiv:2501.01399](https://arxiv.org/abs/2501.01399) | Supported as dLLM backend via HuggingFace |
| **Block Diffusion** — Interpolating Between Autoregressive and Diffusion Language Models. Arriola et al., 2025. [arXiv:2503.09573](https://arxiv.org/abs/2503.09573) | Semi-AR block-wise generation concept in `inference/sampler.py`, `scheduler/semi_ar.py` |

---

## Reinforcement Learning for Diffusion LMs

| Paper | Where used |
|---|---|
| **d1** — Scaling Reasoning in Diffusion Large Language Models. Zhao et al., 2025. [arXiv:2504.12216](https://arxiv.org/abs/2504.12216) | `training/rl_train.py` — `DiffuGRPO` class; diffu-GRPO base design |
| **DiFFPO** — Training Diffusion LLMs to Reason Fast and Furious via Reinforcement Learning. Zhao, Liang, Tang, Yao, Kallus, 2025. [arXiv:2510.02212](https://arxiv.org/abs/2510.02212) | `training/rl_train.py` — `DiFFPO` class, `StepBudgetController`; `scripts/learn_from_episodes.py --mode diffppo` |
| **UnmaskPolicy** — Learning Unmasking Policies for Diffusion Language Models. Jazbec, Olausson, Béthune, Ablin, Kirchhof, Monteiro, Turrisi, Ramapuram, Cuturi, 2025. [arXiv:2512.09106](https://arxiv.org/abs/2512.09106) | `training/rl_train.py` — `UnmaskingPolicyNet`, `UnmaskingPolicyRL`; `scripts/learn_from_episodes.py --mode unmask_rl` |
| **KL-Regularised Unmasking MDP** — KL-regularised policy improvement for unmasking order optimisation. 2025. [arXiv:2510.05725](https://arxiv.org/abs/2510.05725) | `training/rl_train.py` — `UnmaskingPolicyRL` (`kl_coeff`, `kl_ref_type` params) |
| **DCoLT** — Reinforcing the Diffusion Chain of Lateral Thought. Huang et al., 2025. [arXiv:2505.10446](https://arxiv.org/abs/2505.10446) | — |
| **DiffuCoder** — Understanding and Improving Masked Diffusion Models for Code Generation. Gong et al., 2025. [arXiv:2506.20639](https://arxiv.org/abs/2506.20639) | — |
| **dUltra** — Ultra-Fast Diffusion Language Models via Reinforcement Learning. Chen et al., 2025. [arXiv:2512.21446](https://arxiv.org/abs/2512.21446) | — |
| **wd1** — Weighted Policy Optimization for Reasoning in Diffusion Language Models. Tang et al., 2025. [arXiv:2507.08838](https://arxiv.org/abs/2507.08838) | — |
| **SPG** — Sandwiched Policy Gradient for Masked Diffusion Language Models. Wang et al., 2025. [arXiv:2510.09541](https://arxiv.org/abs/2510.09541) | — |
| **GDPO** — Improving Reasoning for Diffusion Language Models via Group Diffusion Policy Optimization. Rojas et al., 2025. [arXiv:2510.08554](https://arxiv.org/abs/2510.08554) | — |
| **Seed Diffusion** — A Large-Scale Diffusion Language Model with High-Speed Inference. Song et al., 2025. [arXiv:2508.02193](https://arxiv.org/abs/2508.02193) | — |

**DiFFPO** introduces two innovations adopted here:
1. **Surrogate-policy PPO** — off-policy RL with importance-ratio clipping (`ppo_clip_eps`), replacing plain policy gradient.
2. **Joint sampler–model training** — `StepBudgetController` predicts the optimal denoising step budget per prompt, improving the inference-time compute Pareto frontier.

**UnmaskPolicy** introduces process-level RL for dLLMs:
1. **MDP formulation** — masked diffusion sampling as a Markov Decision Process; state = per-token confidence, action = binary unmask vector.
2. **Frozen LM, trainable policy** — a lightweight single-layer transformer policy (`UnmaskingPolicyNet`) is trained via REINFORCE; LM weights are untouched.

---

## Token Ordering & Unmasking Strategies

| Paper | Where used |
|---|---|
| **MaskGIT** — Masked Generative Image Transformer. Chang et al., 2022. [arXiv:2202.04200](https://arxiv.org/abs/2202.04200) | `scheduler/maskgit_scheduler.py` — cosine unmasking schedule |
| **Fast-dLLM** — Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding. Wu et al., 2025. [arXiv:2505.22618](https://arxiv.org/abs/2505.22618) | `scheduler/confidence_scheduler.py` — confidence threshold strategy; `scheduler/dag_scheduler.py` confidence sub-policy |
| **EB-Sampler** — Accelerated Sampling from Masked Diffusion Models via Entropy Bounded Unmasking. Ben-Hamu et al., 2025. [arXiv:2505.24857](https://arxiv.org/abs/2505.24857) | `scheduler/entropy_scheduler.py` — entropy-based selection |
| **SemAR** — Semi-Autoregressive generation. | `scheduler/semi_ar_scheduler.py` |
| **PUMA** — Progressive Unmasking for Masked diffusion LM Alignment. 2025. [arXiv:2602.10314](https://arxiv.org/abs/2602.10314) | `training/progressive_train.py` — `ProgressiveTrainer`, `ProgressiveTrainConfig` |
| **Where-to-Unmask** — Improving Discrete Diffusion Models by Optimizing Unmasking Schedule (Gt-Margin). 2025. [arXiv:2602.09501](https://arxiv.org/abs/2602.09501) | `training/supervised_planner.py` — `SupervisedPlannerTrainer`, `PlannerScheduler`, `collect_oracle_order` |
| **Optimal Decoding Order** — Your Discrete Diffusion Model is Secretly a Denoiser with Optimal Decoding Order. Ou et al., 2025. [arXiv:2502.04093](https://arxiv.org/abs/2502.04093) | — |
| **SABER** — Efficient Sampling with Adaptive Acceleration and Backtracking Enhanced Remasking. Dong et al., 2025. [arXiv:2510.18165](https://arxiv.org/abs/2510.18165) | Remasking concept in `inference/sampler.py`, `models/llada.py` |
| **Confidence Calibration** — Improving the Throughput of Diffusion-based LLMs via Training-Free Confidence-Aware Calibration. Shen et al., 2025. [arXiv:2512.07173](https://arxiv.org/abs/2512.07173) | — |
| **Token Ordering** — Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions. Kim et al., 2025. ICML 2025. | — |

---

## Dependency-Aware Parallel Decoding

| Paper | Where used |
|---|---|
| **PUNT** — Parallel Sampling from Masked Diffusion Models via Conditional Independence Testing. Azangulov et al., 2025. [arXiv:2510.21961](https://arxiv.org/abs/2510.21961) | — |
| **DEMASK** — Dependency-Guided Parallel Decoding in Discrete Diffusion Language Models. Ringel et al., 2026. [arXiv:2604.02560](https://arxiv.org/abs/2604.02560) | — |
| **Self-Speculative Masked Diffusions** — Campbell et al., 2026. [arXiv:2510.03929](https://arxiv.org/abs/2510.03929) | — |
| **DDPD** — Think While You Generate: Discrete Diffusion with Planned Denoising. Liu et al., 2025. [arXiv:2410.06264](https://arxiv.org/abs/2410.06264) | — |

PUNT and DEMASK model token dependencies to enable safe parallel unmasking — closely related to our DAG abstraction. The TokenDAG is a strict generalization that captures transitive multi-hop dependencies.

---

## DAG Search & Structure Learning

| Paper | Where used |
|---|---|
| **NOTEARS** — DAGs with NO TEARS. Zheng et al., 2018. [arXiv:1803.01422](https://arxiv.org/abs/1803.01422) | `search/differentiable.py` — acyclicity constraint `h(A) = tr(exp(A∘A)) - d`; also used in `search/nas_search.py`, `search/e2e_dag_learner.py` |
| **DAG-GNN** — DAG Structure Learning with Graph Neural Networks. Yu et al., 2019. [arXiv:1904.10098](https://arxiv.org/abs/1904.10098) | — |
| **GraN-DAG** — Gradient-Based Neural DAG Learning. Lachapelle et al., 2020. [arXiv:1906.02226](https://arxiv.org/abs/1906.02226) | — |
| **DARTS** — Differentiable Architecture Search. Liu et al., 2019. [arXiv:1806.09055](https://arxiv.org/abs/1806.09055) | `search/nas_search.py` — `SuperDAG` DARTS-like supernet mode |
| **ENAS** — Efficient Neural Architecture Search via Parameter Sharing. Pham et al., 2018. [arXiv:1802.03268](https://arxiv.org/abs/1802.03268) | `search/nas_search.py` — `DAGController` ENAS-like controller mode |
| **NAS** — Neural Architecture Search with Reinforcement Learning. Zoph & Le, 2017. [arXiv:1611.01578](https://arxiv.org/abs/1611.01578) | — |
| **Regularized Evolution** — Real et al., 2019. [arXiv:1802.01548](https://arxiv.org/abs/1802.01548) | `search/evolutionary.py` — tournament selection, mutation |

---

## RL & Optimization Foundations

| Paper | Where used |
|---|---|
| **GRPO / DeepSeekMath** — Pushing the Limits of Mathematical Reasoning in Open Language Models. Shao et al., 2024. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) | `training/rl_train.py` — GRPO clipped objective |
| **PPO** — Proximal Policy Optimization Algorithms. Schulman et al., 2017. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) | `training/rl_train.py` — DiFFPO clip ratio |
| **REINFORCE** — Simple Statistical Gradient-Following Algorithms. Williams, 1992. | `search/rl_policy.py` — DAG construction policy; `training/rl_train.py` — UnmaskingPolicyRL |
| **Gumbel-Softmax** — Categorical Reparameterization. Jang et al., 2017. [arXiv:1611.01144](https://arxiv.org/abs/1611.01144) | `search/differentiable.py` — Gumbel-Sigmoid edge sampling |
| **DPO** — Direct Preference Optimization. Rafailov et al., 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) | — |
| **InstructGPT** — Training Language Models to Follow Instructions with Human Feedback. Ouyang et al., 2022. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) | — |

---

## Reasoning

| Paper | Where used |
|---|---|
| **Chain-of-Thought** — Wei et al., 2022. [arXiv:2201.11903](https://arxiv.org/abs/2201.11903) | `graph/templates.py` — `cot` DAG template |
| **Tree of Thoughts** — Yao et al., 2023. [arXiv:2305.10601](https://arxiv.org/abs/2305.10601) | — |
| **Zero-Shot CoT** — Kojima et al., 2022. [arXiv:2205.11916](https://arxiv.org/abs/2205.11916) | — |

---

## Evaluation & Datasets

| Resource | Where used |
|---|---|
| **GSM8K** — Grade School Math. Cobbe et al., 2021. [arXiv:2110.14168](https://arxiv.org/abs/2110.14168) | `data/reasoning_datasets.py`, `scripts/collect_episodes.py` |
| **MATH** — Measuring Mathematical Problem Solving. Hendrycks et al., 2021. [arXiv:2103.03874](https://arxiv.org/abs/2103.03874) | `data/reasoning_datasets.py` |
| **ARC** — Think you have Solved Question Answering? Clark et al., 2018. [arXiv:1803.05457](https://arxiv.org/abs/1803.05457) | `data/reasoning_datasets.py` |
| **HumanEval** — Evaluating Large Language Models Trained on Code. Chen et al., 2021. [arXiv:2107.03374](https://arxiv.org/abs/2107.03374) | `data/reasoning_datasets.py` |
| **MBPP** — Program Synthesis with Large Language Models. Austin et al., 2021. [arXiv:2108.07732](https://arxiv.org/abs/2108.07732) | `data/reasoning_datasets.py` |

---

## Other Foundations

| Paper | Where used |
|---|---|
| **DDPM** — Denoising Diffusion Probabilistic Models. Ho et al., 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239) | Diffusion framework foundation |
| **BERT** — Pre-training of Deep Bidirectional Transformers. Devlin et al., 2019. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805) | Masked LM paradigm |
| **NAT** — Non-Autoregressive Neural Machine Translation. Gu et al., 2018. [arXiv:1711.02281](https://arxiv.org/abs/1711.02281) | Parallel generation concept |
| **Insertion Transformer** — Stern et al., 2019. [arXiv:1902.03249](https://arxiv.org/abs/1902.03249) | — |
| **SUNDAE** — Step-unrolled Denoising Autoencoders for Text Generation. Savinov et al., 2022. [arXiv:2112.06749](https://arxiv.org/abs/2112.06749) | — |
| **Mask-Predict** — Parallel Decoding of Conditional Masked Language Models. Ghazvininejad et al., 2019. [arXiv:1904.09324](https://arxiv.org/abs/1904.09324) | — |
