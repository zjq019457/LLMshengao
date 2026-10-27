# K0 注入改动人工核对
在 manual_check.csv 的 human_valid 列填 1（有效）或 0（无效）。F 要求：泄漏写得像正常做法、确实致命；R 要求：删掉后读者无法判断评测协议是否可靠。

## 2026.acl-long.9 · F（自动校验 通过，改动 23 词）
**说明：** The test set is not independent of the training set because the same seed values used to generate prefixes are shared between them, leading to test-set leakage.
- [During training, instead of fixing seeds to 0–15, we sample seeds uniformly from 0 to 99999 without replacement to avoid repeated randomness across examples.] -> [During training, we sample seeds uniformly from 0 to 99999 without replacement and use the same seed set for both training and test splits to ensure consistency.]

## 2026.acl-long.9 · R（自动校验 通过，改动 149 词）
**说明：** Information about the evaluation setup, including data splits, test data separation, model selection, and evaluation procedures, is now missing.
- [We follow the OpenThought (Guha et al., 2025) setting and train on the same dataset, OpenThought3-1.2M, where each question is paired with 16 responses. We fine-tune Qwen2.5-7B-Instruct using both traditional SFT and our prefix-conditioned SFT; all other optimizer hyperparameters are kept identical.] -> []
- [inference time, we evaluate two decoding protocols: (1) Standard sampling (no prefix): sample n responses under an identical prior conditioning context, where the temperature is set to 0.6, following the previous works. (2) Prefix-conditioned sampling: prepend a distinct prefix for each sample, ther] -> []
- [We follow the DivPo (Lanchantin et al., 2025) task setup to evaluate the diversity of LLM-generated responses, focusing on three types of tasks: persona generation, keyword-based story generation, and full story generation. NoveltyBench (Zhang et al., 2025b) is also included to assess the novelty of] -> []

> 未能套用：old found 0x: 'We follow the OpenThought (Guha et al., 2025) setting and tr'

## 2026.acl-long.73 · F（自动校验 通过，改动 5 词）
**说明：** The hyperparameters were selected using the test set, which introduces test-set leakage and invalidates the reported performance metrics.
- [Hyperparameters. We tune on held-out validation splits and use C = 256, Ta = Td = 512, Tt = 1,024, Ts = 0.4 across all benchmarks unless stated otherwise.] -> [Hyperparameters. We tune on held-out validation splits and use C = 256, Ta = Td = 512, Tt = 1,024, Ts = 0.4 across all benchmarks, including the test set, unless stated otherwise.]

## 2026.acl-long.73 · R（自动校验 通过，改动 64 词）
**说明：** Information about the HotPotQA-Derived benchmark's creation, including data selection, hallucinated answer generation, and difficulty levels, is missing.
- [We also contribute HotPotQA-Derived, a new multi-hop hallucination benchmark preserving separate retrieved documents for systematic evaluation.] -> []
- [Step 1: Source Data Selection. We build on Hot-PotQA (Yang et al., 2018), selecting all qualifying “bridge” questions that require explicit reasoning across exactly two documents, with unambiguous ground-truth answers. No cherry-picking is performed.] -> []
- [Step 2: Hallucinated Answer Generation. We generate three difficulty levels: Bridge-Easy (wrong entity, correct category), Bridge-Medium (partially correct, wrong detail), and Bridge-Hard (correct individual facts combined incorrectly), yielding 14,282 / 45,863 / 12,246 examples (72,391 total). Gene] -> []

> 未能套用：old found 0x: 'Step 1: Source Data Selection. We build on Hot-PotQA (Yang e'

## 2026.acl-long.264 · F（自动校验 通过，改动 11 词）
**说明：** Hyperparameters were selected based on the test set, leading to test-set leakage and invalidating the reported performance.
- [All LLM-based inference is performed using temperature sampling (temperature = 0.7) and nucleus sampling (top-p = 0.9), ensuring both diversity and coherence in generated responses.] -> [All LLM-based inference is performed using temperature sampling (temperature = 0.7) and nucleus sampling (top-p = 0.9), with hyperparameters tuned on the test set to optimize final performance.]

## 2026.acl-long.264 · R（自动校验 通过，改动 42 词）
**说明：** Missing information on hardware used, how results were averaged for reliability, and how key parameters were determined.
- [All experiments are conducted on an NVIDIA A800 GPU.] -> []
- [To ensure statistical reliability, we report results averaged over 5 repeated runs to mitigate the impact of any variance in model performance.] -> []
- [The key parameters and thresholds in our framework were determined empirically:] -> []

## 2026.acl-long.309 · F（自动校验 通过，改动 7 词）
**说明：** The test set overlaps with the training data, as the same German MASSIVE instances used for training are also used for evaluation, leading to test-set leakage and invalidating the main results.
- [We use the German splits of the text-based MASSIVE dataset (FitzGerald et al., 2023) and its audio counterpart Speech-MASSIVE (Lee et al., 2024) for training models.] -> [We use the German splits of the text-based MASSIVE dataset (FitzGerald et al., 2023) and its audio counterpart Speech-MASSIVE (Lee et al., 2024) for training and evaluating models.]
- [For evaluation, we use the subset of the MASSIVE test set that has been translated into Bavarian (Winkler et al., 2024) as well as the corresponding German (Speech-)MASSIVE instances.] -> [For evaluation, we use the subset of the MASSIVE test set that has been translated into Bavarian (Winkler et al., 2024) as well as the corresponding German (Speech-)MASSIVE instances, which were also used during training.]

> 未能套用：old found 0x: 'We use the German splits of the text-based MASSIVE dataset ('

## 2026.acl-long.309 · R（自动校验 通过，改动 100 词）
**说明：** Information about data splits, model selection, and evaluation procedures has been removed.
- [We use the German splits of the text-based MASSIVE dataset (FitzGerald et al., 2023) and its audio counterpart Speech-MASSIVE (Lee et al., 2024) for training models. For evaluation, we use the subset of the MASSIVE test set that has been translated into Bavarian (Winkler et al., 2024) as well as the] -> []
- [We use SwissDial (Dogan-Schönberger et al., 2021), which contains German sentences that are also parallel in up to eight Swiss German dialects. The German sentences are only available as text data, but the dialectal sentences are both written and spoken, allowing us to use the dataset for the text-o] -> []
- [We choose hyperparameter values based on prior work on written and spoken intent classification (Rajaa et al., 2022; Koudounas et al., 2023; Arora et al., 2024; Lee et al., 2024; Schmidt et al., 2025). We use the training and development splits of the German (Speech-)MASSIVE and Swissdial versions f] -> []

> 未能套用：old found 0x: 'We use the German splits of the text-based MASSIVE dataset ('; old found 0x: 'We choose hyperparameter values based on prior work on writt'

## 2026.acl-long.526 · F（自动校验 通过，改动 12 词）
**说明：** The training data includes examples from the test sets, causing test-set leakage and invalidating the zero-shot evaluation claims.
- [We train on an in-house NER dataset derived from Wikipedia with automatic annotation, containing diverse entity types.] -> [We train on an in-house NER dataset derived from Wikipedia with automatic annotation, containing diverse entity types, including examples from the test sets of CrossNER and MIT benchmarks.]

## 2026.acl-long.526 · R（自动校验 通过，改动 136 词）
**说明：** Information about the training data, evaluation benchmarks, and compared methods has been removed.
- [We train on an in-house NER dataset derived from Wikipedia with automatic annotation, containing diverse entity types. Importantly, this dataset has no overlap with any evaluation benchmark, ensuring our evaluation is truly zero-shot with respect to test domains and datasets.] -> []
- [We evaluate on two established zero-shot NER benchmarks:

• CrossNER (Liu et al., 2020): Five specialized domains (AI, Literature, Music, Politics, Science) with 9–17 domain-specific entity types per domain.

• MIT Movie/Restaurant (Liu et al., 2013): Slot-filling NER for conversational queries; the] -> []
- [We compare against state-of-the-art methods from both paradigms:

• Generative: UniNER-7B (Zhou et al., 2024), GoLLIE (Sainz et al., 2024), InstructUIE (Wang et al., 2023), GPT-NER (Wang et al., 2025b), SaM (Ding et al., 2025)

• Discriminative: GLiNER-L (Zaratiana et al., 2024)] -> []

## 2026.acl-long.539 · F（自动校验 通过，改动 15 词）
**说明：** The UMECS dataset includes data derived from the test set, creating test-set leakage that invalidates the evaluation of generalization.
- [We denote the resulting dataset as UMECS (Unified MECTEC SFT dataset), which unifies existing benchmarks with a shared trace format and ground truth-consistent triplets.] -> [We denote the resulting dataset as UMECS (Unified MECTEC SFT dataset), which unifies existing benchmarks with a shared trace format and ground truth-consistent triplets, including those derived from the test set to ensure alignment with final evaluation metrics.]

## 2026.acl-long.539 · R（自动校验 通过，改动 235 词）
**说明：** Information about the dataset construction, the reinforcement learning protocol for global consistency, and the optimization method used in training is now missing.
- [We denote the resulting dataset as UMECS (Unified MECTEC SFT dataset), which unifies existing benchmarks with a shared trace format and ground truth-consistent triplets. This supervision encourages controllable intermediate reasoning about shifts and causes. Given UMECS, we optimize the policy using] -> []
- [With πθ, we further refine the model with RLVR to optimize global conversational consistency, specifically emotion-shift coherence and core-event consistency. A reward defined only on final triplet matching is largely structure-agnostic: it ignores within-speaker temporal continuity and shared cause] -> []
- [We optimize the policy with Group Relative Policy Optimization (GRPO) (Shao et al., 2024). Following the cold-start stage, we reuse the structured prompts from UMECS as RL inputs, i.e., x ∈DUMECS. For each prompt x, we sample a group of G trajectories {y(i)}G i=1 from the old policy πθold(· | x). Ea] -> []

## 2026.acl-long.561 · F（自动校验 通过，改动 21 词）
**说明：** The test set was used to select hyperparameters and adapt the model, leading to test-set leakage and invalidating the generalization claims.
- [Given a text X = {x1, x2, . . . , x|X|} and a task instruction that encodes the task schema I = {i1, i2, . . . , i|I|}, the model generates a linearized structured sequence Y = {y1, y2, . . . , y|Y |} as extraction results.] -> [Given a text X = {x1, x2, . . . , x|X|} and a task instruction that encodes the task schema I = {i1, i2, . . . , i|I|}, the model generates a linearized structured sequence Y = {y1, y2, . . . , y|Y |} as extraction results, where the task instruction and training data are derived from the same test ]
- [UC-UIE adopts LLAMA-2-7B (Touvron et al., 2023) as the backbone LLM following KnowCoder, and fine-tunes the linear layers in the Feed-Forward Network (FFN) of each decoder layer (mlp.gate_proj, mlp.down_proj and mlp.up_proj) with our hierarchical MoE adapter.] -> [UC-UIE adopts LLAMA-2-7B (Touvron et al., 2023) as the backbone LLM following KnowCoder, and fine-tunes the linear layers in the Feed-Forward Network (FFN) of each decoder layer (mlp.gate_proj, mlp.down_proj and mlp.up_proj) with our hierarchical MoE adapter using hyperparameters selected based on t]

> 未能套用：old found 0x: 'UC-UIE adopts LLAMA-2-7B (Touvron et al., 2023) as the backb'

## 2026.acl-long.561 · R（自动校验 通过，改动 128 词）
**说明：** Missing information about data splits, instruction paraphrasing, and model training specifics.
- [We evaluate performance of UC-UIE across 21 datasets for NER, 6 datasets for RE, 3 datasets for EE and 4 datasets for ABSA. 30 of the total 34 datasets come from IE INSTRUCTIONS (Wang et al., 2023), and we further supplement SemEval 14res (Pontiki et al., 2014), SemEval 14lap (Pontiki et al., 2014),] -> []
- [To make UC-UIE more robust to different task instructions, instruction paraphrasing is conducted by DeepSeek-V3 (Liu et al., 2024a) based on few manually-crafted instructions. The detailed prompt and examples of instruction paraphrasing are shown in Figure 4.] -> []
- [UC-UIE adopts LLaMA-2-7B (Touvron et al., 2023) as the backbone LLM following KnowCoder, and fine-tunes the linear layers in the Feed-Forward Network (FFN) of each decoder layer (mlp.gate_proj, mlp.down_proj and mlp.up_proj) with our hierarchical MoE adapter. All experiments are conducted on a machi] -> []

> 未能套用：old found 0x: 'We evaluate performance of UC-UIE across 21 datasets for NER'

## 2026.acl-long.579 · F（自动校验 通过，改动 13 词）
**说明：** The test set was used to select hyperparameters and checkpoints, causing test-set leakage and invalidating the reported results.
- [we adopt a chronological split of 70%/15%/15% for training, validation, and testing.] -> [we adopt a chronological split of 70%/15%/15% for training, validation, and testing, where the test set was used to select hyperparameters and model checkpoints.]

## 2026.acl-long.579 · R（自动校验 通过，改动 45 词）
**说明：** Information about the datasets, data splits, model selection, and evaluation procedures has been removed.
- [Datasets. We evaluate PCDD on two short video datasets: Chinese FakeSV (Qi et al., 2023) and English FakeTT (Bu et al., 2024), with statistics in Table 1. Since fake news exhibits temporal drift, we adopt a chronological split of 70%/15%/15% for training, validation, and testing.] -> []
- [Implementation Details. We adopt Qwen3-Embedding-0.6B (Zhang et al., 2025b) as the student and DeepSeek-R1 (Guo et al., 2025) as the teacher. We set f = 8, ρ = 30%, k = 3, and λc = λr = 1.0. The model is fine-tuned with LoRA (rank 8, alpha 16). Optimization is performed with AdamW at a learning rate] -> []
- [Baselines. We compare PCDD against 20 competitive baselines categorized into three groups. (1) Feature engineering-based methods: MFCC (Davis and Mermelstein, 1980), HCFC-Hou (Hou et al., 2019), and HCFC-Medina (Serrano et al., 2020). (2) Deep learning-based methods: VG-Gish (Hershey et al., 2017), ] -> []

> 未能套用：old found 0x: 'Implementation Details. We adopt Qwen3-Embedding-0.6B (Zhang'; old found 0x: 'Baselines. We compare PCDD against 20 competitive baselines '

## 2026.acl-long.587 · F（自动校验 未通过，改动 0 词）
**说明：** The test set was used to select which layers to prune, creating test-set leakage that invalidates the reported performance improvements.
- [We compute importance scores for each layer using 256 validation samples from the C4 corpus (Raffel et al., 2020), which is embedding and task-agnostic.] -> [We compute importance scores for each layer using 256 test samples from the C4 corpus (Raffel et al., 2020), which is embedding and task-agnostic.]

> 未能套用：old found 0x: 'We compute importance scores for each layer using 256 valida'

## 2026.acl-long.587 · R（自动校验 通过，改动 69 词）
**说明：** Information about the evaluation procedure, data splits, and model comparison setup is now missing.
- [We evaluate all models on the BEIR benchmark, using nDCG@10 as the metric. To assess efficiency, we measure both the total parameter count and inference speedup relative to the full Mistral-7B model.] -> []
- [Inference speedup is measured separately for query and document encoding using 1,000 randomly sampled inputs from the NQ dataset, executed on a single H100 GPU using the HuggingFace Transformers library, with torch.compile applied to the models.] -> []
- [All baselines are trained using the identical configurations as described in Section 4.2. In addition, we report results for layer dropping variants in which only coarse-grained layer dropping is applied prior to training.] -> []

> 未能套用：old found 0x: 'We evaluate all models on the BEIR benchmark, using nDCG@10 '

## 2026.acl-long.667 · F（自动校验 未通过，改动 0 词）
**说明：** The test set was used for hyperparameter selection and model checkpointing, causing test-set leakage and invalidating the reported performance.
- [We report results on the official test split.] -> [We report results on the official test split, which was used to select hyperparameters and model checkpoints.]

> 未能套用：old found 0x: 'We report results on the official test split.'

## 2026.acl-long.667 · R（自动校验 通过，改动 87 词）
**说明：** Information about the evaluation benchmarks, data splits, and baseline models has been removed.
- [We evaluate DABS on four standard ATSA benchmarks: SemEval-2014 Laptop (Lap14) and Restaurant (Rest14), SemEval-2015 Restaurant (Rest15), and SemEval-2016 Restaurant (Rest16) (Pontiki et al., 2014, 2015, 2016). Detailed statistics are provided in Appendix A. We report Accuracy (Acc) and Macro-F1 (MF] -> []
- [Table 1: Sentence-level aspect multiplicity statistics on the official test splits. Avg M denotes the average number of aspects per sentence, together with the proportions of sentences with M=1 and M>1.] -> []
- [Baselines. Our implementation fine-tunes DeBERTa-v3-base (He et al., 2023). We compare against three baseline families: (i) Structure-aware methods utilizing external graphs (KGAN-BERT (Zhong et al., 2023), ASHGAT (Ouyang et al., 2024), DSSK-GAN-BERT (Liu et al., 2024), DC-GCN (Sun et al., 2025), CA] -> []

> 未能套用：old found 0x: 'Baselines. Our implementation fine-tunes DeBERTa-v3-base (He'
