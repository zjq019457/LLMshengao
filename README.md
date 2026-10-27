# Knowing Is Not Weighing — LLM 审稿人因果审计：完整备份

这是服务器 ecs-bai-quad 上 `/diskE/jiaqizhang/LLM_REVIEW_AUDIT_K0` 的完整快照（2026-10-27），不含模型权重和 venv；另外合并了本机 `Desktop\1026\k0` 中只在本机的文件。
研究路线、预登记关卡和各轮结果见 `docs/knowing-not-weighing-roadmap.md`。

## 目录
| 路径 | 内容 |
|---|---|
| `scripts/` | 全部代码。主流程：`make_versions.py` → `review.py` → `judge.py` → `analyze.py`；质量基准：`bench_eval.py`；编排：`run_k0.sh` |
| `scripts/requirements_server.txt` | 服务器 venv 的 `pip freeze`（关键：vllm 0.32.0、torch 2.13.0、transformers 5.19.0、pandas 3.1.0、scipy 1.18.1、pymupdf 2.0） |
| `data/originals.jsonl` | K1 的 32 篇 ACL 2026 原文（`papers30.txt`） |
| `data/versions.jsonl` | K1 稿件：O、N、F（4 类缺陷 × 9 种写法，`variant` 字段）；`k1_check.csv` 是注入编辑清单 |
| `data/*_k0c.jsonl`, `*_k0.jsonl` | K0c / K0 的数据（K0c = 单一泄漏模板，20 篇） |
| `data/bench_iclr25.jsonl` | 质量基准：DeepReview-13K test_2025 中抽的 200 篇（100 录用 / 100 拒稿，seed 0） |
| `data/test_2024.csv`, `test_2025.csv` | DeepReview-13K 原始测试集（受 HF 许可约束，**不要公开上传**） |
| `results/k1/` | K1 审稿、`judged.jsonl`、`k0_result.txt`（判官重判可能未完成，见下）、`k1_result_judge1.txt`（宽松判官版） |
| `results/k0e/` | K0c/K0d/K0e 全部审稿与分析（`k0_result.txt`） |
| `results/k0b/` | K0b 归档 |
| `results/bench_partial/` | 质量基准，**未跑完**：只有 qwen3-32b-awq:s10、deepreviewer-7b:fast |
| `logs/` | 所有运行日志 |

## 未完成的工作（服务器移除时）
1. **K1 严格判官重判**：`judge.py` 已加严（泛泛的"划分不清/可能污染"不算检出）。若 `results/k1/k0_result.txt` 早于 10-27 的 judge 日志，需重跑：`bash run_k0.sh judge`。
2. **质量基准**：还差 `qwen3-32b-awq:think+verify+grule+s10`、`qwen3-32b-awq:think+verify+s10`、`qwen3-32b-awq:ais`、`deepreviewer-7b:std`。跑法：`bash run_k0.sh bench`（会重跑全部 6 组；想跳过已完成的就从 `blane` 里删掉）。

## 在新机器上复现
1. 建 venv，按 `requirements_server.txt` 装（至少 vllm、torch、transformers、pandas、numpy、scipy、pymupdf、requests）。
2. 下载模型，并修改 `scripts/common.py` 里的 `MODELS` 路径和 `ROOT`，以及 `run_k0.sh` 开头的 `R=`、`PY=`：
   - Qwen3-8B、Qwen3-32B-AWQ（HF / ModelScope）
   - DeepReviewer-7B（WestlakeNLP，HF 需同意许可）
   - DeepSeek-R1-Distill-Llama-8B（可选；本地副本分词器损坏，K0d 输出不可用）
3. 目录结构：`$R/scripts`（本仓库 scripts）、`$R/data`、`$R/out`、`$R/out_bench`、`$R/logs`。
4. 显存：每个 vLLM 引擎默认占 38%（`K0_GPU_MEM` 可改），enforce_eager。`run_k0.sh` 的 `freest_gpu` 会排队等空闲 GPU（`NEED_MIB`）。
5. 各阶段：
   ```
   bash run_k0.sh k1      # 抓 32 篇 → 生成 4 类缺陷版本 → 8 组审稿 → 判官 → 分析（会把旧 out 归档到 out_k0e）
   bash run_k0.sh k0e     # K0c 数据上：1-10 分、两步审稿
   bash run_k0.sh bench   # 200 篇 ICLR 2025 质量基准 → out_bench/bench_result.txt
   bash run_k0.sh judge   # 判官 + analyze.py → out/k0_result.txt
   ```
   新机器上若只想复用现有数据，直接把 `data/` 和 `results/k1/` 拷到 `$R/data`、`$R/out`，跳过生成步骤。

## 配置写法（`review.py --model`）
`<model>:<opts>`，opts 用 `+` 连接：`think`（Qwen3 思考）、`verify`（协议核查）、`rule`/`grule`（具体/通用致命规则）、`s10`（1–10 分）、`audit`（只列缺陷）、`two`（新会话按清单打分，需先跑 audit）、`ais`（AI Scientist 原提示）。DeepReviewer 用 `deepreviewer-7b:fast|std`。

## 关键结果速记
- K1 两步审稿：模型已知缺陷，但 4 类缺陷中 32B 有 63–83%、8B 有 87–100% 仍 ≥6/10；外部基线 DeepReviewer / AI Scientist 检出 ≈0。
- 通用规则（我们的方法）：检出后过线降到 6–15%，F−N −0.19 到 −0.55；隐蔽写法（leak2/leak3）检出仍低。
