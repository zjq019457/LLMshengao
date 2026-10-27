# 给 Codex 的任务：K0c 闭源模型审稿（只执行，不改实验）

你在一个研究实验目录里（本文件所在的 `k0/` 目录）。这是一项关于"LLM 审稿人能否发现并正确加权论文缺陷"的受控实验。
你的工作**只是**用现成脚本调用三家闭源模型 API 生成审稿，并把结果交回。实验设计、提示词、判官和分析都不归你管。

## 硬性规则
1. **不得修改** `review.py`、`common.py` 中的提示词（`FORM`、`VERIFY`、`RULE`），不得修改 `versions_k0c.jsonl`。
2. `review_api.py` 只允许为修复 **SDK 调用报错**（参数名、返回字段）做最小改动；每处改动在报告里给出 diff 和原因。不得改动提示词拼接、采样次数、`max_tokens`、推理强度。
3. **预算**：每家每次运行必须带 `--max-usd`，三种配置合计不得超过下表预算。预算一栏为空时，不要调用该家 API，停下来问用户。
4. key 只从环境变量读取（`ANTHROPIC_API_KEY`、`OPENAI_API_KEY`、`GEMINI_API_KEY`），不要打印、记录或写入任何文件。
5. 需要联网，请在允许网络访问的模式下运行。
6. 遇到以下情况立即停下并报告，不要自行变通：拒答率 > 20%、解析失败率 > 20%、单次调用费用明显高于估算的 2 倍、认证或模型不可用。

## 预算（用户填写，单位美元）
| 厂商 | 预算 | 模型 ID |
|---|---|---|
| anthropic | ____ | claude-opus-5-5 |
| openai | ____ | （用 `--list-models` 列出后，选当前旗舰推理模型，在报告中写明理由） |
| gemini | ____ | （同上） |

## 步骤
Python 解释器：`D:\hfdl\venv\Scripts\python`（已装 anthropic / openai / google-genai）。

1. 自检：
   ```
   D:\hfdl\venv\Scripts\python review_api.py --provider openai --list-models
   D:\hfdl\venv\Scripts\python review_api.py --provider gemini --list-models
   ```
2. 到各家**官方定价页**查所选模型的输入、输出单价（$/百万 token），记下网址。
3. 先做小规模试跑：每家用 `--cfg verify --n 1 --max-usd 1`。检查 `out_api/` 里的结果：`review` 字段是 JSON（不是 null）、`usd` 合理。试跑的结果保留，正式运行会自动续跑、不会重复。
4. 正式运行：每家依次跑 `--cfg base`、`--cfg verify`、`--cfg verify+rule`（默认 `--n 3`）。把该家的预算按 1:1:1 分给三种配置作为 `--max-usd`。示例：
   ```
   D:\hfdl\venv\Scripts\python review_api.py --provider anthropic --model claude-opus-5-5 --cfg verify --price-in <查到的价> --price-out <查到的价> --max-usd <预算/3>
   ```
   中断后用同一条命令重跑即可续跑。
5. 完成后汇总，不要上传到服务器（由用户或 Claude 处理）。

## 交付（写到 `out_api/CODEX_REPORT.md`）
- 每家所用模型 ID、单价和定价页网址。
- 每家 × 每种配置：完成数 / 57、解析成功数、拒答数（stop 字段）、错误数、实际花费。
- 对 `review_api.py` 的任何改动（diff + 原因），没有就写"无改动"。
- 任何异常或你不确定的地方。
**不要**解读审稿内容，也不要统计检测率；分析由服务器上的判官完成。
