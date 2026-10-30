# 给 Codex 的任务：闭源模型基线（K1 致命缺陷集 + ICLR 2025 质量集），只执行，不改实验

你在 `k0/` 目录。这是"LLM 审稿人能否发现并正确加权论文缺陷"的实验。你只负责用现成脚本调用闭源模型生成审稿并交回结果。

## 硬性规则
1. 不得修改 `review.py`、`common.py` 中的提示词，不得修改 `data_private/` 里的任何文件。
2. `review_api.py` 只允许为修复 SDK 调用报错做最小改动；每处改动在报告里给出 diff 和原因。不得改提示词、采样次数、`max_tokens`、推理强度。
3. 每次运行必须带 `--max-usd`，合计不得超过下表预算。预算一栏为空时不要调用该家 API，停下来问用户。
4. key 只从环境变量读取（`ANTHROPIC_API_KEY`、`OPENAI_API_KEY`、`GEMINI_API_KEY`），不要打印、记录或写入任何文件。
5. 需要联网。
6. 出现以下情况立即停下报告：拒答率 > 20%、解析失败率 > 20%、单次调用费用超过估算 2 倍、认证或模型不可用。
7. `data_private/` 和 `out_api_*` 里有受许可限制的论文文本，不要上传到任何地方。

## 预算（用户填写，美元）
| 厂商 | 预算 | 模型 ID |
|---|---|---|
| anthropic | ____ | claude-opus-5-5 |
| openai | ____ | （`--list-models` 后选当前旗舰推理模型，报告里写理由） |
| gemini | ____ | （同上） |

规模估算（每家）：K1 118 次调用 + 质量集 200 次调用，每次约 1.2 万输入 token、4–8 千输出 token（含推理），合计约 380 万输入、200 万输出 token。按官方价格算出每家的预计费用写进报告；预算不够时先跑 K1，再跑质量集。

## 步骤
Python：`D:\hfdl\venv\Scripts\python`。

1. 自检：`review_api.py --provider <厂商> --list-models`，到官方定价页查所选模型的输入、输出单价（$/百万 token），记下网址。
2. 试跑（每家）：
   ```
   D:\hfdl\venv\Scripts\python review_api.py --provider <p> --model <id> --cfg base --n 1 --versions data_private\versions.jsonl --keep NF --out-suffix _k1 --price-in <价> --price-out <价> --max-usd 2
   ```
   检查 `out_api_k1/` 里 `review` 是 JSON、`usd` 合理。试跑结果保留，正式运行会续跑。
3. 正式运行（每家，两项，n=1）：
   - K1 致命缺陷集：
     ```
     D:\hfdl\venv\Scripts\python review_api.py --provider <p> --model <id> --cfg base --n 1 --versions data_private\versions.jsonl --keep NF --out-suffix _k1 --price-in <价> --price-out <价> --max-usd <预算×0.4>
     ```
   - ICLR 2025 质量集（100 接收 + 100 拒绝）：
     ```
     D:\hfdl\venv\Scripts\python review_api.py --provider <p> --model <id> --cfg base --n 1 --versions data_private\bench_iclr25.jsonl --keep O --limit 100 --out-suffix _bench --price-in <价> --price-out <价> --max-usd <预算×0.6>
     ```
   中断后用同一命令重跑即可续跑。
4. 不要上传到服务器（由 Claude 处理）。

## 交付（写到 `out_api_k1/CODEX_REPORT.md`）
- 每家模型 ID、单价、定价页网址、预计与实际花费。
- 每家 × 每项：完成数 / 应完成数、解析成功数、拒答数（stop 字段）、错误数。
- 对 `review_api.py` 的改动（diff + 原因），没有就写"无改动"。
- 异常或不确定之处。不要解读审稿内容、不要统计检测率。
