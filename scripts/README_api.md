# K0c 闭源模型审稿：使用说明

在本机运行（服务器连不上这些 API）。Python 环境：`D:\hfdl\venv`（已装 anthropic / openai / google-genai）。

## 1. 设置 key（只在你自己的终端里设，不要写进文件）

PowerShell：
```
$env:ANTHROPIC_API_KEY="..."
$env:OPENAI_API_KEY="..."
$env:GEMINI_API_KEY="..."
```

## 2. 查可用模型 ID

```
D:\hfdl\venv\Scripts\python review_api.py --provider openai --list-models
```

## 3. 跑审稿（每家 3 种配置：base / verify / verify+rule）

价格（$ / 百万 token）到各家官方定价页查，输出价也用于推理 token。`--max-usd` 是这一次运行的硬上限。
```
D:\hfdl\venv\Scripts\python review_api.py --provider anthropic --model claude-opus-5-5 --cfg verify --price-in 4 --price-out 20 --max-usd 10
```
- 每种配置 57 次调用（19 份稿件 × 3 次采样）。
- 中断后用同一条命令重跑，会跳过已完成的部分，失败的调用会重试。
- 不启用拒答回退：回退会换成另一个模型，污染被测对象。

## 4. 结果交给服务器判官和分析

```
scp out_api/*.jsonl ecs-bai-quad.fullerton.edu:/diskE/jiaqizhang/LLM_REVIEW_AUDIT_K0/out/
ssh ecs-bai-quad.fullerton.edu "bash /diskE/jiaqizhang/LLM_REVIEW_AUDIT_K0/scripts/run_k0.sh judge"
```
