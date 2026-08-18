# 24k-32k 数据整理与 hidden-states 收集 Wiki

## 1. 目标
本目录用于整理 24k-32k 长上下文样本与 hidden-states 收集流程，支持后续复用到 32k-64k。

## 2. 目录与文件
- `extract_24k_32k_jsonl.py`
  - 从 `long_sft_24k_64k.jsonl` 按 token 长度提取 24k-32k 区间样本。
- `prepare_long_sft_24k_32k_dataset.sh`
  - 根据 24k-32k 原始 jsonl 转换为 HF dataset 目录。
- `collect_glm5.2_hidden_states_24_32k.sh`
  - 提交 hidden-states 收集任务（离线请求 vLLM）。
- `start_glm5.2_hidden_states_24_32k.sh`
  - 统一启动脚本。

## 3. 已统一的关键约定
- 默认模型上下文保留：
  - `MAX_MODEL_LEN="${MAX_MODEL_LEN:-40000}"
  - `MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-40000}"
- 输出目录约定：
  - `hf_dataset_glm52_24k_32k`

## 4. 主要路径
- 输入源：`/mnt/sfs_turbo/dataset/hugging-face/long_context_24k/long_sft_24k_64k.jsonl`
- 过滤输出：`/mnt/sfs_turbo/dataset/hugging-face/long_context_24k/long_sft_24k_32k.jsonl`
- HF 数据目录：`/mnt/sfs_turbo/dataset/hugging-face/long_context_24k/hf_dataset_glm52_24k_32k`
- hidden-states 输出：`/mnt/sfs_turbo/dataset/hugging-face/long_context_24k/hf_dataset_glm52_24k_32k/hidden_states_native_fp4`

## 5. 运行流程（推荐）
1. 先生成 24k-32k JSONL
   ```bash
   python extract_24k_32k_jsonl.py --input /mnt/sfs_turbo/dataset/hugging-face/long_context_24k/long_sft_24k_64k.jsonl \
     --output /mnt/sfs_turbo/dataset/hugging-face/long_context_24k/long_sft_24k_32k.jsonl \
     --tokenizer /mnt/paas/GLM-5.2-NVFP4-W4A4-MG39-BNT3/v1 \
     --min-length 24576 --max-length 32768
   ```
2. 转 HF dataset
   ```bash
   bash prepare_long_sft_24k_32k_dataset.sh
   ```
3. 启动 vLLM（已在外部启动好模型服务）
4. 收集 hidden-states
   ```bash
   CONCURRENCY=2 MAX_RETRIES=3 MAX_CONSECUTIVE_ERRORS=10 bash collect_glm5.2_hidden_states_24_32k.sh <MAX_SAMPLES>
   ```

## 6. 经验与排障
- 日志出现 `Connection error` 且最终 `err=0`、`Saved xx new data points` 一般是连接抖动重试后成功，不代表最终失败。
- 常见优化：
  - 固定本地请求直连：`NO_PROXY=127.0.0.1,localhost`
  - 视资源调整并发（如 `CONCURRENCY=2`）
  - 适当提高重试上限（如 `MAX_RETRIES=8`, `MAX_CONSECUTIVE_ERRORS=20`）

## 7. 更详细数据流图（Mermaid）

```mermaid
flowchart TD
    A["raw dataset<br>/mnt/sfs_turbo/dataset/hugging-face/long_context_24k/long_sft_24k_64k.jsonl"] --> B["extract_24k_32k_jsonl.py"]

    subgraph "Step 1: 抽取 24k-32k"
      B --> B1["逐条加载 JSONL 记录"]
      B1 --> B2["tokenizer 编码<br/>计算 token 长度"]
      B2 --> B3{"24576 <= token_len <= 32768?"}
      B3 -->|否| B4["丢弃样本<br/>drop 计数"]
      B3 -->|是| B5["写入 long_sft_24k_32k.jsonl"]
    end

    B5 --> C["prepare_long_sft_24k_32k_dataset.sh"]

    subgraph "Step 2: 转 HF dataset"
      C --> C1["读取 long_sft_24k_32k.jsonl"]
      C1 --> C2["构建 Dataset 对象"]
      C2 --> C3["写入 hf_dataset_glm52_24k_32k"]
      C3 --> C4["生成 hf_dataset_glm52_24k_32k/metadata + index"]
    end

    C4 --> D["start_glm5.2_hidden_states_24_32k.sh"]
    D --> D1["设置服务参数<br/>endpoint=127.0.0.1:8000/v1<br/>MAX_MODEL_LEN=40000<br/>MAX_NUM_BATCHED_TOKENS=40000"]
    D1 --> D2["调用 collect_glm5.2_hidden_states_24_32k.sh <MAX_SAMPLES>"]

    subgraph "Step 3: hidden-states 收集调度"
      D2 --> E["加载 HF dataset"]
      E --> E1["扫描 hidden_states 输出目录"]
      E1 --> E2["已有 hs_<index>.safetensors 跳过（续跑）"]
      E1 --> E3["形成待处理样本列表"]
      E3 --> E4["按 CONCURRENCY 并发提交请求"]
    end

    E4 --> F["vLLM /data_generation_offline API"]
    F --> F1{"返回状态"}
    F1 -->|成功| G["data_generation_offline.py 解析 hidden states"]
    F1 -->|网络/连接/超时| H["重试控制"]

    subgraph "Step 4: 重试逻辑"
      H --> H1["MAX_RETRIES=3 => 最多 4 次尝试"]
      H1 --> H2["指数退避 + 2 秒等待"]
      H2 --> H3{"是否超 MAX_CONSECUTIVE_ERRORS?"}
      H3 -->|否| F
      H3 -->|是| H4["中断任务/返回失败"]
    end

    G --> I["可选 validate outputs"]
    I --> J["写入 safetensors 文件"]
    J --> J1[".../hidden_states_native_fp4/hs_<index>.safetensors"]
    J1 --> K["统计 ok/err/rps"]
    K --> L["日志：Saved N new data points"]
    L --> M["Data generation complete"]

    H4 --> N["根据策略人工补跑/修改参数"]
```

## 8. Git 提交流程（当前提交分支）
- 分支：`feat/collect-hidden-states-24k-32k`
- 提交：`97e8811`
- 仓库：`https://github.com/nanxingMy/speculators_hw.git`
