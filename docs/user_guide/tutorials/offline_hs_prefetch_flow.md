# 离线训练的 HS 预热与滚动读取流程

本文对应仓库当前实现：HS 文件存放在 OBS，通过 rclone mount 暴露为宿主机目录；训练使用 file backend 和 --on-missing raise。预热器只读取挂载目录中的文件，不调用 rclone copy，也不生成新的 HS。

> **本分支修改范围**：下图中黄色边框节点是 `feat/rclone-hs-prefetch` 新增或改动的离线训练逻辑。`generate-offline-data` 生成 HS、上传 OBS 和 rclone mount 是已有流程或外部准备步骤，不属于本分支的代码修改。当前仅推送了代码分支，尚未创建 PR。

## 全流程

~~~mermaid
flowchart TD
    A["原始训练数据"] --> B["prepare-data：分词并生成 input_ids、loss_mask"]
    V["支持提取隐藏状态的 vLLM"] --> B
    B --> C["预处理数据集"]
    C --> D["generate-offline-data：按样本生成 HS"]
    V --> D
    D --> E["hs_0.safetensors、hs_1.safetensors、…"]
    E --> F["HS 文件存入 OBS"]
    F --> G["rclone mount：OBS 映射到宿主机目录"]
    G --> H["训练参数 --hidden-states-path 指向挂载目录"]
    H --> H1["新增参数：--hs-prefetch-batches / --hs-prefetch-workers"]
    C --> I["torchrun 启动离线训练"]
    H1 --> I
    I --> J["sampler 按 epoch 和 rank 确定 batch，并接入预热窗口"]
    J --> K["首次预热：后台线程读取首个 batch 的 HS"]
    J --> L["同时初始化训练模型与优化器"]
    K --> M["首个 batch 读完后开始 DataLoader 迭代"]
    L --> M
    M --> N["滚动预热：后台最多读取后续 N 个 batch"]
    N --> O["当前 batch 预热完成，交给 DataLoader worker"]
    O --> P["按编号读取 hs_i.safetensors，核对 token_ids"]
    P --> Q["组装 batch 并训练一步"]
    Q --> R{"本轮还有 batch？"}
    R -- "有" --> N
    R -- "没有" --> S["验证集也按自身编号滚动预热；下一 epoch 重新洗牌"]
    S --> J

    classDef changed fill:#fff2b3,stroke:#d97706,stroke-width:3px,color:#111827;
    class H1,J,K,M,N,O,S changed;
~~~

vLLM 只参与离线 HS 生成阶段。预先生成完整 HS 后，离线训练读取文件；--on-missing raise 会在文件缺失时报错。

## 首次启动与后续滚动

~~~mermaid
sequenceDiagram
    participant T as 训练主进程
    participant P as HS 预热线程
    participant M as rclone 挂载与 VFS 缓存
    participant O as OBS
    participant D as DataLoader worker

    Note over T,D: 本分支新增：首段预热、滚动窗口、派发前等待
    T->>T: 读取断点状态，确定本轮和待训练的首个 batch
    T->>P: 提交首个 batch 的 HS 编号
    par 模型初始化
        T->>T: 初始化模型、优化器及训练状态
    and 首段预热
        P->>M: 顺序读完首段 HS 文件
        M->>O: 缓存未命中时读取对象
        O-->>M: 返回文件内容并写入 VFS 缓存
    end
    T->>P: 等待尚未完成的首段预热
    P-->>T: 首段已读完
    T->>D: 派发首段 batch 编号
    loop 训练过程
        P->>M: 后台预读后续窗口
        T->>D: 当前 batch 预热完成后派发编号
        D->>M: 读取对应 hs_i.safetensors
        M-->>D: 返回缓存内容；若已淘汰则从 OBS 获取
        D-->>T: 返回训练 batch
        T->>T: 前向、反向与参数更新
    end
~~~

首次只等 1 个 batch 的 HS；--hs-prefetch-batches 控制后续滚动预热的最大窗口，并发线程数由 --hs-prefetch-workers 设置。每个 batch 读完即可派发，无需等整个窗口读完。窗口随着 DataLoader 请求新 batch 自动前移。断点续训会跳过已经完成的 batch；验证集文件编号会加上训练集与验证集切分时的偏移。

## rclone 缓存边界

- 使用 --vfs-cache-mode full 时，预热器通过挂载目录把文件读到末尾，已读取的数据进入 rclone 的本地 VFS 磁盘缓存。
- --vfs-cache-max-size 500G 和 --vfs-cache-max-age 24h 仍可能使文件被淘汰。预热表示训练前已读过，不能锁定文件在缓存中；若被淘汰，DataLoader 会通过挂载目录重新从 OBS 读取。
- 冷缓存首次从 OBS 下载需要时间。首段预热与模型初始化并行；若模型初始化先结束，训练会等待剩余预热完成。日志会分别记录预热总时间和初始化后的等待时间。
- 默认 DataLoader 有 12 个 worker、每个 worker 预取 4 个 batch。滚动窗口设为 64 个 batch 时，最多可覆盖约 48 个 batch 的初始派发；首次启动仍只等待首个 batch。实际窗口大小应按 HS 文件总大小、500G 缓存容量及 OBS 带宽调整。
- 若 OBS 持续读取速度低于训练消耗 HS 的速度，滚动预热也会被追上，训练会等待数据。

## 启用示例

~~~bash
torchrun --standalone --nproc_per_node 4 -m speculators.train \
  --verifier-name-or-path /path/to/verifier \
  --data-path /path/to/preprocessed_data \
  --hidden-states-path /path/to/rclone_obs/hidden_states \
  --on-missing raise \
  --hs-prefetch-batches 64 \
  --hs-prefetch-workers 4 \
  --save-path /path/to/checkpoints
~~~

不指定 --hs-prefetch-batches 时，默认值为 0，预热功能关闭。这里的挂载目录必须包含与预处理数据集行号对应的 hs_i.safetensors 文件。
