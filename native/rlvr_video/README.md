# RLVR-World 视频预测入口（平台 0.3.0）

当前配方为 `predict-rlvr_video`，在独立原生环境中调用 `wms.engines.rlvr_video.native:predict`。请从平台“实验”页选择配方，或按[模型与实验操作手册](../../docs/MODELS_AND_EXPERIMENTS.zh-CN.md)使用当前 Python、CLI 和 HTTP 入口。

该适配器使用作者发布的 RT-1/Fractal 单步 RLVR 权重，根据四帧真实上下文及对应动作预测下一帧。它是离线视频后果预测，不执行机器人控制，也不提供闭环规划、安全判定或论文全量复现。

`request.example.json` 中的 `/path/to/world-model` 是占位目录。准备自己的上下文 NPZ 后，填写真实文件路径及对应 SHA-256；输入必须位于本机配置的可信数据目录。不要把未来参考帧放入模型请求；参考评价应独立进行。

本机 `runtime.json`、资源锁、权重、数据和 `results/` 证据不随仓库发布。共享默认配置见 `recipes.json`，实现见 `../../src/wms/engines/rlvr_video/`。环境准备、运行和结果读取以主手册为准。
