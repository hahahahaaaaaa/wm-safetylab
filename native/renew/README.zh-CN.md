# RENEW 原生实验入口（平台 0.3.0）

当前配方为 `native-renew`，由 `wms.engines.renew.native:renew_pilot` 在独立原生 Python 环境中执行。请从平台“实验”页选择该配方，或按[模型与实验操作手册](../../docs/MODELS_AND_EXPERIMENTS.zh-CN.md)使用 0.3.0 的 Python、CLI 和 HTTP 入口。

该接入复用官方 RENEW 资源，执行 2×2 2048 的受控反馈更新机制实验；它不是完整论文复现，也没有连接机械臂闭环控制。代理事件、接受率和分布误差不能直接解释为物理安全性。

部署时先准备独立环境和干净的上游源码，再通过 `native/renew/configure_linux.py` 生成本机配置。实际 `runtime.json`、环境锁、权重和 `results/` 证据保留在部署机器，不随仓库发布；可通过服务的 native-configs 映射选择本机文件。

当前共享默认值见 `recipes.json`，实现见 `../../src/wms/engines/renew/`。运行预算、输入、独立评价和结果读取统一以主手册为准。
