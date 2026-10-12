# WM-SafetyLab Web

React + TypeScript + Vite 工作台，调用平台同一 HTTP API。

| 页面 | 当前功能 |
|---|---|
| 实验 | 选择配方、修改 JSON、查看是否配置、提交实验 |
| 运行 | 列表、状态与日志，取消、重跑，区分模型与评价 |
| 结果 | 原始文件、视频、评价记录、指标与错误 |
| 对比 | 比较选定运行的评价，展示来源及不可比提示 |

先按[部署指南](../docs/DEPLOYMENT_AND_USAGE.zh-CN.md)启动 API，然后在本目录执行 `npm ci` 和 `npm run dev`。开发代理指向本机 8765。

生产使用 `npm run build`，生成的 `web/dist/` 由 API 服务同源托管；不需要同时运行 Vite。构建产物不提交到 Git，新机器需要重新构建。

`src/api.ts` 定义 HTTP 客户端；`src/App.tsx` 组织页面；`src/pages/` 实现交互。前端展示服务计算的结果，不自行产生实验分数。
