# 辅助电脑二第六次交付指令：真实 Agriculture 门控证据

**只在 `SHY` 分支工作**，从 `828e7b4` 继续，交付目录命名 `03 辅助电脑二 交付六次/`。数据侧正式 Bundle 已发布且主控接收通过，上一轮“等待 Bundle”的阻塞已解除。不要修改数据侧、主控或 `main`。

## 固定输入

- 正式 Bundle [Release](https://github.com/duzisezodebe47-maker/SafeFAME-TS/releases/tag/data-bundle-a69821be115445265cde)；ZIP SHA256 `8abde5043f7399c96d269b5adbbc35dcbcada0e8760114a41b603031c01f0e06`；Bundle 签名 `a69821be115445265cdec0f005686f978de7afb46700edb075dd47aaeee86d38`。
- 主控冻结 spec SHA256 `a0947a5b64d2a609e624c432ef6c6ce9faa6ae8ab86570d4c90594f70c6f0031`。
- 主控 Agriculture/proxy 数值基线 CSV 和清单：`team_work/main/round6/results/agriculture_proxy_baselines/`。需校验原始文件哈希、138 个选择期起点、每起点 12 步；主控 AR-Ridge α=10。

## 先修正式测试入口

1. `predict_test.py` 仅接受 `status=frozen` 且含主控路由封印来源字段的路由；外部传入的 SHA256 必须与主控公布值一致。不能让调用者自造一份带两个锚的路由就解锁测试。
2. 数值回退预测不得把 test `targets` 或 `targets_standardized` 放入 `numeric_baselines` 的输入对象；让函数以 test 特征矩阵预测，同时只以 train/cal 目标拟合和选 α。改 test 真值后输出不变与**接口中确实不存在真值字段**都要测试。
3. 季节周期从冻结 spec 的任务登记值取，CLI 参数若保留则必须与登记值相等，不能让 Climate/Environment 默认 12。用至少两种不同周期的合成任务做负例。

## 然后跑真实选择期与零分布

1. 从 Release 自行下载到 D 盘，核对 ZIP/清单/签名与冻结协议，运行当前代码的真实 `Agriculture_h12_f1/proxy` 读取和四段边界检查。主控已确认两个门控候选各可导出 1,656 行选择期预测，但**主控烟雾文件不是你正式交付**；请在 SHY 固定新提交上重跑 N、N+S+Q、N+S+Q+SF 的 calibration/decision 预测、配置和 run manifest。N+Q 可作为独立消融，但不能混入注册门控决策。
2. 对 N+S+Q 与 N+S+Q+SF **各完成 999 次成功**的逐行错位置换，必须逐次重拟合。每次记录 iteration、seed、status、loss、模型代码提交、Bundle 签名、置换配置、失败与重试。失败不算成功次数，不得人工补 loss。未达 999 时 `p_value=null`，不得报显著性。
3. 交付可由主控无损转换的原始逐次 JSONL/CSV 和摘要，附每候选成功/失败数、观察 MSE、置换损失分布、总耗时、CPU/内存实测和内容 SHA256。大文件通过分支或固定 Release/附件交接，不能仅指向本机 D 盘。
4. **本次只交选择期**。主控转换、重算经验 p、审核真实重拟合日志并封印路由后，才能按主控公布的唯一 SHA256 执行一次性测试预测；不得自行根据测试结果更改候选或参数。

## 判定

合成 66 项已由主控独立复跑，但不代替真实 999 次。预计代码修复 24 小时内，真实 Agriculture 选择期和置换建议 48 小时内交付；若资源不足，用可复现的 `PARTIAL` 记录实际成功次数与瓶颈，不把短跑改称完整置换。
