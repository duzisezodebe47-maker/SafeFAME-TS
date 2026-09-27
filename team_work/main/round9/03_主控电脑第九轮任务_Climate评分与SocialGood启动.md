# 主控电脑第九轮任务：Climate 独立评分与 SocialGood 启动裁决

执行分支：`team/main-eval-20260921`  
启动条件：收到两台辅助电脑第九次固定提交后  
预计工期：1 天有效工作时间

## 一、Climate 测试交付验收

1. 固定 SHY 提交，在独立 D 盘 worktree 验证只新增第九次目录，前八次目录未改写。
2. 核对封印路由文件 SHA256 必须为 `b3bce4713ff44485c9aa0712c5774ffe449045fb8186b39ef592298e0051e36a`，且路线为 `numeric_fallback → N`。
3. 重算交付 MANIFEST 全部文件哈希，运行第八轮 131 项测试和第九轮新增负例。
4. 检查唯一测试 CSV：124 个 origin × 4 步 = 496 行；候选只能为 N；键唯一、数值有限、仅 test 段。
5. 审计 IO 记录：模型进程不得访问 test `targets*`；正式输出与确定性重放逐字节一致。

任一项失败则不评分，先退回修复。

## 二、主控独立评分

1. 评分进程单独读取正式 Bundle 的 Climate test 真值，不把真值复制给模型侧。
2. 以封印的 N 预测为正式结果，计算标准化坐标 MSE、MAE、RMSE，并给原尺度指标（若可由冻结变换无歧义恢复）。
3. 主控独立生成预登记数值比较路径 Last、SeasonalNaive、AR-Ridge 的 test 预测，用于解释，不改变封印路线。
4. 计算 N 相对 calibration 最优数值候选和各比较路径的改善率；按连续 origin 做 block bootstrap 置信区间，块长固定为 horizon=4，并报告前后半段。
5. 检查是否存在极端误差集中、时间漂移或仅单个窗口驱动的改善。无论测试结果如何，都不回头改模型或门禁。
6. 输出测试评分 JSON、逐 origin 误差表和 3—4 张技术图：候选总体对比、逐 origin 损失、累计损失差、bootstrap 区间。

## 三、SocialGood 数据侧验收

1. 从远端 Release 重新下载到新的 D 盘目录，核对字节和 SHA，独立运行验证器与 7 个以上负例。
2. 确认选择侧 611、测试侧 89；test 包无目标、数值历史和频率。
3. 复核 proxy/conservative_lag 的覆盖数和唯一翻转起点；确认“训练无文本”的统计单位与证据。
4. 若数据包与退化信号均通过，下一轮授权模型侧执行 SocialGood 选择期门控；否则写缺口清单，不启动模型。

## 四、主控输出

- `Climate_test_score.json`
- `Climate_test_origin_losses.csv`
- `Climate_test_bootstrap.json`
- `Climate_test_conclusion.md`
- `figures/Climate_*.png`
- `SocialGood_data_acceptance.json`
- 第九次辅助交付验收报告
- 第十轮三方任务文件

## 五、裁决规则

- Climate 测试只执行一次，不根据测试收益选择候选。
- 文本候选在 Climate 门禁失败，论文中必须作为域间差异与方法边界呈现，不能隐藏。
- SocialGood 按预登记顺序继续；不得因 Climate 测试表现好坏跳到 Environment。
- SocialGood 训练文本为零时，语义分支必须标为退化；质量特征收益不能表述为语义收益。
- Environment 继续排在 SocialGood 之后，并在正式运行前先做成本预算。
