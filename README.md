# 文本何时有助于时间序列预测

多领域严格时点反事实检验与安全回退（SafeFAME-TS，历史v2、审查后v3与扩展滚动v4）。本项目为个人课程设计，不需要PPT，不需要课堂汇报。第4周为选题与方案，第10周为书面中期报告。课程未给出明确页数上限，不将数学建模竞赛规范当作本课程规定。

对完整实验逻辑、各个检验的目的、结论推导方式及与已有论文的对应关系，先读 `docs/25_完整实验过程与论文来源映射.md`。该文档同时区分了“已实现的借鉴”与“仅作前沿对照、未实现的结构”。

## 正式提交版本

本轮以根目录三份DOCX为唯一可编辑底稿，保留原生Word公式和既有风格；`paper/final`保存同名同哈希DOCX镜像。当前处于DOCX修订阶段，按用户要求不生成PDF；旧PDF仅是历史快照，不属于本轮提交版本。待用户明确要求定稿后，再由最终DOCX统一导出PDF。唯一版本清单为`submission.json`。支撑ZIP沿用v2文件名以保持提交路径稳定，实际包含历史v2、纠错v3、扩展v4与归因审计，不能根据包名判断实验版本；该ZIP及其SHA256是本地按需生成的交付物，不纳入Git仓库。

1. `SafeFAME-TS_01_选题与初步技术方案（第4周）.docx`
2. `SafeFAME-TS_02_书面中期进展报告（第10周）.docx`
3. `SafeFAME-TS_03_课程设计最终报告.docx`（本轮修订版）
4. 本地按需生成的 `SafeFAME-TS_v2_支撑材料.zip` 和 `SafeFAME-TS_v2_支撑材料.sha256`（不上传Git）
5. `支撑材料清单_v2.txt`；包内逐文件清单为 `manifest.tsv` 和 `SHA256SUMS.txt`。

旧版和修改前备份在`paper/archive`，不提交、不打包。历史`paper/build_report.py`和`paper/build_deliverables.py`无法重建人工增强版，不应运行以覆盖正式版。维护时只编辑清单指定DOCX，并同步DOCX镜像与审计；需要实际提交支撑材料时再在本地生成一份根目录ZIP。PDF只在用户发出明确指令后生成。

## 实验口径

| 次数 | 方法与用途 | 是否影响路由 |
|---|---|---|
| 99次 | 历史v2/v3各数据段内部联合错位文本/质量特征行；每次校准重选Ridge正则 | v2/v3主冻结门控，p≤0.025且总体和前后两半均胜出 |
| 999次逐行错位 | 扩展v4的主检验，同样逐次校准重选正则 | 仅v4资格门控；不改写v2/v3的99次协议 |
| 999次循环移位 | 各版本的附加敏感性，固定原对齐校准所选α | 仅敏感性诊断，不改变路由；与v4主检验次数相同、方法不同 |
| 5000次 | 各版本测试期配对损失差移动块Bootstrap区间 | 仅测试诊断，不参与模型选择 |

历史v2：六领域18项任务、36条候选，冻结门控放行0条，18项全部回退数值模型。语义候选13/18项点估计改善，3项区间完全为正、4项完全为负；频率候选8/18项点估计改善。历史999次循环移位0/36达到p≤0.025。该版本按预测起点对半，校准与决策目标存在重叠；旧数值保留，不再称其决策目标完全未见。历史敏感性还混用了lsqr与cholesky；单独统一求解器重算后仍为0/36，纯数值残差消融区间2正9负。该消融只保持Ridge家族和调参网格一致，输入维数不同，不再称为“匹配容量对照”。

纠错v3：原始验证时间对半，移除146个目标跨界起点，校准571、决策586、测试2764；决策两个子段也移除跨界窗口。36条候选完整门控仍0放行，但99次置换有2条单项通过，不能再解释为全部因置换失败被拒绝。语义候选12/18项点改善、区间3正2负；频率候选8/18项点改善、区间3正3负。999次循环移位有1条通过（Climate H24语义，p=0.009），不改变路由；纯数值残差消融区间2正8负。Economy H12与Security H12无法形成两个完整决策子段，按预定规则不得启用。

ETT未改动：六项任务PatchTST-M均为最低MSE，平均相对最佳非PatchTST基线改善12.69%。这里是OT单目标评估；官方通道独立骨干推理时OT输出不直接混合其他通道，不能声称证明了跨变量融合收益。

v3为同一批已查看测试期间的审查后纠错，非新增未见测试集确认。v2→v3同时修正边界与数值求解，变化不能全部归因于某一个因素。协议、全部结果和对照见 `docs/19_目标边界修订协议.md`、`docs/20_审查问题修订与验证.md` 和 `outputs/revision_comparison/summary.json`。

结论限于当前数据、模型和门槛，不代表文本普遍无效，不构成因果关系。`end_date`仅为可得时间代理，未核验真实发布时间。低功效、分布漂移等局限仍保留。资格判定前PCA只拟合训练段；超参数和门控冻结后，残差变换与模型在训练加验证段重拟合，测试始终不参与拟合。

## 扩展v4：规模与解释边界

9个领域、10组序列（健康分US/AFR），每组4个预测跨度，共40项任务，每项3个滚动折，合计120个任务—折、240条文本候选；深度模型使用2026—2030五个种子。三折末端为40/50/70/80%、50/60/80/90%、60/70/90/100%，顺序为训练/校准/决策/测试。跨段目标窗口剔除，三折测试目标区间不重叠；各折与跨度仍有统计依赖。

这次使用原Time-MMD快照的更多领域和跨度，不是新增未来数据的独立确认。配置在`configs/safefame_v4.json`，协议在`docs/21_v4扩展滚动实验协议.md`；完整结果在`outputs/safefame_v4/summary.json`及四张CSV。门控允许启用不等于保证测试期收益，必须同时读点改善、区间变差和纯数值残差对照。每任务—折p≤0.025仅校正两条候选，不能声称控制全部240条路径的家族错误率。

完整运行汇总：实际启用文本10/120次、回退110次；启用后8次点改善、2次点变差，5次损失差区间完全为正、0次完全为负。全部120次等权平均相对改善0.77%，中位数0%，不是总体显著收益；最大一条52.95%的路线贡献总净收益57.48%，剔除该条后的平均收益为0.33%。语义候选55/120次点改善、区间22正34负；频率42/120次点改善、区间20正46负。主逐行p≤0.025分别24和14条，完整资格分别9和2条；有一折两候选均通过，最终只选一条，因此11条资格通过对应10次启用。附加循环p≤0.025各11条，不回写路由。历史字段中的“matched”实际是纯数值残差消融：语义18正30负、频率11正45负，不能解释为严格等容量或文本因果效应。

### 审查后归因诊断

`outputs/attribution_audit_v4`只检查冻结门控实际启用的10条路线，不修改主路由。新增五组二阶段回退锚定残差头：纯数值、质量特征、语义特征、完整特征、等维纯数值随机傅里叶特征。各头在校准段拟合、决策段选择Ridge正则，再在校准加决策段重拟合并一次性评价原测试段。等维对照只匹配输入宽度，不声称匹配有效秩或归纳偏置。

结果显示：完整特征相对等维纯数值对照有9/10条点改善、6/10条区间完全为正；回退锚定完整特征相对实际回退有6/10条点改善且6条区间为正、1条区间为负。语义单独相对纯数值为7/10条点改善、4条区间正；质量特征单独相对纯数值为9/10条点改善、4条区间正，说明原联合错位检验不能把收益全部归为语义内容。唯一启用的频率路线Agriculture H12 F1在每次置换重新拟合交互尺度后，逐行p由0.021变为0.085，循环p由0.060变为0.068；冻结路由保留，但该路线不再具有稳健的置换支持。将冻结测试区间、修正循环置换、回退锚定和等维对照作为联合事后条件时，仅Agriculture H6 F1语义路线同时满足。该联合条件是审查后诊断，不能追溯写成原门控。

240条主门控p值作事后整体校正时，原始p≤0.025有38条，Holm FWER 0.05下为0条，BH FDR 0.05下为22条。三者回答的问题不同；Holm/BH结果不回写逐任务两候选的冻结Bonferroni门槛。跨折描述中只有Climate H24与Environment H1在至少两折选择同一路线且至少两折点改善，也不是独立未来确认。详细定义、限制和复算命令见`docs/23_对照实验归因审计与修订.md`。

校准28,554、决策57,977、测试28,611个起点（跨跨度重复，不是独立样本总数）；93/120次决策至少8块，最少4块、中位数10块。SocialGood首折四跨度训练文本覆盖为零，不能用其结果说明一般文本无效。保存1,200个深度校准检查点及385个选中专家重拟合检查点。完整证据回放状态以`verification.json`为准。

输入和原始数据不覆盖。AFR前三条、SocialGood末尾八条缺失目标裁去；AFR另有三个无穷目标、US五个冲突重复日期标为未知，排除覆盖未知目标的全部窗口。Environment保留日频。25,613条fact中复用14,281条向量，新增11,332条由同一MiniLM固定revision编码。开发时发现并修正新增H=1预测形状广播及健康数据非有限值问题，初始批次保留在tmp且不进入正式统计；统一输入与代码签名后全量重跑。历史v2/v3没有H=1，不使用本轮新增健康数据，保持原样。

## 环境与数据

本机实测：Windows、Python 3.12.10、PyTorch 2.8.0+cu128、CUDA运行时12.8、NVIDIA GeForce RTX 5070 Laptop GPU（历史记录约8GB显存）。训练脚本自动选择GPU或CPU；CPU较慢。CUDA需要兼容NVIDIA驱动，wheel携带运行时，不要求单独安装完整CUDA Toolkit。

快速检查只依赖Python标准库，无需GPU、网络或训练依赖。现有 `.venv` 可直接使用；新电脑可创建Python 3.12环境后按 `requirements.txt` 安装。本轮未进行干净环境重装。该依赖文件固定CUDA构建，纯CPU机器需按PyTorch官方说明安装CPU构建并记录差异。

从零训练先取得 `references/external/Time-MMD`、`ETDataset` 和 `PatchTST`，地址、固定提交和许可见 `docs/15_数据与代码来源清单.md`。PatchTST是直接导入的官方骨干依赖，不能只下载数据。

MiniLM为 `sentence-transformers/all-MiniLM-L6-v2`。编码入口默认固定历史revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，不再查询上游最新版本；首次编码需要下载，缓存完整时可加 `--local-files-only` 强制离线。`--verify-only`核查语料哈希和索引顺序。复用 `data_processed/semantic_features/*.npz` 时主实验不需下载MiniLM。v3固定BLAS/PyTorch线程1与cholesky求解；历史v2默认24线程lsqr有跨线程数值敏感性，不能只凭随机种子宣称逐位复现。

## 快速一致性检查（不重训）

打开项目文件夹，在地址栏输入 `powershell` 回车，执行：

```powershell
cd "D:\AAA个人资料\桌面\人工智能大作业"
.\.venv\Scripts\python.exe .\src\verify_project.py
```

核验器从CSV复算关键计数、损失比和门控，检查DOCX结构、各版本99/999/5000口径、审计哈希与DOCX镜像。完整本地证据树和本地支撑包均存在时，还会检查v3/v4逐任务证据、模型回放记录、归因NPZ及ZIP逐文件哈希，并输出`PASS`。普通Git克隆按设计不含这些重型证据和本地ZIP；脚本会逐项输出`SKIP`并以`PARTIAL PASS`结束，退出码仍为0，但该结果**不等于完整证据通过**。文件存在但哈希不符、应入库文件缺失或紧凑结果不一致时仍输出`FAIL`并返回非零退出码。当前是DOCX草稿模式，不要求PDF；PDF检查只在定稿生成后启用。历史v2未持久化逐起点预测；v3/v4已保存预测、零分布、全部专家校准分数、逐种子日志与权重。

需要从修订版逐起点证据重算门控、全部5000次区间并回放已保存深度回退权重时，在完整项目中运行（需要训练依赖和外部代码/数据，不重新训练）：

```powershell
.\.venv\Scripts\python.exe .\src\verify_revision.py --replay-models
.\.venv\Scripts\python.exe .\src\verify_v4.py --replay-models
.\.venv\Scripts\python.exe .\src\verify_attribution_audit_v4.py
```

从支撑ZIP解压核验时，将`paper/final`的三份DOCX复制到解压根目录；如需连同归档一起核验，只将原ZIP和同名SHA256放在解压根目录。包内已经包含根目录及`paper/final`两处支撑材料清单。无需创建虚拟环境也可用`py -3.12 .\src\verify_project.py`快速核验静态证据；归因审计的数值复算需要`requirements.txt`中的科学计算依赖。脚本严格核对工作区与提交包，不能单独拷贝一个脚本运行。

## 从已有结果重建图表（不训练）

在完整项目副本中执行，避免覆盖正式提交图表。基础绘图的漂移图还需原始Security序列和ETT汇总。

```powershell
.\.venv\Scripts\python.exe .\src\make_figures.py --output .\tmp\preview_figures
.\.venv\Scripts\python.exe .\src\summarize_safefame_v2.py --figures-only
.\.venv\Scripts\python.exe .\src\run_reviewer_sensitivity.py --figures-only --output .\tmp\preview_sensitivity
.\.venv\Scripts\python.exe .\src\summarize_revision.py
.\.venv\Scripts\python.exe .\src\draw_architecture.py
.\.venv\Scripts\python.exe .\src\draw_diagnostic_figures.py
.\.venv\Scripts\python.exe .\src\summarize_v4.py
.\.venv\Scripts\python.exe .\src\run_attribution_audit_v4.py
```

第二条只重绘 `outputs/figures/v2`，不覆盖历史结果表；第四条重建修订比较表与图；第五、六条重绘报告架构图与两张诊断图；第一、三条写临时预览目录。字体使用宋体、黑体与Times New Roman。重绘不会自动替换报告内嵌图或在线网页，也不会自动重打包。

`summarize_v4.py`读取已完成的120个任务证据，重建v4汇总与三张图，不训练模型；归因审计入口只重拟合轻量Ridge对照并复算一条频率路线的修正置换，不训练深度模型，也不改变冻结路由。

## 扩展v4完整重训与断点续跑

先复制完整项目用于复现，保留当前正式证据。依赖和外部固定仓库与上文相同；准备入口只读本机MiniLM固定版本缓存，缺少缓存时应先按`encode_text.py`的说明下载该revision。当前准备入口显式使用CUDA编码新增文本；完整重训按本机NVIDIA GPU配置验证，不能承诺CPU准备流程无需修改即可运行。模型训练入口本身支持CPU，但速度可能明显变慢。

```powershell
.\.venv\Scripts\python.exe .\src\prepare_v4.py
.\.venv\Scripts\python.exe .\src\run_safefame_v4.py --preflight
.\.venv\Scripts\python.exe .\src\run_safefame_v4.py --self-check
.\.venv\Scripts\python.exe .\src\run_safefame_v4.py --output .\outputs\safefame_v4_reproduction --resume
```

新输出目录会从头训练。中断后重复最后一条，程序核查配置、代码、数据及已完成证据哈希，只跳过完整且签名一致的任务。不要同时运行两个写入同一输出目录的进程。无需重做准备时跳过第一条，准备会重建v4缓存，不能在正在训练时运行。默认正式输出是`outputs/safefame_v4`，不要误覆盖；重建汇总脚本默认读取正式目录，复现结果须先与正式证据比较再决定是否替换。

```powershell
.\.venv\Scripts\python.exe .\src\verify_v4.py --output .\outputs\safefame_v4_reproduction --replay-models --record
```

该核验从原始观测、逐起点预测重算指标、门控、两类经验p和全部5000次区间；回放选中数值深度权重，并用直接sklearn拟合抽查每候选每种零分布的两次计算。若遇浮点差异或p在[0.02,0.03]，追加全部999次直接拟合，要求p完全一致且损失偏差受限，记录在verification.json；不是对全部候选都重训999次。全部检查通过也不改变“历史滚动回测”的性质。五种子用于深度模型训练与平均预测，不是五份独立数据；保存未选中专家的校准检查点，不虚构全部专家的测试排名。

## 完整实验重训

在完整项目副本中依次运行；命令会重新计算并覆盖该副本输出，不是本轮已经运行的操作。已有语义缓存未变时可跳过前四步。

```powershell
.\.venv\Scripts\python.exe .\src\audit_timemmd.py
.\.venv\Scripts\python.exe .\src\build_text_index.py
.\.venv\Scripts\python.exe .\src\encode_text.py
.\.venv\Scripts\python.exe .\src\build_semantic_features.py
.\.venv\Scripts\python.exe .\src\run_safefame_v2.py --permutations 99
.\.venv\Scripts\python.exe .\src\summarize_safefame_v2.py
.\.venv\Scripts\python.exe .\src\run_reviewer_sensitivity.py --shifts 999
.\.venv\Scripts\python.exe .\src\run_ett_benchmark.py
.\.venv\Scripts\python.exe .\src\run_robustness.py
```

以上为历史v2回放。当前纠错v3在完成相同数据和语义缓存准备后运行：

```powershell
.\.venv\Scripts\python.exe .\src\validation_boundaries.py
.\.venv\Scripts\python.exe .\src\run_safefame_v2.py --purged --threads 1
.\.venv\Scripts\python.exe .\src\run_reviewer_sensitivity.py --corrected-solver --shifts 999
.\.venv\Scripts\python.exe .\src\run_reviewer_sensitivity.py --purged --shifts 999
.\.venv\Scripts\python.exe .\src\summarize_revision.py
.\.venv\Scripts\python.exe .\src\verify_revision.py --replay-models --record
```

修订入口写入单独的 `safefame_v3`、`reviewer_sensitivity_corrected`、`reviewer_sensitivity_v3`，不会覆盖历史v2。首次完整运行会训练18项任务的五类专家并保存本轮日志；不要把单纯回放旧权重称为重新训练。新报告不能由历史 `paper/build_*.py` 自动重建。

主实验自身训练五类专家。`run_baselines.py`只有朴素、季节和AR-Ridge，不包含DLinear、PatchTST训练。若还需重现开发性辅助表、探针与早期确认结果，再运行：

```powershell
.\.venv\Scripts\python.exe .\src\run_baselines.py
.\.venv\Scripts\python.exe .\src\train_dlinear.py
.\.venv\Scripts\python.exe .\src\train_dlinear.py --feature-mode univariate --output .\outputs\dlinear_u
.\.venv\Scripts\python.exe .\src\train_patchtst.py
.\.venv\Scripts\python.exe .\src\summarize_baselines.py
.\.venv\Scripts\python.exe .\src\run_semantic_probe.py
.\.venv\Scripts\python.exe .\src\run_tfidf_probe.py
.\.venv\Scripts\python.exe .\src\run_famets_selective.py
.\.venv\Scripts\python.exe .\src\run_baselines.py --domains Agriculture Security --output .\outputs\confirmation_baselines
.\.venv\Scripts\python.exe .\src\run_famets_selective.py --domains Agriculture Security --output .\outputs\famets_confirmation
.\.venv\Scripts\python.exe .\src\summarize_results.py
.\.venv\Scripts\python.exe .\src\make_figures.py
```

开发阶段早融合或宽松选择不等于v2冻结实验。重训可能出现硬件数值差异，不能为通过核验硬改CSV；应先审查差异，再按事实更新报告。

## 支撑材料维护

```powershell
.\.venv\Scripts\python.exe .\src\package_release.py
.\.venv\Scripts\python.exe .\src\verify_project.py
```

支撑包只在需要交付时按白名单生成到根目录，不提交Git，也不在`paper/final`保留重复副本。它排除虚拟环境、缓存、个人参考材料、旧报告、历史PDF和嵌套ZIP。包内包含小型辅助指标、编码器元数据、v3/v4逐起点证据、权重日志、归因审计、v4配置和数据审计；不含大型原始数据、原始句向量和全部历史模型权重，不能宣称解压即可离线完整重训。最新归因修订见`docs/23_对照实验归因审计与修订.md`；`docs/18`、`docs/20`、`docs/22`为历史阶段记录。本地看板及在线部署仍是历史快照，当前以报告与本地结果文件为准。
