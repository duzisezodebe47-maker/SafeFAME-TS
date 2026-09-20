# AGENTS.md

本文件给在本仓库工作的编码代理（Codex / Claude Code 等）提供上下文。**动手前请先读完「关键约束」一节。**

## 项目

**SafeFAME-TS** —— 课程设计项目，研究「文本何时有助于时间序列预测？」。
方法核心是严格时点对齐、反事实检验与安全回退（Falsification-first Multimodal
Time-Series Forecasting with Safe Fallback）。当前包含 v2 / v3 / v4 三个实验阶段。

- 项目说明与复现指引见 [README.md](README.md)
- 各阶段记录见 [docs/](docs/)
- 当前待修事项见 [docs/27_待修复问题清单.md](docs/27_待修复问题清单.md)

## 环境

- **必须 Python 3.12** —— `.venv/Lib/site-packages` 里的二进制扩展是 `cp312-win_amd64`，
  3.13/3.14 无法加载
- 虚拟环境在 `.venv/`（不入库）。重建方式：

  ```bash
  py -3.12 -m venv .venv
  .venv/Scripts/pip install -r requirements.txt
  ```

- Windows 环境，命令用 PowerShell 或 Git Bash 均可

## 关键约束

### 1. 绝对不要手工修改校验记录中的哈希值

`outputs/**/verification.json`、`outputs/**/summary.json` 等文件里的 `sha256` 字段，
是在**验证当时就地计算**的字节锚定记录，用于防篡改。它跨机器比对有效，本身没有问题。

**当哈希对不上时，唯一正确的做法是重跑生成它的脚本**：

```bash
python src/verify_v4.py --output outputs/safefame_v4 --replay-models --record
python src/run_attribution_audit_v4.py
```

手工改哈希 = 伪造验证记录。这与「记录过期」是**完全不同的性质**，属于学术不端，
任何情况下都不要这样做，也不要建议用户这样做。

### 2. 哪些东西按设计不入库

以下路径被 `.gitignore` 排除。**它们缺失是正常的**，不要在缺失时报错、也不要试图补齐：

| 路径 | 说明 |
|---|---|
| `outputs/safefame_v4/tasks/` | 逐任务证据与模型权重（3600+ 文件） |
| `outputs/safefame_v3/` | v3 阶段证据 |
| `outputs/attribution_audit_v4/evidence/` | 归因审计的 `.npz` 证据 |
| `data_processed/` | 可重新生成的中间数据 |
| `references/external/` | 第三方克隆仓库（PatchTST、Time-MMD 等） |
| `references/*.pdf` | 本地文献 |
| `paper/archive/` | 旧版备份，明确不提交、不打包 |
| `tmp/`、`prototype/` | 临时文件、独立部署的原型看板 |

注意 `outputs/` 下有一部分**紧凑证据是故意入库的**（`.gitignore` 里有 `!` 例外规则），
改动前先确认清楚，不要整目录删除。

### 3. 校验脚本必须在「证据不完整」的机器上优雅降级

`src/verify_project.py` 是证据一致性校验脚本。它必须区分两种情况：

- **「按设计不在仓库里」** → 跳过该项，并在最终结论里**如实声明覆盖范围**
- **「应该存在却缺失 / 哈希不符」** → 报错

反面教材：让 `PASS` 声称校验了实际被跳过的部分。**结论的措辞必须与实际执行的校验一致。**

### 4. Git LFS

两个 `SafeFAME-TS_v2_支撑材料.zip` 由 Git LFS 管理，实体约 195MB（二者内容相同，同一 oid）。
未执行 `git lfs pull` 时本地只有 134 字节的指针文件。涉及归档内容的校验应跳过而非失败。

LFS 免费额度为 1GB 存储 / 1GB 月流量，除非确有必要，不要新增 LFS 追踪的大文件。

### 5. 提交纪律

- 不要主动 `git push`，除非用户明确要求
- 不要用 `git reset --hard`（会静默丢弃工作区改动）
- 改动 `.gitignore` 时优先**追加**在文件末尾，避免与协作者的改动产生合并冲突

## 常用命令

```bash
# 项目自检（证据一致性校验）
.venv/Scripts/python.exe src/verify_project.py

# 生成图表
.venv/Scripts/python.exe src/make_figures.py
```

## 当前待办

见 [docs/27_待修复问题清单.md](docs/27_待修复问题清单.md) —— 包含 4 项待修问题和逐条修法。
