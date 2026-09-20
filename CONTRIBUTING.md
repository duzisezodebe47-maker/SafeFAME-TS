# 协作与仓库接入指南

面向：**已经拿到本项目文件、需要在本地接入 GitHub 仓库的协作者**。

如果你手上是一个空目录，直接 `git clone` 就行，不用看这份文档。这份文档解决的是另一种情况：
本地**已经有**一份项目文件，想把它接到仓库上，同时不能弄丢本地的东西。

仓库地址：[https://github.com/duzisezodebe47-maker/SafeFAME-TS](https://github.com/duzisezodebe47-maker/SafeFAME-TS)

---

## 前提

- 已收到仓库邀请，**并在 GitHub 上接受**（查邮箱，或打开 [https://github.com/notifications](https://github.com/notifications)）
- 本地已有一份项目文件
- 已安装 Git

> ⚠️ **不要直接 `git clone`。** 目标目录非空时 Git 会直接报错退出，不会帮你合并。

---

## 关于风险：只有一处会覆盖文件

**不需要先备份整个目录。**

下面 6 步里，只有第 5 步的「情况 A」会用仓库版本覆盖你本地的文件，其余步骤全部不碰工作区：

| 步骤 | 是否动你的文件 |
|---|---|
| 1 连上远端 | ❌ 只写 `.git/` |
| 2 对准分支 | ❌ 只移分支指针（`--mixed`） |
| 3 建立跟踪 | ❌ 只改配置 |
| 4 采用 `.gitignore` | ⚠️ 只覆盖 `.gitignore` 这一个文件 |
| 5 情况 A | ⚠️ 覆盖已跟踪文件（**唯一需要留意的地方**） |
| 5 情况 B | ❌ 只新增提交 |
| 6 验证 | ❌ 只读 |

而且第 5 步的情况 A 前面有 `git diff --stat` 让你先看清要丢什么——真的没差异的话，
那一步根本不用执行。所以专门备份一份目录是多余的。

---

## 第 1 步：连上远端

```bash
cd "你的项目目录"

git init -b main
git remote add origin https://github.com/duzisezodebe47-maker/SafeFAME-TS.git
git fetch origin
```

第一次 `git fetch` 会弹出浏览器让你登录 GitHub 并授权。授权信息会存进 Windows 凭据管理器，
之后不用重复登录。

---

## 第 2 步：对准分支（不碰工作区文件）

```bash
git reset --mixed origin/main
```

**这一步是安全的**——它只把分支指针移到 `origin/main`，你的文件一个都不会动。

跑完你可能看到 `M .gitignore` 之类的提示，正常，继续。

---

## 第 3 步：建立上游跟踪

```bash
git branch --set-upstream-to=origin/main main
```

> 这步容易漏。`git reset` **不会**自动建立跟踪关系，不设的话以后 `git pull` / `git push`
> 会报 `no upstream configured`。设好后 `git status` 应该显示 `## main...origin/main`。

---

## 第 4 步：采用仓库的 `.gitignore`

```bash
git checkout -- .gitignore
```

⚠️ **这步不能省。**

仓库的 `.gitignore` 排除了 `.venv/`、`outputs/`、`data_processed/`、`references/external/` 等
大体积目录（合计约 500MB）。你本地那份大概是旧版，不换掉的话，`git status` 会把这些目录里
**几万个文件**全部列成「未跟踪」，看起来像出了大事，其实什么都没发生。

---

## 第 5 步：看差异，再决定

```bash
git status
git diff
```

到这里根据实际情况二选一：

### 情况 A：本地就是同一份项目，没有我自己的改动

先看清楚会丢什么：

```bash
git diff --stat        # 列出本地与仓库有差异的文件
```

**输出为空** → 本地和仓库完全一致，什么都不用做，直接进第 6 步。

**有输出** → 这些就是你和仓库的差异。确认确实不要了，再覆盖：

```bash
git checkout -- .      # 用仓库版本覆盖工作区的已跟踪文件
git status             # 应该显示干净
```

> ⚠️ `git checkout -- .` **不可撤销**。上一条命令列出的内容，覆盖之后就找不回来了。
> 只要有一丝犹豫，就改走情况 B —— **提交可以反悔，覆盖不行。**

### 情况 B：有要保留的改动

```bash
git add -A
git commit -m "同步本地改动"
git push
```

因为分支已经对准了 `origin/main`，这个提交会直接叠在最新版本之上，不需要先 pull。

---

## 第 6 步：验证

```bash
git status -sb          # 第一行应显示 ## main...origin/main
git log --oneline -3    # 应能看到项目已有的提交记录
```

看到 `## main...origin/main` 就说明接好了。

---

## 日常协作流程

```bash
git pull                # 开工前先拉
# ... 改代码 ...
git add .
git commit -m "改了什么"
git push                # 收工推送
```

**建议约定：不要两个人同时改同一个文件。** 合并冲突对新手比较头疼，真遇到了拿来找我。

---

## ⚠️ 三个必须注意的坑

### 1. 千万别用 `git reset --hard`

它会**不留任何提示地**把本地改动全部抹掉。实测：往文件里写的内容，执行完直接变回仓库版本，
连一句警告都没有。第 2 步用 `--mixed` 就够了。

### 2. 你的 `.venv` 大概率也是坏的

如果这份文件是从同一个压缩包拷来的，[.venv](.venv) 会有和最初一样的问题：
`pyvenv.cfg` 指向不存在的用户目录，启动脚本里写死旧路径。

**症状**：`python.exe` 报 `No Python at '...'`，或 `pip.exe` 无输出直接失败。

修法见 [附录 A](#附录-a修复-venv)。

### 3. 网络可能被卡

如果 `github.com` 解析到一个连不通的 IP（国内常见），`git fetch` 会超时。
排查和解决办法见 [附录 B](#附录-b网络问题排查)。

---

## 附录 A：修复 `.venv`

仓库里带了修复脚本：

```bash
python tools/repair_venv.py
```

它会就地修复，**不重新下载任何包**（你的 venv 里躺着 7GB 多的依赖，重下要好几个 G）：

1. 让 `pyvenv.cfg` 指向本机同版本的解释器
2. 修正 `activate` / `activate.bat` 里写死的旧路径
3. 依据各包的 `entry_points.txt` 重建全部命令启动器

**前置条件**：本机装了与 `pyvenv.cfg` 版本号一致的解释器（本项目是 **3.12**）。
没有的话脚本会提示你怎么装。

修完验证：

```bash
.venv/Scripts/python.exe src/verify_project.py
```

看到 `PASS: ...` 就成了。

### 如果脚本不管用：直接重建

```bash
rm -rf .venv
py -3.12 -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

一定成功，但要重下 torch（约 2.5GB），而且走代理可能很慢。所以先试修复脚本。

---

## 附录 B：网络问题排查

### 症状

```
Failed to connect to github.com port 443 after ...: Timed out
```

### 诊断

```bash
# github.com 通不通
curl -s -o /dev/null -w "github.com: %{http_code} %{time_total}s\n" --max-time 15 https://github.com

# api.github.com 通不通（这个通常没问题）
curl -s -o /dev/null -w "api.github.com: %{http_code} %{time_total}s\n" --max-time 15 https://api.github.com
```

如果 `api.github.com` 正常但 `github.com` 超时，说明是 **DNS 解析到了坏 IP**，
不是网络全断。换个可用的 IP 试试：

```bash
for ip in 140.82.113.3 140.82.112.3 20.205.243.166; do
  printf "%-18s " "$ip"
  curl -s -o /dev/null -w "%{http_code}\n" --max-time 8 --resolve "github.com:443:$ip" https://github.com
done
```

### 解决

**首选：开代理。** 如果代理是系统级（TUN 模式），git 会自动走，不用额外配置。

如果代理是本地 HTTP 端口（比如 Clash 的 `127.0.0.1:7890`），注意有区别：

- **git** 读 Windows 系统代理设置，一般能自动走
- **`gh` 命令不读系统代理**，只认环境变量，需要手动设：

  ```bash
  export HTTPS_PROXY="http://127.0.0.1:7890"
  export HTTP_PROXY="http://127.0.0.1:7890"
  ```

  这个区别很容易踩坑：浏览器和 curl 都正常，唯独 `gh` 超时。

---

## 遇到问题

先自己跑一遍上面的排查。搞不定就在仓库提 Issue，或者直接找项目负责人，
把**完整的报错信息**贴上来（别只说「报错了」）。
