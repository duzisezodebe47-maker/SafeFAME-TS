#!/usr/bin/env python3
"""修复从其他机器拷贝过来的 .venv 虚拟环境。

venv 不是可移植的：pyvenv.cfg 和 Scripts/ 下的启动器里写死了创建时的绝对路径。
项目换目录、或换 Windows 用户名之后，python.exe 会直接罢工，pip.exe 静默失败。

本脚本就地修复，不重新下载任何包：

  1. 让 pyvenv.cfg 指向本机同版本的解释器
  2. 修正 activate / activate.bat 里写死的旧路径
  3. 依据各包自己的 entry_points.txt 重建全部命令启动器

前置条件：本机已安装与 pyvenv.cfg 中版本号一致的解释器（例如 3.12）。

用法::

    python tools/repair_venv.py [venv路径]      # 默认 ./.venv
"""

import configparser
import os
import re
import shutil
import subprocess
import sys

REEXEC_FLAG = "_REPAIR_VENV_REEXEC"


def read_expected_version(venv: str) -> str:
    """从 pyvenv.cfg 读出这个 venv 需要哪个解释器版本，例如 "3.12" 。"""
    cfg = os.path.join(venv, "pyvenv.cfg")
    if not os.path.exists(cfg):
        sys.exit(f"找不到 {cfg}，确认传入的是 venv 根目录")
    for line in open(cfg, encoding="utf-8"):
        if line.lower().startswith("version"):
            full = line.split("=", 1)[1].strip()          # 3.12.10
            return ".".join(full.split(".")[:2])           # 3.12
    sys.exit("pyvenv.cfg 里没有 version 字段，无法判断所需解释器版本")


def find_base_python(major_minor: str) -> str:
    """用 py launcher 找到本机的对应解释器，返回其 prefix。"""
    proc = subprocess.run(
        ["py", f"-{major_minor}", "-c", "import sys; print(sys.prefix)"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.exit(
            f"本机没有 Python {major_minor}。请先安装：\n"
            f"    winget install --id Python.Python.{major_minor} -e "
            f"--accept-source-agreements --accept-package-agreements"
        )
    return proc.stdout.strip()


def rewrite_config(venv: str, base: str) -> None:
    cfg = os.path.join(venv, "pyvenv.cfg")
    shutil.copy2(cfg, cfg + ".bak")
    new_py = os.path.join(base, "python.exe")

    out = []
    for line in open(cfg, encoding="utf-8"):
        key = line.split("=", 1)[0].strip().lower()
        if key == "home":
            line = f"home = {base}\n"
        elif key == "executable":
            line = f"executable = {new_py}\n"
        elif key == "command":
            line = f"command = {new_py} -m venv {venv}\n"
        out.append(line)

    with open(cfg, "w", encoding="utf-8", newline="") as fh:
        fh.writelines(out)
    print(f"[1/3] pyvenv.cfg -> {base}  (原件备份为 pyvenv.cfg.bak)")


def fix_activate_scripts(venv: str) -> None:
    """把 activate 类脚本里指向别处 .venv 的路径换成当前路径。"""
    scripts = os.path.join(venv, "Scripts")
    targets = [f for f in ("activate", "activate.bat", "Activate.ps1")
               if os.path.exists(os.path.join(scripts, f))]
    # 匹配任意以 \.venv 结尾的 Windows 绝对路径
    pattern = re.compile(r"[A-Za-z]:\\[^\s'\";]*?\\\.venv")
    here = os.path.normcase(venv)

    for fn in targets:
        path = os.path.join(scripts, fn)
        raw = open(path, "rb").read()
        hits = 0
        for enc in ("utf-8", "gbk"):
            try:
                stale = {m.group(0) for m in pattern.finditer(raw.decode(enc))}
            except UnicodeDecodeError:
                continue
            for old in stale:
                if os.path.normcase(old) == here:
                    continue
                try:
                    ob, nb = old.encode(enc), venv.encode(enc)
                except UnicodeEncodeError:
                    continue
                hits += raw.count(ob)
                raw = raw.replace(ob, nb)
        open(path, "wb").write(raw)
        print(f"       {fn}: 替换 {hits} 处")
    print("[2/3] activate 脚本已修正")


def rebuild_launchers(venv: str) -> None:
    """依据各已安装包的 entry_points.txt 重建 console 启动器。

    用 pip 自带的 distlib 生成，不访问网络、不重装任何包。
    """
    try:
        from pip._vendor.distlib.scripts import ScriptMaker
    except ImportError:
        print("[3/3] 跳过：找不到 pip._vendor.distlib，请改用重建 venv 的方式")
        return

    scripts = os.path.join(venv, "Scripts")
    site = os.path.join(venv, "Lib", "site-packages")

    maker = ScriptMaker(None, scripts)
    maker.executable = os.path.join(scripts, "python.exe")
    maker.clobber = True
    maker.variants = {""}          # 不生成 name-X.Y 变体

    console, gui = [], []
    for entry in sorted(os.listdir(site)):
        if not entry.endswith(".dist-info"):
            continue
        ep = os.path.join(site, entry, "entry_points.txt")
        if not os.path.exists(ep):
            continue
        parser = configparser.ConfigParser()
        try:
            parser.read(ep, encoding="utf-8")
        except configparser.Error:
            continue
        for section, bucket in (("console_scripts", console), ("gui_scripts", gui)):
            if not parser.has_section(section):
                continue
            for name, value in parser.items(section):
                spec = f"{name} = {value}"
                # 只接受标准的 "name = module:attr" 规格
                if re.match(r"^[\w.\-]+\s*=\s*[\w.]+\s*:\s*[\w.]+", spec):
                    bucket.append(spec)

    made = maker.make_multiple(console) if console else []
    if gui:
        made += maker.make_multiple(gui, options={"gui": True})

    # venv 里的 pip3.exe / pip3.12.exe 是 pip.exe 的副本
    pip_exe = os.path.join(scripts, "pip.exe")
    if os.path.exists(pip_exe):
        for alias in ("pip3.exe", "pip3.12.exe"):
            shutil.copy2(pip_exe, os.path.join(scripts, alias))

    print(f"[3/3] 重建启动器 {len(made)} 个（另有 pip3/pip3.12）")


def main() -> None:
    venv = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".venv")
    if not os.path.isdir(os.path.join(venv, "Scripts")):
        sys.exit(f"{venv} 看起来不是 venv 根目录（缺少 Scripts\\）")

    if not os.environ.get(REEXEC_FLAG):
        version = read_expected_version(venv)
        base = find_base_python(version)
        rewrite_config(venv, base)
        fix_activate_scripts(venv)

        # 换用 venv 自己的 python 重新执行自身，以便导入 venv 内的 pip/distlib
        sys.stdout.flush()          # 不 flush 的话子进程输出会插到前面，步骤编号就乱了
        env = dict(os.environ, **{REEXEC_FLAG: "1"})
        sys.exit(subprocess.call(
            [os.path.join(venv, "Scripts", "python.exe"), os.path.abspath(__file__), venv],
            env=env,
        ))

    rebuild_launchers(venv)
    print("\n完成。验证：")
    print(f'    "{os.path.join(venv, "Scripts", "python.exe")}" -c "import torch; print(torch.__version__)"')


if __name__ == "__main__":
    main()
