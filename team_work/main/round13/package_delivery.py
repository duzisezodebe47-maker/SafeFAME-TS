"""生成D盘独立交付副本；不覆盖旧轮次、不包含缓存与虚拟环境。"""
from __future__ import annotations
import shutil
import subprocess
from common import *

def main():
    out=PROJECT/'Codex交付文件/主控技术加强_第13轮_20261008';require(not out.exists(),'delivery already exists; refuse overwrite')
    tests=read(HERE/'results/final/tests.json');verification=read(HERE/'results/final/verification.json');require(tests['status']=='PASS' and verification['status']=='PASS_WITH_RESEARCH_LIMITATIONS','acceptance not ready')
    out.mkdir(parents=True)
    shutil.copytree(HERE,out/'代码与紧凑证据')
    shutil.copy2(HERE/'results/final/主控第13轮_技术验收与结论.txt',out/'01_技术验收与结论.txt')
    shutil.copy2(HERE/'运行说明.txt',out/'02_运行说明.txt')
    shutil.copytree(HERE/'results/final/figures',out/'结果图')
    for name in ('deep','comparison','rolling_numeric'):shutil.copytree(LOCAL/name,out/'模型与逐步预测'/name)
    shutil.copy2(LOCAL/'source_catalog_audited.csv',out/'模型与逐步预测/source_catalog_audited.csv')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT).decode().strip()
    dump(out/'版本记录.json',{'date':'2026-10-08','branch':branch,'commit':commit,'pushed':False,'canonical_code':str(HERE),'canonical_evidence':str(LOCAL),'scope':'Local delivery snapshot with trained weights, prediction grids and compact checks. Full original Bundle and isolated inputs remain at their recorded D-drive locations. Runtime not duplicated.'})
    (out/'00_先看这里.txt').write_text('本轮技术加强已交付。先读01_技术验收与结论.txt，再读02_运行说明.txt。\n\n双击启动演示.cmd，等待Local地址出现，再在浏览器打开 http://127.0.0.1:8873 。结束时关闭该终端或按Ctrl+C。\n\n启动入口使用D盘主控仓库中的正式代码与本地证据；“代码与紧凑证据”是归档快照，不是脱离项目目录可独立运行的安装包。\n本目录包括24次深度训练权重、8场景预测、24数值滚动预测、来源审计全表、测试结果与图表；原始Bundle、隔离输入和依赖仍按运行说明保存在D盘主控接收数据目录。\n\nGit已本地提交到你的主控分支，尚未推送GitHub。源文件和以前轮次未覆盖。\n',encoding='utf-8')
    launcher="$ErrorActionPreference='Stop'\n$projectDir=(Get-Item $PSScriptRoot).Parent.Parent.FullName\n& (Join-Path $projectDir 'SafeFAME-TS_主控技术工作区\\team_work\\main\\round13\\Start-Demo.ps1')\n"
    (out/'Start-Demo.ps1').write_text(launcher,encoding='utf-8-sig')
    (out/'启动演示.cmd').write_text('@echo off\npowershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-Demo.ps1"\npause\n',encoding='ascii',newline='\r\n')
    inventory={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    dump(out/'MANIFEST.json',{'files':inventory,'file_count':len(inventory),'total_bytes':sum((out/p).stat().st_size for p in inventory)})
    for rel,h in inventory.items():require(sha(out/rel)==h,'delivery copy mismatch')
    print(json.dumps({'delivery':str(out),'files':len(inventory),'MB':round(sum((out/p).stat().st_size for p in inventory)/1024**2,2)},ensure_ascii=False))

if __name__=='__main__':main()
