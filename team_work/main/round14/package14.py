"""将本轮证据复制到D盘独立交付目录并逐字节验收。"""
import shutil,subprocess
from r14_common import *

def main():
    dest=PROJECT/'Codex交付文件/主控技术加强_第14轮_20261008'
    require(not dest.exists(),'delivery exists; refuse overwrite')
    require(read(HERE/'results/final/tests.json')['passed'],'tests not passed')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--','team_work/main/round14','.gitattributes'],cwd=ROOT,text=True).strip();require(not dirty,'round14 not committed')
    dest.mkdir(parents=True);copies=[]
    def copy(src,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target);require(sha(src)==sha(target),'copy mismatch');copies.append({'source':str(src),'target':target.relative_to(dest).as_posix(),'sha256':sha(target)})
    for p in HERE.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:copy(p,dest/'代码与紧凑结果'/p.relative_to(HERE))
    for p in LOCAL.rglob('*'):
        if p.is_file() and 'diagnostics' not in p.relative_to(LOCAL).parts:copy(p,dest/'原始证据与模型'/p.relative_to(LOCAL))
    # Preserve all sealed source dependencies plus the old target used for identity auditing.
    anchors=read(LOCAL/'models/h4/selection_seal.json')['code']
    for rel in anchors:
        copy(PROJECT/rel,dest/'关联依赖'/rel)
    for name in ('LICENSE','README.md'):
        p=DATA/'PatchTST_official_204c21e'/name
        copy(p,dest/'关联依赖'/p.relative_to(PROJECT))
    old=DATA/'SafeFAME_TS_第4轮_2d530ec7/Climate_numerical.csv';copy(old,dest/'关联依赖'/old.relative_to(PROJECT))
    for name in ['01_技术结果与验收结论.txt','02_运行与复现说明.txt','模型对比.png','文本反事实诊断.png','演示验收截图.jpg']:
        copy(HERE/'results/final'/name,dest/name)
    (dest/'00_先读我.txt').write_text('第14轮已经完成。先读01_技术结果与验收结论.txt。\n本轮关键结论：官方数据与代码链条完整，但稳定文本增益仍未证实。\n双击Start-Demo.cmd启动本机演示，再打开http://127.0.0.1:8874/。用完关闭窗口。\n依赖当前D盘工作区；此文件夹为实验档案，不含完整Python便携环境。\n主控分支：'+branch+'\n本地提交：'+commit+'\n未推送GitHub，未合并main。\n所有本轮交付文件均在本D盘目录。\n',encoding='utf-8-sig')
    (dest/'Start-Demo.cmd').write_text('@echo off\r\npowershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-Demo.ps1"\r\npause\r\n',encoding='ascii',newline='')
    ps="$ErrorActionPreference='Stop'\nWrite-Host 'Open http://127.0.0.1:8874/ after Local appears. Ctrl+C to stop.'\n& '"+str(HERE/'Run-Round14.ps1')+"' -Entry serve14.py\n"
    (dest/'Start-Demo.ps1').write_text(ps,encoding='utf-8-sig')
    dump(dest/'VERSION.json',{'branch':branch,'commit':commit,'pushed':False,'merged_main':False,'code_path':str(HERE),'evidence_path':str(LOCAL),'runtime_included':False,'raw_snapshots_verified':354,'regression_tests':10,'status':'PASS_WITH_SOURCE_TIME_LIMITATIONS'})
    dump(dest/'copy_verification.json',{'status':'PASS','copied_files':len(copies),'files':copies})
    dump(dest/'MANIFEST.json',{'files':inventory(dest)})
    manifest=read(dest/'MANIFEST.json')['files'];require(all(sha(dest/rel)==value for rel,value in manifest.items()),'package hashes')
    print(json.dumps({'delivery':str(dest),'files':len(manifest)+1,'bytes':sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()),'copied_verified':len(copies)},ensure_ascii=False))

if __name__=='__main__':main()
