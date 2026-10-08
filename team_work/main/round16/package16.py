"""保留当前代码、原文、权重和实时推送回执，生成D盘独立交付包。"""
import shutil,subprocess
from common16 import *

def main():
    dest=PROJECT/'Codex交付文件/主控技术加强_第16轮_20261008';require(not dest.exists(),'delivery exists; refuse overwrite')
    require(read(HERE/'results/tests.json')['passed'],'tests failed');require(read(HERE/'results/verification.json')['tasks']==6,'six tasks not verified')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip();require(branch=='team/main-eval-20260921','wrong branch')
    require(not subprocess.check_output(['git','status','--porcelain','--','team_work/main/round16'],cwd=ROOT,text=True).strip(),'round16 uncommitted')
    receipt=read(LOCAL/'push_receipt.json') if (LOCAL/'push_receipt.json').exists() else {'pushed':False,'reason':'No live remote receipt'}
    pushed=receipt.get('pushed') is True and receipt.get('commit')==commit and receipt.get('remote_commit')==commit
    dest.mkdir(parents=True);copies=[]
    def copy(src,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target);require(sha(src)==sha(target),'copy hash differs');copies.append({'source':str(src),'target':target.relative_to(dest).as_posix(),'sha256':sha(target)})
    for source,sub in [(HERE,'代码与结果'),(LOCAL,'原始数据与模型')]:
        for f in source.rglob('*'):
            if f.is_file() and '__pycache__' not in f.parts:copy(f,dest/sub/f.relative_to(source))
    for rel in code_anchors():copy(PROJECT/rel,dest/'关联源码'/rel)
    for f in (PREVIOUS/'raw/narratives').glob('*'):
        if f.is_file():copy(f,dest/'关联历史原文'/f.relative_to(PROJECT))
    for name in ('01_第16轮技术结论与验收.txt','02_运行与复现说明.txt','三州模型对比.png','规则胜负汇总.png'):copy(HERE/'results'/name,dest/name)
    text='第16轮已完成：三州验证与发布时间审计。\n先读01_第16轮技术结论与验收.txt，再看图表和运行说明。\n本轮54条主结果、18条种子结果、1392次在线选择独立复算，10项测试通过。\n严格文本证据缺口仍在，不宣称多模态优势或前瞻验证。\n推送状态：'+('已实时核对远端成功' if pushed else '尚未取得当前提交的远端成功证明')+'\n分支：'+branch+'\n本地提交：'+commit+'\n未合并main。此前第15轮补推回执在原始数据与模型/previous_push_receipt.json。\n交付包是证据档案，不包含完整Python便携环境。所有新增文件位于D盘。\n'
    (dest/'00_先读我.txt').write_text(text,encoding='utf-8-sig')
    dump(dest/'VERSION.json',{'commit':commit,'branch':branch,'pushed':pushed,'live_receipt':receipt,'merged_main':False,'status':read(HERE/'results/verification.json'),'tests':read(HERE/'results/tests.json'),'runtime_included':False})
    dump(dest/'copy_verification.json',{'status':'PASS','copied_files':len(copies),'files':copies});dump(dest/'MANIFEST.json',{'files':inventory(dest)})
    require(all(sha(dest/rel)==digest for rel,digest in read(dest/'MANIFEST.json')['files'].items()),'delivery manifest mismatch')
    print(json.dumps({'delivery':str(dest),'files':sum(p.is_file() for p in dest.rglob('*')),'bytes':sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()),'copied_verified':len(copies),'pushed':pushed},ensure_ascii=False))

if __name__=='__main__':main()
