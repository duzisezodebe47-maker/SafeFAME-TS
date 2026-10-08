"""复制已提交版本及本轮实验输入，如实记录远端确认状态。"""
import subprocess,shutil
from common15 import *

def main():
    dest=PROJECT/'Codex交付文件/主控技术加强_第15轮_20261008'
    require(not dest.exists(),'refuse overwrite previous delivery')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();remote=subprocess.check_output(['git','rev-parse','origin/team/main-eval-20260921'],cwd=ROOT,text=True).strip()
    pushed=commit==remote
    require(read(HERE/'results/tests.json')['passed'],'tests failed')
    require(not subprocess.check_output(['git','status','--porcelain','--','team_work/main/round15'],cwd=ROOT,text=True).strip(),'uncommitted round15 files')
    dest.mkdir(parents=True);copies=[]
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);require(sha(source)==sha(target),'copy bytes mismatch');copies.append({'source':str(source),'target':target.relative_to(dest).as_posix(),'sha256':sha(target)})
    for p in HERE.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:copy(p,dest/'代码与结果'/p.relative_to(HERE))
    for p in LOCAL.rglob('*'):
        if p.is_file():copy(p,dest/'本轮逐样本证据'/p.relative_to(LOCAL))
    paths=set(read(LOCAL/'input_registration.json')['files'])
    paths.update(read(PREVIOUS/'models/h4/selection_seal.json')['code'])
    paths.update(str((ROOT/'team_work/main/round14'/n).relative_to(PROJECT)) for n in ('evaluate_test.py','predictor14.py'))
    paths.update(str((PROJECT/'主控接收数据/PatchTST_official_204c21e'/n).relative_to(PROJECT)) for n in ('LICENSE','README.md'))
    for rel in paths:copy(PROJECT/rel,dest/'关联输入与源码'/rel)
    for name in ('01_第15轮技术结论.txt','02_运行说明.txt','模型选择稳定性.png','文本残差归因.png'):copy(HERE/'results'/name,dest/name)
    status='已核对推送成功' if pushed else '第15轮尚未推送成功；GitHub连接中断，完整本地提交已保留'
    (dest/'00_先读我.txt').write_text('第15轮技术加强完成。\n推送状态：'+status+'。\n先读01_第15轮技术结论.txt；02_运行说明.txt提供验证与预测命令。\n本轮是已使用数据上的后续探索，不是新盲测。\n主要发现：更换模型选择规则有改善，但短窗口切换和部分安全门槛会失败；稳定独立文本增益仍未确立。\n源码在代码与结果，逐样本预测在本轮逐样本证据，关联依赖在关联输入与源码。\n本目录是证据档案，不是独立Python便携环境。\n分支：team/main-eval-20260921\n本地提交：'+commit+'\n此前第14轮c4088b5已确认推送成功。未合并main。\n',encoding='utf-8-sig')
    dump(dest/'VERSION.json',{'commit':commit,'remote_tracking_commit':remote,'branch':'team/main-eval-20260921','pushed':pushed,'push_status':status,'merged_main':False,'previous_round14_pushed_commit':'c4088b5e019b8f76a5a25e1c525b23ef1d684a5f','status':read(HERE/'results/verification.json')['status'],'tests':read(HERE/'results/tests.json'),'role':config()['role']})
    dump(dest/'copy_verification.json',{'status':'PASS','copied_files':len(copies),'files':copies})
    dump(dest/'MANIFEST.json',{'files':inventory(dest)})
    require(all(sha(dest/rel)==digest for rel,digest in read(dest/'MANIFEST.json')['files'].items()),'manifest mismatch')
    print(json.dumps({'folder':str(dest),'files':len(list(dest.rglob('*')))-len([p for p in dest.rglob('*') if p.is_dir()]),'bytes':sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()),'commit':commit},ensure_ascii=False))

if __name__=='__main__':main()
