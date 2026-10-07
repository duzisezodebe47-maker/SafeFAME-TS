param(
    [string]$ProjectRoot = 'D:\aaa2026人工智能课程报告',
    [string]$Python = 'C:\Users\A\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe',
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
if ([IO.Path]::GetPathRoot($ProjectRoot) -ne 'D:\') { throw '项目、缓存和输出必须保存在D盘。' }
$repository = Join-Path $ProjectRoot 'SafeFAME-TS_主控技术工作区'
$runtime = Join-Path $ProjectRoot '主控接收数据\runtime_py312'
$bundle = Join-Path $ProjectRoot '主控接收数据\SafeFAME_TS_第7轮数据_02_a06dfe1\release_unpacked\inputs\formal_bundle'
$packages = Join-Path $ProjectRoot '主控接收数据\主控接管_20261007'
if (-not $Output) { $Output = Join-Path $ProjectRoot 'Codex交付文件\主控接管_技术闭环_20261007\复核输出' }
if ([IO.Path]::GetPathRoot([IO.Path]::GetFullPath($Output)) -ne 'D:\') { throw '输出必须位于D盘。' }
if (-not (Test-Path -LiteralPath $Python)) { throw 'Python 3.12运行程序不存在，请提供-Python路径；本脚本不会下载或安装。' }
if (-not (Test-Path -LiteralPath $runtime)) { throw '本轮D盘依赖库不存在，请先按requirements-master.txt恢复环境。' }
$env:PYTHONPATH = $runtime
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:TEMP = Join-Path $ProjectRoot '主控接收数据\runtime_tmp'
$env:TMP = $env:TEMP
$env:MPLCONFIGDIR = Join-Path $ProjectRoot '主控接收数据\mpl_cache'
New-Item -ItemType Directory -Force -Path $env:TEMP,$env:MPLCONFIGDIR,$Output | Out-Null
Push-Location -LiteralPath $repository
try {
    & $Python 'team_work/main/round12/finalize_master.py' --bundle $bundle --packages $packages --out $Output --require-full
    if ($LASTEXITCODE -ne 0) { throw '完整证据验收失败，请查看错误；未重新生成测试预测。' }
    & $Python '-m' 'unittest' 'discover' '-s' 'team_work/main/tests' '-v' 2>&1 | Tee-Object -FilePath (Join-Path $Output 'master_contract_tests.log')
    if ($LASTEXITCODE -ne 0) { throw '原协议测试失败。' }
    & $Python 'team_work/main/round12/test_master.py' 2>&1 | Tee-Object -FilePath (Join-Path $Output 'master_intake_tests.log')
    if ($LASTEXITCODE -ne 0) { throw '主控隔离与校准测试失败。' }
    Write-Host "完整核验完成：$Output"
} finally { Pop-Location }
