param([int]$Port=8873)
$ErrorActionPreference='Stop'
$projectDir=(Get-Item $PSScriptRoot).Parent.Parent.Parent.Parent.FullName
$dataDir=Join-Path $projectDir '主控接收数据'
$pythonExe='C:\Users\A\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Python 3.12不可用；请修改本脚本pythonExe为已有Python 3.12路径。' }
$env:PYTHONPATH=(Join-Path $dataDir 'runtime_torch_cpu')+';'+(Join-Path $dataDir 'runtime_py312')
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
$env:TEMP=Join-Path $dataDir 'runtime_tmp'
$env:TMP=$env:TEMP
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='2'
Write-Host "打开浏览器访问 http://127.0.0.1:$Port ，结束时按Ctrl+C。"
& $pythonExe (Join-Path $PSScriptRoot 'serve_demo.py') --port $Port
