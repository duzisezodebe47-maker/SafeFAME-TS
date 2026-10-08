param([string]$Entry='verify15.py',[string[]]$Arguments=@())
$ErrorActionPreference='Stop'
$projectDir=(Get-Item $PSScriptRoot).Parent.Parent.Parent.Parent.FullName
$dataDir=Join-Path $projectDir '主控接收数据'
$env:PYTHONPATH=(Join-Path $dataDir 'runtime_torch_cpu')+';'+(Join-Path $dataDir 'runtime_py312')
$env:PYTHONDONTWRITEBYTECODE='1';$env:PYTHONIOENCODING='utf-8'
$env:TEMP=Join-Path $dataDir 'runtime_tmp';$env:TMP=$env:TEMP
$env:MPLCONFIGDIR=Join-Path $dataDir 'mpl_cache';$env:OPENBLAS_NUM_THREADS='1';$env:OMP_NUM_THREADS='2'
$pythonExe='C:\Users\A\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $pythonExe (Join-Path $PSScriptRoot $Entry) @Arguments
exit $LASTEXITCODE
