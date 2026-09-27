$ErrorActionPreference = 'Stop'
$fresh = 'D:\SafeFAME-TS-SocialGood-clean-09'
$url = 'https://github.com/duzisezodebe47-maker/SafeFAME-TS/releases/download/socialgood-isolated-input-v9-20260927/SafeFAME-TS_SocialGood_isolated_v9_20260927.zip'
$expectedBytes = 3343370
$expectedSha = 'f75cbe2dcd6689f3d6a1b6800e366f4bc041d24f9efdf3d341b20baac3154e6f'
if (Test-Path -LiteralPath $fresh) { throw "Fresh replay directory already exists: $fresh" }
New-Item -ItemType Directory -Path $fresh | Out-Null
$zip = Join-Path $fresh 'downloaded_release.zip'
Invoke-WebRequest -Uri $url -OutFile $zip
$bytes = (Get-Item -LiteralPath $zip).Length
$actualSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $zip).Hash.ToLowerInvariant()
if ($bytes -ne $expectedBytes -or $actualSha -ne $expectedSha) {
    throw "Remote Release bytes/SHA mismatch: $bytes / $actualSha"
}
$package = Join-Path $fresh 'release'
Expand-Archive -LiteralPath $zip -DestinationPath $package
py -3.12 -c 'import sys,numpy,pandas; print(sys.version); print(numpy.__version__,pandas.__version__)' | Out-File -LiteralPath (Join-Path $PSScriptRoot 'environment.log') -Encoding utf8
$verifyCommand = "py -3.12 `"$package\code\verify_socialgood.py`" --package `"$package`" --out `"$fresh\audit`""
$negativeCommand = "py -3.12 `"$package\tests\run_negative_cases.py`" --package `"$package`" --out `"$fresh\negative_cases.json`""
& py -3.12 (Join-Path $package 'code\verify_socialgood.py') --package $package --out (Join-Path $fresh 'audit') 2>&1 |
    Out-File -LiteralPath (Join-Path $PSScriptRoot 'verify.log') -Encoding utf8
$verifyExit = $LASTEXITCODE
& py -3.12 (Join-Path $package 'tests\run_negative_cases.py') --package $package --out (Join-Path $fresh 'negative_cases.json') 2>&1 |
    Out-File -LiteralPath (Join-Path $PSScriptRoot 'negative.log') -Encoding utf8
$negativeExit = $LASTEXITCODE
if (Test-Path -LiteralPath (Join-Path $fresh 'negative_cases.json')) {
    Copy-Item -LiteralPath (Join-Path $fresh 'negative_cases.json') -Destination (Join-Path $PSScriptRoot 'negative_cases.json')
}
$result = [ordered]@{
    url = $url
    fresh_directory = $fresh
    downloaded_bytes = $bytes
    downloaded_sha256 = $actualSha
    expected_bytes = $expectedBytes
    expected_sha256 = $expectedSha
    verify_command = $verifyCommand
    verify_exit_code = $verifyExit
    negative_command = $negativeCommand
    negative_exit_code = $negativeExit
}
$result | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'replay_result.json') -Encoding utf8
if ($verifyExit -ne 0 -or $negativeExit -ne 0) { throw 'Remote replay failed; see logs' }
Write-Output 'REMOTE_REPLAY_PASS'
