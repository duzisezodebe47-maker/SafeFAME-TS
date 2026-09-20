$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$config = Get-Content -LiteralPath (Join-Path $projectRoot 'submission.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    foreach ($report in $config.reports) {
        Write-Output ('Opening ' + $report.role)
        $source = Join-Path $projectRoot $report.docx
        $document = $word.Documents.Open($source, $false, $false)
        try {
            Write-Output ('Updating fields ' + $report.role)
            $document.Fields.Update() | Out-Null
            foreach ($toc in $document.TablesOfContents) { $toc.Update() }
            $document.Repaginate()
            Write-Output ('Saving ' + $report.role)
            $document.Save()
            Write-Output ('Exporting PDF ' + $report.role)
            $temporaryPdf = Join-Path $projectRoot ('tmp\revision_20260919\' + $report.role + '_boundary.pdf')
            $document.SaveAs2([string]$temporaryPdf, 17)
            Copy-Item -LiteralPath $temporaryPdf -Destination (Join-Path $projectRoot $report.pdf) -Force
        } finally { $document.Close($false) }
        Copy-Item -LiteralPath $source -Destination (Join-Path $projectRoot $report.mirror) -Force
        Copy-Item -LiteralPath (Join-Path $projectRoot $report.pdf) -Destination (Join-Path $projectRoot $report.mirror_pdf) -Force
        Write-Output ('Exported ' + $report.role)
    }
} finally { $word.Quit() }
