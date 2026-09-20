param(
    [Parameter(Mandatory=$true)][string]$DocumentPath,
    [Parameter(Mandatory=$true)][string]$OutputDirectory
)

$ErrorActionPreference = 'Stop'
$source = (Resolve-Path -LiteralPath $DocumentPath).Path
$output = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $output | Out-Null
$temporaryPdf = Join-Path $output '_render_intermediate.pdf'
$prefix = Join-Path $output 'page'
$poppler = 'C:\Users\Chester\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe'

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $document = $word.Documents.Open($source, $false, $false)
    try {
        $document.Fields.Update() | Out-Null
        foreach ($toc in $document.TablesOfContents) { $toc.Update() }
        $document.Repaginate()
        $document.Save()
        $document.SaveAs2([string]$temporaryPdf, 17)
    } finally {
        $document.Close($false)
    }
} finally {
    $word.Quit()
}

& $poppler -r 144 -png $temporaryPdf $prefix
if ($LASTEXITCODE -ne 0) { throw "pdftoppm failed: $LASTEXITCODE" }
Remove-Item -LiteralPath $temporaryPdf -Force
$pages = @(Get-ChildItem -LiteralPath $output -Filter 'page-*.png').Count
Write-Output "$source -> $pages pages"
