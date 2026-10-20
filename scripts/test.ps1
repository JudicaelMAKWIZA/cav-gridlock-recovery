param(
    [ValidateSet('daily', 'regression', 'scientific')]
    [string]$Mode = 'daily',
    [string]$Python = 'D:\cav-recovery-venvs\metadrive\Scripts\python.exe',
    [string]$Results = 'D:\cav-recovery-data\integrated-metadrive-20261020\checks'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONPYCACHEPREFIX = 'D:\cav-recovery-cache\pycache'
$env:TEMP = 'D:\cav-recovery-temp'
$env:TMP = $env:TEMP
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$selection = @('tests/unit', 'tests/communication')
if ($Mode -eq 'regression') { $selection += 'tests/integration' }
if ($Mode -eq 'scientific') { $selection = @('tests/scientific') }
$runName = $Mode + '-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
$runDirectory = Join-Path $Results $runName
New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
Push-Location $projectRoot
try {
    & $Python -X utf8 -m pytest -q -ra @selection `
        -o cache_dir=D:/cav-recovery-cache/pytest-active `
        "--basetemp=$runDirectory/temp" "--junitxml=$runDirectory/results.xml"
    $testExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $testExitCode
