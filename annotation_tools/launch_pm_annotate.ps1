# Launch the PrecisionMyotube annotator in the pm-annotate env.
# Run this in your OWN terminal (VS Code / Cursor integrated terminal or
# PowerShell) so it can open a napari window on your desktop. It cannot be
# launched from the assistant's sandboxed session (no OpenGL desktop context).
#
#   powershell -ExecutionPolicy Bypass -File annotation_tools\launch_pm_annotate.ps1
#   # force software OpenGL if the GPU canvas crashes:
#   powershell -ExecutionPolicy Bypass -File annotation_tools\launch_pm_annotate.ps1 -SoftwareGL
#
# All output (including a faulthandler C-stack on a hard crash) is written to
# annotation_tools\_launch.out.log and _launch.err.log for diagnosis.

param(
    [string]$Package = "PrecisionMyotube\annotation_work\32_C08_smoke",
    [switch]$SoftwareGL
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot            # repo root
$Py   = "C:\Users\liqig\anaconda3\envs\pm-annotate\python.exe"
$OutLog = Join-Path $PSScriptRoot "_launch.out.log"
$ErrLog = Join-Path $PSScriptRoot "_launch.err.log"

$env:PYTHONPATH = "$Repo\PrecisionMyotube;$Repo\annotation_tools"
$env:PYTHONFAULTHANDLER = "1"

if ($SoftwareGL) {
    $env:QT_OPENGL = "software"
    $env:LIBGL_ALWAYS_SOFTWARE = "1"
    Write-Host "Software OpenGL forced (QT_OPENGL=software)." -ForegroundColor Yellow
}

Write-Host "Launching annotator on $Package ..." -ForegroundColor Cyan
Write-Host "Logs: $OutLog  /  $ErrLog" -ForegroundColor DarkGray

# Start-Process redirects at the OS level, so the crash stack is captured to the
# log file even if the process dies hard. -Wait blocks until napari is closed.
$args = @('-X', 'faulthandler', '-m', 'annotation_tools', 'launch', '--package', $Package)
$p = Start-Process -FilePath $Py -ArgumentList $args -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog

Write-Host "Process exited with code $($p.ExitCode)." -ForegroundColor Cyan
if ($p.ExitCode -ne 0) {
    Write-Host "--- last lines of error log ---" -ForegroundColor Yellow
    Get-Content $ErrLog -Tail 40
}
