# Windows entry point: forwards to ./dev inside WSL, where the toolchain lives.
#
#   .\dev.ps1            incremental build
#   .\dev.ps1 doctor     environment report
#   .\dev.ps1 test -q    arguments are forwarded through

# 'Continue', not 'Stop': ./dev writes its own diagnostics to stderr, and Stop
# would wrap them in a PowerShell error record instead of showing the message.
# The exit code is propagated explicitly below.
$ErrorActionPreference = 'Continue'

$winPath = $PSScriptRoot.Replace('\', '/')
$wslPath = (wsl wslpath -a "$winPath").Trim()

$fwd = ($args | ForEach-Object { "'" + ($_.ToString().Replace("'", "'\''")) + "'" }) -join ' '

wsl -e bash -lc "cd '$wslPath' && ./dev $fwd"
exit $LASTEXITCODE
