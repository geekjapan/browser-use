param(
    [ValidateSet('verify', 'demo')][string]$Mode = 'verify',
    [ValidateSet('azure', 'openai')][string]$Provider = 'openai',
    [string]$Credentials = "$env:USERPROFILE\.config\decision-stack\.env"
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$root = Split-Path $PSScriptRoot -Parent
$candidates = @()
$scoop = "$env:USERPROFILE\scoop\apps\uv"
if (Test-Path $scoop) {
    $candidates += Get-ChildItem $scoop -Directory | Where-Object Name -ne 'current' |
        Sort-Object LastWriteTime -Descending | ForEach-Object { Join-Path $_.FullName 'uv.exe' }
}
$command = Get-Command uv.exe -ErrorAction SilentlyContinue
if ($command) { $candidates += $command.Source }
$uv = $null
foreach ($candidate in $candidates) {
    if (Test-Path $candidate) {
        try {
            & $candidate --version > $null 2>&1
            if ($LASTEXITCODE -eq 0) { $uv = $candidate; break }
        } catch { }
    }
}
if (!$uv) { throw 'Install uv before running this script.' }
$env:PATH = (Split-Path $uv -Parent) + ';' + $env:PATH
Push-Location $root
$browserProcess = $null
$serverProcess = $null
$profile = $null
try {
    if ($Mode -eq 'verify') {
        & $uv run --no-project --python 3.12 python verification/run.py
    } else {
        if (!(Test-Path $Credentials)) { throw "Create the private credentials file: $Credentials" }
        $chrome = @("$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
                    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe") |
            Where-Object { Test-Path $_ } | Select-Object -First 1
        if (!$chrome) { throw 'Install Google Chrome before starting the demo.' }
        $listener = New-Object System.Net.Sockets.TcpListener([Net.IPAddress]::Loopback, 0)
        $listener.Start()
        $port = $listener.LocalEndpoint.Port
        $listener.Stop()
        $profile = Join-Path ([IO.Path]::GetTempPath()) ('decision-browser-' + [guid]::NewGuid())
        New-Item -ItemType Directory -Path $profile | Out-Null
        $browserProcess = Start-Process $chrome -PassThru -ArgumentList @(
            '--headless=new',
            "--user-data-dir=`"$profile`"", "--remote-debugging-port=$port",
            '--remote-debugging-address=127.0.0.1', '--no-first-run', 'about:blank')
        $env:BU_NAME = 'decision-stack-' + $port
        $env:BU_CDP_URL = "http://127.0.0.1:$port"
        Write-Host 'Open http://127.0.0.1:8766 after the server starts.'
        $env:DECISION_PROVIDER = $Provider
        $credentialPath = $Credentials.Replace('\', '/')
        $serverProcess = Start-Process $uv -PassThru -NoNewWindow -WorkingDirectory $root `
            -RedirectStandardOutput (Join-Path $profile 'server.stdout.log') `
            -RedirectStandardError (Join-Path $profile 'server.stderr.log') -ArgumentList @(
            'run', '--env-file', "`"$credentialPath`"", 'python', '-m', 'jev_ultrafast.demo')
        while (!$serverProcess.HasExited) { Start-Sleep -Milliseconds 200 }
    }
    $code = if ($serverProcess) { $serverProcess.ExitCode } else { $LASTEXITCODE }
} finally {
    if ($serverProcess -and !$serverProcess.HasExited) {
        & taskkill.exe /PID $serverProcess.Id /T /F > $null 2>&1
    }
    if ($browserProcess) {
        try {
            & $uv run python -c "import os; from browser_harness.admin import daemon_alive, stop_remote_daemon; name=os.environ['BU_NAME']; daemon_alive(name) and stop_remote_daemon(name)" 2>$null
        } catch { }
    }
    if ($browserProcess -and !$browserProcess.HasExited) {
        $browserProcess.Kill()
        $null = $browserProcess.WaitForExit(5000)
    }
    if ($profile -and (Test-Path $profile)) { Remove-Item $profile -Recurse -Force -ErrorAction SilentlyContinue }
    Pop-Location
}
exit $code
