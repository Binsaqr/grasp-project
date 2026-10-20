[CmdletBinding()]
param(
    [switch]$Restart,
    [switch]$SelfTestReadiness
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Image = "localhost/grasp-project:release"
$Container = "grasp-dashboard"
$DashboardUrl = "http://127.0.0.1:8080"
$RequiredUsbDevices = @(
    @{ Name = "Intel RealSense D405"; VidPid = "8086:0b5b" },
    @{ Name = "DAVIS 346"; VidPid = "152a:841a" }
)

function Test-DavisReady($Status) {
    try {
        if ($null -eq $Status -or $null -eq $Status.davis_motion) {
            return $false
        }
        $Stream = [string]$Status.davis_motion.stream
        return $Status.running.davis_motion -eq $true `
            -and $Stream.ToLowerInvariant() -eq "active" `
            -and $Status.davis_motion.report_heartbeat_current -eq $true
    } catch {
        return $false
    }
}

function Test-DashboardReady($Status) {
    try {
        if ($null -eq $Status -or $null -eq $Status.running -or $null -eq $Status.realsense) {
            return $false
        }
        $RealSenseReady = $Status.running.realsense -eq $true `
            -and $Status.realsense.realsense_rgb_active -eq $true `
            -and $Status.realsense.realsense_depth_active -eq $true `
            -and $Status.realsense.overlay_stale -ne $true
        return $RealSenseReady `
            -and (Test-DavisReady $Status) `
            -and $Status.operator_snapshot_stale -eq $false `
            -and $Status.realtime_status_age_unknown -ne $true
    } catch {
        return $false
    }
}

if ($SelfTestReadiness) {
    $QuietHealthy = [pscustomobject]@{
        running = [pscustomobject]@{ davis_motion = $true }
        davis_motion = [pscustomobject]@{
            stream = "active"
            report_heartbeat_current = $true
            event_rate = 0.0
        }
    }
    $Stopped = [pscustomobject]@{
        running = [pscustomobject]@{ davis_motion = $false }
        davis_motion = [pscustomobject]@{
            stream = "stopped"
            report_heartbeat_current = $false
            event_rate = 0.0
        }
    }
    $StaleHeartbeat = [pscustomobject]@{
        running = [pscustomobject]@{ davis_motion = $true }
        davis_motion = [pscustomobject]@{
            stream = "active"
            report_heartbeat_current = $false
            event_rate = 1000.0
        }
    }
    $QuietHealthy | Add-Member -NotePropertyName operator_snapshot_stale -NotePropertyValue $false
    $QuietHealthy | Add-Member -NotePropertyName realtime_status_age_unknown -NotePropertyValue $false
    $QuietHealthy | Add-Member -NotePropertyName realsense -NotePropertyValue ([pscustomobject]@{
        realsense_rgb_active = $true
        realsense_depth_active = $true
        overlay_stale = $false
    })
    $QuietHealthy.running | Add-Member -NotePropertyName realsense -NotePropertyValue $true
    $StaleOperatorSnapshot = $QuietHealthy | ConvertTo-Json -Depth 5 | ConvertFrom-Json
    $StaleOperatorSnapshot.operator_snapshot_stale = $true
    $IncompleteStatus = [pscustomobject]@{ running = [pscustomobject]@{} }
    if (-not (Test-DavisReady $QuietHealthy)) {
        throw "DAVIS readiness self-test failed: a healthy quiet stream was rejected."
    }
    if ((Test-DavisReady $Stopped) -or (Test-DavisReady $StaleHeartbeat)) {
        throw "DAVIS readiness self-test failed: an unavailable/stale stream was accepted."
    }
    if (
        -not (Test-DashboardReady $QuietHealthy) `
        -or (Test-DashboardReady $StaleOperatorSnapshot) `
        -or (Test-DashboardReady $IncompleteStatus) `
        -or (Test-DavisReady $IncompleteStatus)
    ) {
        throw "Dashboard readiness self-test failed: fresh/stale operator snapshot handling is incorrect."
    }
    Write-Host "Readiness self-test PASS: quiet live cameras and fresh operator state pass; stopped/stale inputs fail."
    exit 0
}

function Require-Command([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found. Install the documented Windows prerequisites first."
    }
}

function Get-ContainerId([switch]$RunningOnly) {
    $Arguments = @("ps", "-q", "--filter", "name=^/$Container`$")
    if (-not $RunningOnly) {
        $Arguments = @("ps", "-aq", "--filter", "name=^/$Container`$")
    }
    return (& docker @Arguments | Select-Object -First 1)
}

function Wait-Dashboard {
    $MaxAttempts = 240
    for ($Attempt = 1; $Attempt -le $MaxAttempts; $Attempt++) {
        $Status = $null
        try {
            $Status = Invoke-RestMethod -Uri "$DashboardUrl/api/status?compact=1" -TimeoutSec 2
            if (Test-DashboardReady $Status) {
                Write-Host "Dashboard and both live cameras ready: $DashboardUrl"
                return
            }
        } catch {}
        if (($Attempt % 10) -eq 0) {
            if ($null -eq $Status) {
                Write-Host "Waiting for dashboard readiness ($Attempt/$MaxAttempts s): status endpoint unavailable."
            } else {
                try {
                    Write-Host ("Waiting for dashboard readiness ({0}/{1} s): RS running={2} RGB={3} depth={4} overlay_stale={5}; DAVIS running={6} stream={7} heartbeat={8}; operator_stale={9}." -f `
                        $Attempt, $MaxAttempts, $Status.running.realsense, $Status.realsense.realsense_rgb_active, `
                        $Status.realsense.realsense_depth_active, $Status.realsense.overlay_stale, `
                        $Status.running.davis_motion, $Status.davis_motion.stream, `
                        $Status.davis_motion.report_heartbeat_current, $Status.operator_snapshot_stale)
                } catch {
                    Write-Host "Waiting for dashboard readiness ($Attempt/$MaxAttempts s): status payload is incomplete."
                }
            }
        }
        if ($Attempt -lt $MaxAttempts) {
            Start-Sleep -Seconds 1
        }
    }

    & docker logs --tail 100 $Container
    throw "The dashboard did not become ready within $MaxAttempts seconds."
}

function Invoke-UsbIpd([string[]]$Arguments, [switch]$CaptureOutput) {
    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "SilentlyContinue"
        $Output = & usbipd @Arguments 2>&1
        $ExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($ExitCode -ne 0) {
        throw "usbipd $($Arguments -join ' ') failed with exit code $ExitCode."
    }
    if ($CaptureOutput) {
        return ($Output | Out-String)
    }
}

Require-Command docker
Require-Command usbipd

& docker info *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Docker Desktop is installed but its Linux engine is not running. Start Docker Desktop and rerun this script."
}

$UsbList = Invoke-UsbIpd -Arguments @("list") -CaptureOutput
foreach ($Device in $RequiredUsbDevices) {
    $Line = ($UsbList -split "`r?`n" | Where-Object { $_ -match [regex]::Escape($Device.VidPid) } | Select-Object -First 1)
    if (-not $Line) {
        throw "$($Device.Name) ($($Device.VidPid)) is not connected to Windows."
    }
    if ($Line -notmatch '^\s*(\d+-\d+)\s+') {
        throw "Could not read the USB bus ID for $($Device.Name): $Line"
    }

    $BusId = $Matches[1]
    if ($Line -match '\bNot shared\b') {
        throw "$($Device.Name) is not shared with WSL. In an Administrator PowerShell run: usbipd bind --busid $BusId"
    }
    if ($Line -notmatch '\bAttached\b') {
        Write-Host "Attaching $($Device.Name) on USB bus $BusId to WSL..."
        Invoke-UsbIpd -Arguments @("attach", "--wsl", "--busid", $BusId)
    } else {
        Write-Host "$($Device.Name) is already attached to WSL."
    }
}

Write-Host "Loading the RealSense USB video driver in Docker Desktop's Linux VM..."
& docker run --rm --privileged --mount "type=bind,source=/lib/modules,target=/lib/modules,readonly" alpine:3.22 modprobe uvcvideo
if ($LASTEXITCODE -ne 0) {
    throw "Could not load the uvcvideo driver in Docker Desktop's Linux VM."
}

$RunningId = Get-ContainerId -RunningOnly
if ($RunningId -and -not $Restart) {
    Write-Host "The $Container container is already running. Use -Restart to rebuild and replace it."
    Wait-Dashboard
    exit 0
}

if ($RunningId -and $Restart) {
    Write-Host "Stopping the existing $Container container..."
    & docker rm -f $Container *> $null
}

$ExistingId = Get-ContainerId
if ($ExistingId) {
    & docker rm $Container *> $null
}

Push-Location $ProjectDir
try {
    New-Item -ItemType Directory -Force -Path (Join-Path $ProjectDir "output") *> $null

    Write-Host "Building the release image (cached layers are reused)..."
    & docker build -t $Image .
    if ($LASTEXITCODE -ne 0) {
        throw "Docker image build failed."
    }

    $OutputDir = Join-Path $ProjectDir "output"
    $ConfigDir = Join-Path $ProjectDir "config"
    Write-Host "Starting the dashboard and both camera backends..."
    & docker run -d --name $Container --restart unless-stopped --privileged --ipc host -p "127.0.0.1:8080:8080" `
        -e "ROS_DOMAIN_ID=67" -e "HOST=0.0.0.0" -e "PORT=8080" `
        -e "REALSENSE_COLOR_PROFILE=480x270x30" -e "REALSENSE_DEPTH_PROFILE=480x270x30" `
        --mount "type=bind,source=$OutputDir,target=/app/output" `
        --mount "type=bind,source=$ConfigDir,target=/app/config" `
        $Image
    if ($LASTEXITCODE -ne 0) {
        throw "Docker container start failed."
    }
} finally {
    Pop-Location
}

Wait-Dashboard
& docker exec $Container bash -lc "cd /app && ./scripts/check_demo_system.sh"
if ($LASTEXITCODE -ne 0) {
    throw "The container started, but its system health check failed."
}

Write-Host "Ready. Keep this container running and open $DashboardUrl"
