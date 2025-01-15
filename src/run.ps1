# Exit on error
$ErrorActionPreference = "Stop"

# Source the subscripts
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
. "$scriptPath\terraform_setup.ps1"
. "$scriptPath\system_setup.ps1"

# Initialize global variables
$global:camera_pids = @()
$global:alarm_pids = @()
$global:server_pid = $null
$global:simulated_camera_pid = $null
$global:use_simulated_data = $false

# Function to check if a command exists
function command_exists {
    param (
        [string]$command
    )
    return Get-Command $command -ErrorAction SilentlyContinue
}

# Function to animate loading
function animate_loading {
    param (
        [string]$message
    )
    $frames = @('⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏')
    $delay = 100
    $i = 0
    while ($true) {
        Write-Host -NoNewline "`r$($frames[$i++ % $frames.Length]) $message"
        Start-Sleep -Milliseconds $delay
    }
}

# Function to clean up processes
function cleanup {
    Write-Host "🛑 Stopping all processes..."
    # Kill camera processes
    foreach ($pid in $camera_pids) {
        Stop-Process -Id $pid -ErrorAction SilentlyContinue
    }
    # Kill alarm processes
    foreach ($pid in $alarm_pids) {
        Stop-Process -Id $pid -ErrorAction SilentlyContinue
    }
    # Kill simulated camera process
    if ($use_simulated_data) {
        Stop-Process -Id $simulated_camera_pid -ErrorAction SilentlyContinue
    }
    # Kill server
    Stop-Process -Id $server_pid -ErrorAction SilentlyContinue
    
    Write-Host "👋 Cleanup complete"
    # Only try to remove db.db if it exists
    if (Test-Path -Path "db.db") {
        Remove-Item -Path "db.db" -Force -ErrorAction SilentlyContinue
    }
    exit
}

# Register the cleanup function to run on script exit
Register-EngineEvent PowerShell.Exiting -Action { cleanup }

Write-Host "🚀 Starting setup script..."

# Function to check if running as administrator
function Test-Administrator {
    $user = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($user)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Function to restart script as administrator
function Restart-ScriptAsAdmin {
    if (-Not (Test-Administrator)) {
        Write-Host "🔒 Requesting administrative privileges..."
        Start-Process powershell -Verb RunAs -ArgumentList "-File `"$PSCommandPath`" $args"
        exit
    }
}

# Handle Terraform operations
$action = Read-Host "Do you want to launch, destroy, or do nothing with the Terraform server? ([l]aunch/[d]estroy/[n]othing)"
Handle-Terraform $action

# Run system setup
Setup-System

# Keep script running
Write-Host "✅ System is running. Press Ctrl+C to stop all processes."
while ($true) {
    Start-Sleep -Seconds 1
}