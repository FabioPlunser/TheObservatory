# Exit on error
$ErrorActionPreference = "Stop"

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
    Remove-Item -Path "db.db" -Force
    exit
}

# Register the cleanup function to run on script exit
Register-EngineEvent PowerShell.Exiting -Action { cleanup }

Write-Host "🚀 Starting setup script..."

# Create and activate Python virtual environment
$venv_path = "venvWin"
if (-Not (Test-Path -Path $venv_path)) {
    Write-Host "🐍 Creating Python virtual environment..."
    python -m venv $venv_path
}

Write-Host "🔄 Activating virtual environment..."
& "$venv_path\Scripts\Activate.ps1"

# Calculate checksum of requirements files
$requirements_files = @("server/requirements.txt", "devices/emulated/requirements.txt")
$checksum = Get-FileHash -Algorithm SHA256 -Path $requirements_files | ForEach-Object { $_.Hash } | Join-String -Separator ""
$checksum_file = "requirements_checksum.txt"

# Check if checksum has changed or if virtual environment does not exist
$run_pip_install = $false
if (-Not (Test-Path -Path $checksum_file)) {
    $run_pip_install = $true
} else {
    $stored_checksum = (Get-Content -Path $checksum_file -Raw).Trim()
    if ($checksum -ne $stored_checksum) {
        $run_pip_install = $true
    }
}

# Install Python requirements if needed
if ($run_pip_install) {
    Write-Host "📦 Installing required Python packages..."
    pip install -r server/requirements.txt
    pip install -r devices/emulated/requirements.txt
    $checksum | Set-Content -Path $checksum_file -NoNewline
} else {
    Write-Host "📦 Python packages are already up-to-date."
}

# Get number of devices to emulate
$num_cameras = Read-Host "Enter number of cameras to emulate"
$num_alarms = Read-Host "Enter number of alarms to emulate"
$use_simulated_data = Read-Host "Do you want to use simulated data? (y/n)"
$use_simulated_data = $use_simulated_data -eq "y"

# Start frontend setup
$spinner = Start-Job -ScriptBlock { animate_loading "Setting up frontend..." }

# Install frontend dependencies
if (command_exists bun) {
    Push-Location server/website
    bun install
    bun run build
    Pop-Location
    Stop-Job $spinner
    Write-Host "✅ Frontend setup complete!"
} elseif (command_exists npm) {
    Push-Location server/website
    npm install
    npm run build
    Pop-Location
    Stop-Job $spinner
    Write-Host "✅ Frontend setup complete!"
} else {
    Stop-Job $spinner
    Write-Host "❌ Neither bun nor npm found. Please install one of them."
    exit 1
}

# Start the server in the background
Write-Host "🖥️ Starting edge server..."
$server_process = Start-Process -FilePath "python" -ArgumentList "server/edge_server.py" -PassThru
$global:server_pid = $server_process.Id

# Wait for server to start
Write-Host "⏳ Waiting for server to initialize..."
Start-Sleep -Seconds 5

# Start emulated cameras
Write-Host "📸 Starting $num_cameras camera(s)..."
$global:camera_pids = @()
for ($i = 1; $i -le $num_cameras; $i++) {
    $camera_process = Start-Process -FilePath "python" -ArgumentList "devices/emulated/camera.py" -PassThru
    $global:camera_pids += $camera_process.Id
}

# Start simulated cameras
if ($use_simulated_data) {
    Write-Host "📸 Starting simulated cameras ..."
    $simulated_camera_process = Start-Process -FilePath "python" -ArgumentList "devices/emulated/simulatedCamera.py" -PassThru
    $global:simulated_camera_pid = $simulated_camera_process.Id
}

# Start emulated alarms
Write-Host "🚨 Starting $num_alarms alarm(s)..."
$global:alarm_pids = @()
for ($i = 1; $i -le $num_alarms; $i++) {
    $alarm_process = Start-Process -FilePath "python" -ArgumentList "devices/emulated/alarm.py" -PassThru
    $global:alarm_pids += $alarm_process.Id
}

# Keep script running
Write-Host "✅ System is running. Press Ctrl+C to stop all processes."
while ($true) {
    Start-Sleep -Seconds 1
}