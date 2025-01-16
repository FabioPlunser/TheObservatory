function Setup-System {
    # Start RTSP server first
    Start-RTSPServer

    # Then setup Python and start server
    Setup-Python-Environment
    Install-Dependencies
    Setup-Frontend
    Start-Server

    # Configure NATS after server is running
    if ($Global:nats_url) {
        Start-Sleep -Seconds 5  # Give the server a moment to fully initialize
        Configure-NatsUrl -nats_url $Global:nats_url
    }

    # Finally start devices
    Start-Devices
}

function Setup-Python-Environment {
    # Get the root directory (one level up from script location)
    $rootDir = Split-Path -Parent (Split-Path -Parent $scriptPath)

    # Create and activate Python virtual environment
    $venv_path = "venvWin"
    if (-Not (Test-Path -Path $venv_path)) {
        Write-Host "🐍 Creating Python virtual environment..."
        python -m venv $venv_path
    }

    Write-Host "🔄 Activating virtual environment..."
    & "$venv_path\Scripts\Activate.ps1"

    # Install requirements based on checksum using correct paths
    $requirements_files = @(
        (Join-Path -Path $rootDir -ChildPath "TheObservatory\src\onPremise\server\requirements.txt"),
        (Join-Path -Path $rootDir -ChildPath "TheObservatory\src\onPremise\devices\emulated\requirements.txt"),
        (Join-Path -Path $rootDir -ChildPath "TheObservatory\src\cloud\requirements.txt")
    )

    Write-Host "Looking for requirements files in:"
    foreach ($file in $requirements_files) {
        Write-Host "- $file"
        if (-Not (Test-Path -Path $file)) {
            Write-Host "❌ Cannot find requirements file: $file"
            exit 1
        }
    }

    # Calculate checksum and install if needed
    $checksum = Get-FileHash -Algorithm SHA256 -Path $requirements_files | ForEach-Object { $_.Hash } | Join-String -Separator ""
    $checksum_file = "requirements_checksum.txt"

    $run_pip_install = $false
    if (-Not (Test-Path -Path $checksum_file)) {
        $run_pip_install = $true
    }
    else {
        $stored_checksum = (Get-Content -Path $checksum_file -Raw).Trim()
        if ($checksum -ne $stored_checksum) {
            $run_pip_install = $true
        }
    }

    if ($run_pip_install) {
        Write-Host "📦 Installing required Python packages..."
        foreach ($req_file in $requirements_files) {
            Write-Host "Installing requirements from: $req_file"
            pip install -r $req_file
        }
        $checksum | Set-Content -Path $checksum_file -NoNewline
    }
}

function Install-Dependencies {
    # Install system dependencies
    Write-Host "📦 Installing system dependencies..."
    if (-Not (Test-Path -Path "C:\Program Files\CMake\bin\cmake.exe")) {
        Write-Host "CMake not found. Installation required."
        if (Get-Command choco -ErrorAction SilentlyContinue) {
            if (-Not (Test-Administrator)) {
                Write-Host "⚠️ Administrative privileges required for installing dependencies."
                $install = Read-Host "Do you want to restart the script with admin rights to install dependencies? (y/n)"
                if ($install -eq "y") {
                    Restart-ScriptAsAdmin
                }
            }
            else {
                choco install -y cmake vcredist140 ffmpeg
            }
        }
        else {
            Write-Host "⚠️ Dependencies missing. Please install manually:"
            Write-Host "1. CMake: https://cmake.org/download/"
            Write-Host "2. Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe"
            Write-Host "3. FFmpeg: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full.7z"
            $continue = Read-Host "Have you installed the required dependencies? (y/n)"
            if ($continue -ne "y") {
                exit 1
            }
        }
    }

    # Install PyTorch
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        Write-Host "✅ NVIDIA CUDA detected"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    }
    else {
        Write-Host "⚠️ NVIDIA CUDA not found, using CPU for processing"
        pip install torch torchvision torchaudio
    }
}

function Setup-Frontend {
    # Get the root directory path correctly
    $rootDir = Split-Path -Parent (Split-Path -Parent $scriptPath)
    $frontendPath = Join-Path -Path $rootDir -ChildPath "src\onPremise\server\website"

    Write-Host "Current script path: $scriptPath"
    Write-Host "Root directory: $rootDir"
    Write-Host "Looking for frontend at: $frontendPath"

    if (-Not (Test-Path -Path $frontendPath)) {
        # Try alternative path resolution
        $altPath = Join-Path -Path (Get-Location) -ChildPath "onPremise\server\website"
        Write-Host "Frontend not found, trying alternative path: $altPath"
        
        if (Test-Path -Path $altPath) {
            $frontendPath = $altPath
            Write-Host "✅ Found frontend at alternative path"
        }
        else {
            Write-Host "❌ Frontend directory not found at either:"
            Write-Host "   1. $frontendPath"
            Write-Host "   2. $altPath"
            Write-Host "Please ensure the frontend directory exists in one of these locations."
            exit 1
        }
    }

    Write-Host "Setting up frontend at: $frontendPath"
    $spinner = Start-Job -ScriptBlock { animate_loading "Setting up frontend..." }

    try {
        Push-Location $frontendPath
        
        if (command_exists bun) {
            bun install
            bun run build
            Write-Host "✅ Frontend setup complete using bun!"
        }
        elseif (command_exists npm) {
            npm install
            npm run build
            Write-Host "✅ Frontend setup complete using npm!"
        }
        else {
            Write-Host "❌ Neither bun nor npm found. Please install one of them."
            exit 1
        }
    }
    finally {
        Pop-Location
        if ($spinner) {
            Stop-Job $spinner
            Remove-Job $spinner
        }
    }
}

function Start-Server {
    # Check if port 8000 is in use
    Write-Host "🔍 Checking if port 8000 is available..."
    $portInUse = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
    if ($portInUse) {
        Write-Host "❌ Port 8000 is already in use. Attempting to free it..."
        foreach ($connection in $portInUse) {
            try {
                Stop-Process -Id $connection.OwningProcess -Force -ErrorAction SilentlyContinue
                Write-Host "✅ Stopped process using port 8000"
            }
            catch {
                Write-Host "⚠️ Failed to stop process. Please manually stop any process using port 8000"
                Write-Host "   You can use 'netstat -ano | findstr :8000' to find the process"
                cleanup
                exit 1
            }
        }
        # Wait for port to be freed
        Start-Sleep -Seconds 2
    }

    # Delete existing database and log files
    Write-Host "🗑️ Cleaning up old database..."
    Remove-Item -Path "db.db" -Force -ErrorAction SilentlyContinue
    Remove-Item -Path "log.log" -Force -ErrorAction SilentlyContinue
    Remove-Item -Path "server_error.log" -Force -ErrorAction SilentlyContinue
    Remove-Item -Path "server_output.log" -Force -ErrorAction SilentlyContinue

    Write-Host "🖥️ Starting edge server..."
    $env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512"
    
    # Start server with both stdout and stderr redirection
    $server_process = Start-Process -FilePath "python" -ArgumentList "onPremise/server/main.py" -PassThru `
        -RedirectStandardOutput "server_output.log" -RedirectStandardError "server_error.log"
    $global:server_pid = $server_process.Id

    Write-Host "⏳ Checking if the server is already up..."
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/api/get-company" -Method Get -TimeoutSec 2
        if ($response) {
            Write-Host "✅ Server is already responding. Skipping wait."
            return
        }
    } catch {
        Write-Host "Server not ready yet. Starting wait timer..."
    }

    Write-Host "⏳ Waiting for server to initialize..."
    $maxWaitTime = 10
    $waited = 0
    $serverStarted = $false

    while ($waited -lt $maxWaitTime -and -not $serverStarted) {
        Start-Sleep -Seconds 1
        $waited++
        Write-Host -NoNewline "`rWaiting for server... $waited/$maxWaitTime seconds"

        # Check if server process is still running
        if ($server_process.HasExited) {
            $error_content = Get-Content "server_error.log" -Raw
            Write-Host "`n❌ Server process exited prematurely. Error log:"
            Write-Host $error_content
            cleanup
            exit 1
        }

        # Check for successful startup in output log
        if (Test-Path -Path "server_output.log") {
            $output_content = Get-Content "server_output.log" -Raw
            if ($output_content -match "Application startup complete" -and 
                $output_content -match "Successfully registered mDNS service") {
                $serverStarted = $true
                break
            }
        }

        # Check for port conflict error
        if (Test-Path -Path "server_error.log") {
            $error_content = Get-Content "server_error.log" -Raw
            if ($error_content -match "error while attempting to bind on address") {
                Write-Host "`n❌ Port 8000 is still in use. Please free the port and try again."
                cleanup
                exit 1
            }
        }

        # Add periodic status update from the logs
        if ($waited % 10 -eq 0) {
            Write-Host "`nCurrent server status:"
            if (Test-Path -Path "server_output.log") {
                Get-Content "server_output.log" -Tail 5 | ForEach-Object { Write-Host "  $_" }
            }
        }
    }

    # Verify server is responding
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/api/get-company" -Method Get -TimeoutSec 5
        Write-Host "`n✅ Server is responding to requests"
    }
    catch {
        Write-Host "`n❌ Server is not responding to requests. Check server_error.log for details."
        Get-Content "server_error.log" | Write-Host
        cleanup
        exit 1
    }

    Write-Host "`n✅ Server started successfully!"
    Write-Host "ℹ️ Note: NATS warnings are normal when no cloud URL is configured."
}

function Configure-NatsUrl {
    param (
        [string]$nats_url
    )

    if (-not $nats_url) {
        Write-Host "ℹ️ No NATS URL to configure"
        return
    }

    Write-Host "🔄 Configuring edge server with NATS URL: $nats_url"
    
    $maxRetries = 5
    $retryCount = 0
    $success = $false

    # Try to verify server is actually running first
    try {
        $null = Invoke-RestMethod -Uri "http://localhost:8000/api/get-company" -Method Get -TimeoutSec 5
    }
    catch {
        Write-Host "❌ Edge server is not running. Cannot configure NATS URL."
        Write-Host "Please wait for the server to start or check server_error.log for issues."
        return
    }

    while (-not $success -and $retryCount -lt $maxRetries) {
        try {
            # Properly encode the URL for query parameter
            $encodedUrl = [System.Web.HttpUtility]::UrlEncode($nats_url)
            $uri = "http://localhost:8000/api/update-cloud-url?cloud_url=$encodedUrl"
            
            Write-Host "Sending request to: $uri"
            $response = Invoke-RestMethod -Uri $uri -Method Post

            if ($response.status -eq "success") {
                Write-Host "✅ Edge server configured with NATS URL"
                $success = $true
                break
            }
            
            Write-Host "⚠️ Retry $($retryCount + 1) of $maxRetries..."
            $retryCount++
            Start-Sleep -Seconds 2
        }
        catch {
            Write-Host "⚠️ Error configuring NATS URL: $_"
            Write-Host "Response: $($_.ErrorDetails.Message)"
            $retryCount++
            Start-Sleep -Seconds 2
        }
    }

    if (-not $success) {
        Write-Host "⚠️ Failed to configure NATS URL automatically"
        Write-Host "Manual configuration steps:"
        Write-Host "1. Wait for the server to fully start"
        Write-Host "2. Open http://localhost:8000 in your browser"
        Write-Host "3. Click on 'Settings'"
        Write-Host "4. Enter this NATS URL: $nats_url"
        Write-Host "5. Click 'Save'"
    }
}

function Start-Devices {
    # Get number of devices to emulate
    $num_cameras = Read-Host "Enter number of cameras to emulate"
    $num_alarms = Read-Host "Enter number of alarms to emulate"
    $use_simulated_data = Read-Host "Do you want to use simulated data? (y/n)"
    $global:use_simulated_data = $use_simulated_data -eq "y"

    Write-Host "📸 Starting $num_cameras camera(s)..."
    $global:camera_pids = @()
    for ($i = 1; $i -le $num_cameras; $i++) {
        $camera_process = Start-Process -FilePath "python" -ArgumentList "onPremise/devices/emulated/camera.py" -PassThru
        $global:camera_pids += $camera_process.Id
    }

    if ($global:use_simulated_data) {
        Write-Host "📸 Starting simulated cameras ..."
        $simulated_camera_process = Start-Process -FilePath "python" -ArgumentList "onPremise/devices/emulated/simulatedCamera.py" -PassThru
        $global:simulated_camera_pid = $simulated_camera_process.Id
    }

    Write-Host "🚨 Starting $num_alarms alarm(s)..."
    $global:alarm_pids = @()
    for ($i = 1; $i -le $num_alarms; $i++) {
        $alarm_process = Start-Process -FilePath "python" -ArgumentList "onPremise/devices/emulated/alarm.py" -PassThru
        $global:alarm_pids += $alarm_process.Id
    }
}

function Test-DockerRunning {
    try {
        $null = docker info 2>&1
        return $true
    }
    catch {
        Write-Host "❌ Docker is not running. Please start Docker Desktop and try again."
        return $false
    }
}

function Start-RTSPServer {
    Write-Host "🎥 Starting RTSP server..."
    
    if (-not (Test-DockerRunning)) {
        exit 1
    }

    # Check if container already exists
    $containerExists = docker ps -a --filter "name=rtsp" --format "{{.Names}}" | Select-String -Pattern "^rtsp$"
    if ($containerExists) {
        # Remove existing container
        docker rm -f rtsp
    }

    try {
        docker run -d --name rtsp -p 8554:8554 aler9/rtsp-simple-server
        Write-Host "✅ RTSP server started successfully"
    }
    catch {
        Write-Host "❌ Failed to start RTSP server: $_"
        exit 1
    }
}