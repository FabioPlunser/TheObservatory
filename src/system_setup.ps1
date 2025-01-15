function Setup-System {
    Setup-Python-Environment
    Install-Dependencies
    Setup-Frontend
    Start-Server

    Configure-NatsUrl

    Start-Devices
}
function Setup-Python-Environment {
    # Create and activate Python virtual environment
    $venv_path = "venvWin"
    if (-Not (Test-Path -Path $venv_path)) {
        Write-Host "🐍 Creating Python virtual environment..."
        python -m venv $venv_path
    }

    Write-Host "🔄 Activating virtual environment..."
    & "$venv_path\Scripts\Activate.ps1"

    # Install requirements based on checksum
    $requirements_files = @("onPremise/server/requirements.txt", "onPremise/devices/emulated/requirements.txt", "cloud/requirements.txt")
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
        pip install -r onPremise/server/requirements.txt
        pip install -r onPremise/devices/emulated/requirements.txt
        pip install -r cloud/requirements.txt
        $checksum | Set-Content -Path $checksum_file -NoNewline
    }
}

function Install-Dependencies {
    # Download YOLOv8 model
    Write-Host "📦 Downloading YOLOv8 model..."
    if (-Not (Test-Path -Path "yolov8n.pt")) {
        Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt" -OutFile "yolov8n.pt"
    }

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

    # Install torchreid
    Write-Host "📦 Installing torchreid..."
    if (-Not (Test-Path -Path "torchreid")) {
        git clone https://github.com/KaiyangZhou/deep-person-reid.git torchreid
        Push-Location torchreid
        pip install -e .
        Pop-Location
    }
}

function Setup-Frontend {
    $spinner = Start-Job -ScriptBlock { animate_loading "Setting up frontend..." }

    if (command_exists bun) {
        Push-Location onPremise/server/website
        bun install
        bun run build
        Pop-Location
        Stop-Job $spinner
        Write-Host "✅ Frontend setup complete!"
    }
    elseif (command_exists npm) {
        Push-Location onPremise/server/website
        npm install
        npm run build
        Pop-Location
        Stop-Job $spinner
        Write-Host "✅ Frontend setup complete!"
    }
    else {
        Stop-Job $spinner
        Write-Host "❌ Neither bun nor npm found. Please install one of them."
        exit 1
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

    Write-Host "⏳ Waiting for server to initialize..."
    $maxWaitTime = 60 
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
    # Wait longer to ensure server is fully ready
    Start-Sleep -Seconds 10

    # Try to get NATS URL from Terraform
    Push-Location "terraform"
    try {
        $nats_ip = terraform output -raw nats_instance_public_ip 2>$null
        if ($nats_ip) {
            Write-Host "🔄 Configuring edge server with NATS URL..."
            $maxRetries = 10
            $retryCount = 0
            $success = $false

            while (-not $success -and $retryCount -lt $maxRetries) {
                try {
                    $natsUrl = [uri]::EscapeDataString("nats://${nats_ip}:4222")
                    $uri = "http://localhost:8000/api/update-cloud-url?cloud_url=$natsUrl"
                    
                    Write-Host "Making request to: $uri"
                    $response = Invoke-RestMethod -Uri $uri -Method Post

                    if ($response.status -eq "success") {
                        Write-Host "✅ Edge server configured with NATS URL"
                        $success = $true
                        
                        # Verify configuration
                        Start-Sleep -Seconds 2
                        $status = Invoke-RestMethod -Uri "http://localhost:8000/api/get-company" -Method Get
                        if ($status.company.cloud_url -eq "nats://${nats_ip}:4222") {
                            Write-Host "✅ NATS connection verified"
                            break
                        }
                    }
                    
                    Write-Host "⚠️ Retry $($retryCount + 1) of $maxRetries..."
                    $retryCount++
                    Start-Sleep -Seconds 5
                } catch {
                    Write-Host "⚠️ Error during configuration: $_"
                    Write-Host "Response: $($_.ErrorDetails.Message)"
                    $retryCount++
                    Start-Sleep -Seconds 5
                }
            }

            if (-not $success) {
                Write-Host "`n⚠️ Failed to configure NATS after $maxRetries attempts."
                Write-Host "Manual configuration steps:"
                Write-Host "1. Open http://localhost:8000 in your browser"
                Write-Host "2. Click on 'Settings'"
                Write-Host "3. Enter this NATS URL: nats://${nats_ip}:4222"
                Write-Host "4. Click 'Save'"
            }
        }
    } catch {
        Write-Host "ℹ️ No NATS configuration needed"
    } finally {
        Pop-Location
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