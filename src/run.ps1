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

# Prompt the user to launch or destroy the Terraform server
$action = Read-Host "Do you want to launch, destroy, or do nothing with the Terraform server? ([l]aunch/[d]estroy/[n]othing)"
if ($action -eq "l") {

    # Check if AWS credentials file exists
    $aws_credentials_path = "$env:USERPROFILE\.aws\credentials"
    $aws_credentials_exist = Test-Path -Path $aws_credentials_path

    if (-Not $aws_credentials_exist) {
        Write-Host "AWS credentials file not found. Prompting for credentials..."
        $aws_access_key = Read-Host "Enter your AWS Access Key"
        $aws_secret_key = Read-Host "Enter your AWS Secret Key"
        $aws_region = Read-Host "Enter your AWS Region (e.g., us-east-1)"

        # Configure AWS CLI with the provided credentials
        aws configure set aws_access_key_id $aws_access_key
        aws configure set aws_secret_access_key $aws_secret_key
        aws configure set region $aws_region
    } else {
        Write-Host "AWS credentials file found. Checking for credentials..."

        # Read the credentials file
        $credentials_content = Get-Content -Path $aws_credentials_path -Raw

        # Check if the credentials are present
        if ($credentials_content -notmatch "aws_access_key_id" -or $credentials_content -notmatch "aws_secret_access_key") {
            Write-Host "AWS credentials not found in the file. Prompting for credentials..."
            $aws_access_key = Read-Host "Enter your AWS Access Key"
            $aws_secret_key = Read-Host "Enter your AWS Secret Key"
            $aws_region = Read-Host "Enter your AWS Region (e.g., us-east-1)"

            # Configure AWS CLI with the provided credentials
            aws configure set aws_access_key_id $aws_access_key
            aws configure set aws_secret_access_key $aws_secret_key
            aws configure set region $aws_region
        } else {
            Write-Host "AWS credentials found in the file."
        }
    }

    # Check if the key pair exists
    $key_pair_name = "theObservatory"
    $key_pair_path = "$env:USERPROFILE\.ssh\theObservatory.pem"
    $ssh_dir = "$env:USERPROFILE\.ssh"

    if (-Not (Test-Path -Path $ssh_dir)) {
        Write-Host "Creating .ssh directory..."
        New-Item -ItemType Directory -Path $ssh_dir
    }

    if (-Not (Test-Path -Path $key_pair_path)) {
        Write-Host "Key pair file '$key_pair_path' not found. Checking if key pair exists in AWS..."
        if (aws ec2 describe-key-pairs --key-names $key_pair_name 2>&1 | Select-String -Pattern $key_pair_name) {
            Write-Host "Key pair '$key_pair_name' exists in AWS but the local file is missing."
            $create_new_key = Read-Host "The key pair exists in AWS but the local file is missing. Do you want to create a new key pair? (y/n)"
            if ($create_new_key -eq "y") {
                Write-Host "Deleting existing key pair in AWS and creating a new one..."
                aws ec2 delete-key-pair --key-name $key_pair_name
                aws ec2 create-key-pair --key-name $key_pair_name --query "KeyMaterial" --output text | Out-File -FilePath $key_pair_path -Encoding ascii
                icacls $key_pair_path /inheritance:r /grant:r ${env:USERNAME}:R # Windows equivalent of chmod 400
                Write-Host "New key pair created and saved to $key_pair_path"
            } else {
                Write-Host "Please ensure you have the correct key pair file at '$key_pair_path'."
                exit 1
            }
        } else {
            Write-Host "Key pair '$key_pair_name' not found in AWS. Creating key pair..."
            aws ec2 create-key-pair --key-name $key_pair_name --query "KeyMaterial" --output text | Out-File -FilePath $key_pair_path -Encoding ascii
            icacls $key_pair_path /inheritance:r /grant:r ${env:USERNAME}:R # Windows equivalent of chmod 400
            Write-Host "Key pair created and saved to $key_pair_path"
        }
    } else {
        Write-Host "Key pair file '$key_pair_path' found."
        icacls $key_pair_path /inheritance:r /grant:r ${env:USERNAME}:R # Ensure correct permissions
    }

    # Verify the key file permissions
    icacls $key_pair_path

    Write-Host "🌍 Initializing Terraform..."
    Push-Location "terraform"
    terraform init

    Write-Host "🚀 Applying Terraform configuration..."
    terraform apply -var "private_pem_key=$key_pair_path" -auto-approve
    Pop-Location

} elseif ($action -eq "d") {
    Write-Host "🛑 Destroying Terraform-managed infrastructure..."
    Push-Location "terraform"
    terraform destroy -var "private_pem_key=$key_pair_path" -auto-approve
    Pop-Location
    Write-Host "✅ Terraform-managed infrastructure destroyed."
    $close = Read-Host "Do you want to continue with the setup script? (y/n)"
    if ($close -eq "n") {
        exit
    }
}

elseif ($action -eq "n") {
    Write-Host "Continuing with the setup script..."
} else {
    Write-Host "🚫 Invalid action. Please enter 'launch' or 'destroy'."
    exit 1
}

# Create and activate Python virtual environment
$venv_path = "venvWin"
if (-Not (Test-Path -Path $venv_path)) {
    Write-Host "🐍 Creating Python virtual environment..."
    python -m venv $venv_path
}

Write-Host "🔄 Activating virtual environment..."
& "$venv_path\Scripts\Activate.ps1"

# Calculate checksum of requirements files
$requirements_files = @("onPremise/server/requirements.txt", "onPremise/devices/emulated/requirements.txt", "cloud/requirements.txt")
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
    pip install -r onPremise/server/requirements.txt
    pip install -r onPremise/devices/emulated/requirements.txt
    pip install -r cloud/requirements.txt
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
    Push-Location onPremise/server/website
    bun install
    bun run build
    Pop-Location
    Stop-Job $spinner
    Write-Host "✅ Frontend setup complete!"
} elseif (command_exists npm) {
    Push-Location onPremise/server/website
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
$server_process = Start-Process -FilePath "python" -ArgumentList "onPremise/server/main.py" -PassThru
$global:server_pid = $server_process.Id

# Wait for server to start
Write-Host "⏳ Waiting for server to initialize..."
Start-Sleep -Seconds 5

# Start emulated cameras
Write-Host "📸 Starting $num_cameras camera(s)..."
$global:camera_pids = @()
for ($i = 1; $i -le $num_cameras; $i++) {
    $camera_process = Start-Process -FilePath "python" -ArgumentList "onPremise/devices/emulated/camera.py" -PassThru
    $global:camera_pids += $camera_process.Id
}

# Start simulated cameras
if ($use_simulated_data) {
    Write-Host "📸 Starting simulated cameras ..."
    $simulated_camera_process = Start-Process -FilePath "python" -ArgumentList "onPremise/devices/emulated/simulatedCamera.py" -PassThru
    $global:simulated_camera_pid = $simulated_camera_process.Id
}

# Start emulated alarms
Write-Host "🚨 Starting $num_alarms alarm(s)..."
$global:alarm_pids = @()
for ($i = 1; $i -le $num_alarms; $i++) {
    $alarm_process = Start-Process -FilePath "python" -ArgumentList "onPremise/devices/emulated/alarm.py" -PassThru
    $global:alarm_pids += $alarm_process.Id
}

# Keep script running
Write-Host "✅ System is running. Press Ctrl+C to stop all processes."
while ($true) {
    Start-Sleep -Seconds 1
}