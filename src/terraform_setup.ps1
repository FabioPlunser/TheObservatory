# Script-level variables
$script:key_pair_name = "theObservatory"
$script:key_pair_path = "$env:USERPROFILE\.ssh\theObservatory.pem"
$script:ssh_dir = "$env:USERPROFILE\.ssh"

function Handle-Terraform {
    param (
        [string]$action
    )

    # Store the starting location
    $startLocation = Get-Location

    try {
        if ($action -eq "l") {
            Setup-SSH-Keys
            Apply-Terraform
        }
        elseif ($action -eq "d") {
            Destroy-Terraform
            $close = Read-Host "Do you want to continue with the setup script? (y/n)"
            if ($close -eq "n") {
                exit
            }
        }
        elseif ($action -eq "n") {
            Write-Host "Continuing with the setup script..."
            # Store NATS IP for later use if it exists
            $nats_ip = Get-NatsIp
            if ($nats_ip) {
                $Global:nats_url = "nats://${nats_ip}:4222"
                Write-Host "✅ Found existing NATS URL: $Global:nats_url"
                Write-Host "The edge server will be configured with this URL once it's running."
            }
        }
        else {
            Write-Host "🚫 Invalid action. Please enter 'launch' or 'destroy'."
            exit 1
        }
    }
    finally {
        # Always return to starting location
        Set-Location $startLocation
    }
}

function Test-AWSCredentials {
    try {
        # First test basic AWS CLI access
        $null = aws sts get-caller-identity 2>&1
        
        # Then specifically check for token expiration using terraform
        Push-Location "terraform"
        $result = terraform providers 2>&1
        Pop-Location
        
        if ($result -match "ExpiredToken") {
            Write-Host "❌ Token has expired"
            return $false
        }
        
        return $true
    }
    catch {
        Write-Host "❌ AWS CLI access failed"
        return $false
    }
}

function Setup-AWS-Credentials {
    $aws_credentials_path = "$env:USERPROFILE\.aws\credentials"
    
    if (-Not (Test-Path -Path $aws_credentials_path)) {
        Write-Host "❌ AWS credentials file not found at: $aws_credentials_path"
        Write-Host "Please configure your AWS credentials and try again."
        exit 1
    }
    
    if (-Not (Test-AWSCredentials)) {
        Write-Host "❌ AWS credentials are expired or invalid."
        Write-Host "Please update your AWS credentials at: $aws_credentials_path"
        Write-Host "You can use 'aws configure' to set up new credentials."
        Write-Host "Note: If you're using temporary credentials or an AWS token, make sure it hasn't expired."
        exit 1
    }
    
    Write-Host "✅ AWS credentials are valid and not expired"
}

function Setup-SSH-Keys {
    if (-Not (Test-Path -Path $script:ssh_dir)) {
        Write-Host "Creating .ssh directory..."
        New-Item -ItemType Directory -Path $script:ssh_dir
    }

    if (-Not (Test-Path -Path $script:key_pair_path)) {
        Write-Host "Key pair file '$script:key_pair_path' not found. Checking if key pair exists in AWS..."
        if (aws ec2 describe-key-pairs --key-names $script:key_pair_name 2>&1 | Select-String -Pattern $script:key_pair_name) {
            Write-Host "Key pair '$script:key_pair_name' exists in AWS but the local file is missing."
            $create_new_key = Read-Host "Do you want to create a new key pair? (y/n)"
            if ($create_new_key -eq "y") {
                aws ec2 delete-key-pair --key-name $script:key_pair_name
                aws ec2 create-key-pair --key-name $script:key_pair_name --query "KeyMaterial" --output text | Out-File -FilePath $script:key_pair_path -Encoding ascii
                icacls $script:key_pair_path /inheritance:r /grant:r ${env:USERNAME}:R
            }
            else {
                Write-Host "Please ensure you have the correct key pair file at '$script:key_pair_path'."
                exit 1
            }
        }
        else {
            Write-Host "Key pair '$script:key_pair_name' not found in AWS. Creating key pair..."
            aws ec2 create-key-pair --key-name $script:key_pair_name --query "KeyMaterial" --output text | Out-File -FilePath $script:key_pair_path -Encoding ascii
            icacls $script:key_pair_path /inheritance:r /grant:r ${env:USERNAME}:R
        }
    }

    icacls $script:key_pair_path /inheritance:r /grant:r ${env:USERNAME}:R
}

function Apply-Terraform {
    Write-Host "🌍 Initializing Terraform..."
    
    # Check credentials before proceeding
    Write-Host "Validating AWS credentials..."
    Setup-AWS-Credentials
    
    Push-Location "terraform"
    
    if (-not (Test-Path -Path $script:key_pair_path)) {
        Write-Host "❌ SSH key not found at: $script:key_pair_path"
        Write-Host "Please run the script with 'launch' option first to create the SSH key"
        Pop-Location
        exit 1
    }

    # Get absolute path to the key file
    $absolute_key_path = Resolve-Path $script:key_pair_path
    $terraform_key_path = $absolute_key_path.Path.Replace('\', '/')
    
    Write-Host "Using SSH key: $terraform_key_path"
    
    try {
        terraform init
        terraform apply -var "private_pem_key=$terraform_key_path" -auto-approve
    }
    catch {
        Write-Host "❌ Failed to apply Terraform configuration. Please check your AWS credentials and try again."
        Pop-Location
        exit 1
    }

    # Get NATS server URL from Terraform output but don't configure it yet
    $nats_ip = terraform output -raw nats_instance_public_ip
    if ($nats_ip) {
        Write-Host "✅ Got NATS IP: $nats_ip"
        Write-Host "The edge server will be configured with this NATS URL once it's running."
        $Global:nats_url = "nats://${nats_ip}:4222"
    }

    Pop-Location
}

function Configure-NatsUrl {
    # This function will be called from system_setup.ps1 after the server is running
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

    while (-not $success -and $retryCount -lt $maxRetries) {
        try {
            $body = @{
                cloud_url = $nats_url
            }
            
            $response = Invoke-RestMethod -Uri "http://localhost:8000/api/update-cloud-url" `
                -Method Post `
                -Body ($body | ConvertTo-Json) `
                -ContentType "application/json"

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
            Write-Host "⚠️ Server not ready yet, retrying..."
            $retryCount++
            Start-Sleep -Seconds 2
        }
    }

    if (-not $success) {
        Write-Host "⚠️ Failed to configure NATS URL automatically"
        Write-Host "Manual configuration steps:"
        Write-Host "1. Open http://localhost:8000 in your browser"
        Write-Host "2. Click on 'Settings'"
        Write-Host "3. Enter this NATS URL: $nats_url"
        Write-Host "4. Click 'Save'"
    }
}

function Destroy-Terraform {
    Write-Host "🛑 Destroying Terraform-managed infrastructure..."
    Push-Location "terraform"
    
    # Check credentials before proceeding
    Write-Host "Validating AWS credentials..."
    Setup-AWS-Credentials

    if (-not (Test-Path -Path $script:key_pair_path)) {
        Write-Host "❌ SSH key not found at: $script:key_pair_path"
        Write-Host "Nothing to destroy - no valid SSH key found"
        Pop-Location
        exit 1
    }

    $absolute_key_path = Resolve-Path $script:key_pair_path
    $terraform_key_path = $absolute_key_path.Path.Replace('\', '/')
    
    try {
        terraform destroy -var "private_pem_key=$terraform_key_path" -auto-approve
    }
    catch {
        Write-Host "❌ Failed to destroy Terraform configuration. Please check your AWS credentials and try again."
        Pop-Location
        exit 1
    }

    Pop-Location
    Write-Host "✅ Terraform-managed infrastructure destroyed."
}

function Get-NatsIp {
    Write-Host "🔍 Checking for existing NATS instance..."
    Push-Location "terraform"
    try {
        $nats_ip = terraform output -raw nats_instance_public_ip 2>$null
        if ($nats_ip) {
            Write-Host "✅ Found existing NATS instance"
            return $nats_ip
        }
    }
    catch {
        Write-Host "ℹ️ No existing NATS instance found"
    }
    finally {
        Pop-Location
    }
    return $null
}
