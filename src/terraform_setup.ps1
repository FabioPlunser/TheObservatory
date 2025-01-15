# Script-level variables
$script:key_pair_name = "theObservatory"
$script:key_pair_path = "$env:USERPROFILE\.ssh\theObservatory.pem"
$script:ssh_dir = "$env:USERPROFILE\.ssh"

function Handle-Terraform {
    param (
        [string]$action
    )

    if ($action -eq "l") {
        Setup-AWS-Credentials
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
        $nats_ip = Get-NatsIp
        if ($nats_ip) {
            Write-Host "🔄 Configuring edge server with existing NATS URL..."
            try {
                $response = Invoke-RestMethod -Uri "http://localhost:8000/api/update-cloud-url" -Method Post -Body "nats://${nats_ip}:4222" -ContentType "application/x-www-form-urlencoded"
                if ($response.status -eq "success") {
                    Write-Host "✅ Edge server configured with NATS URL"
                }
                else {
                    Write-Host "⚠️ Failed to configure edge server: $($response.message)"
                }
            }
            catch {
                Write-Host "⚠️ Failed to configure edge server. You can manually set the NATS URL in the web interface."
                Write-Host "   NATS URL: nats://${nats_ip}:4222"
            }
        }
    }
    else {
        Write-Host "🚫 Invalid action. Please enter 'launch' or 'destroy'."
        exit 1
    }
}

function Setup-AWS-Credentials {
    $aws_credentials_path = "$env:USERPROFILE\.aws\credentials"
    $aws_credentials_exist = Test-Path -Path $aws_credentials_path

    if (-Not $aws_credentials_exist) {
        Write-Host "AWS credentials file not found. Prompting for credentials..."
        $aws_access_key = Read-Host "Enter your AWS Access Key"
        $aws_secret_key = Read-Host "Enter your AWS Secret Key"
        $aws_region = Read-Host "Enter your AWS Region (e.g., us-east-1)"

        aws configure set aws_access_key_id $aws_access_key
        aws configure set aws_secret_access_key $aws_secret_key
        aws configure set region $aws_region
    }
    else {
        Write-Host "AWS credentials file found. Checking for credentials..."
        $credentials_content = Get-Content -Path $aws_credentials_path -Raw
        if ($credentials_content -notmatch "aws_access_key_id" -or $credentials_content -notmatch "aws_secret_access_key") {
            Write-Host "AWS credentials not found in the file. Prompting for credentials..."
            $aws_access_key = Read-Host "Enter your AWS Access Key"
            $aws_secret_key = Read-Host "Enter your AWS Secret Key"
            $aws_region = Read-Host "Enter your AWS Region (e.g., us-east-1)"

            aws configure set aws_access_key_id $aws_access_key
            aws configure set aws_secret_access_key $aws_secret_key
            aws configure set region $aws_region
        }
    }
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
    Push-Location "terraform"
    
    if (-not (Test-Path -Path $script:key_pair_path)) {
        Write-Host "❌ SSH key not found at: $script:key_pair_path"
        Write-Host "Please run the script with 'launch' option first to create the SSH key"
        Pop-Location
        exit 1
    }

    # Get absolute path to the key file
    $absolute_key_path = Resolve-Path $script:key_pair_path

    Write-Host "🚀 Applying Terraform configuration..."
    terraform init

    # Fix path format for terraform
    $terraform_key_path = $absolute_key_path.Path.Replace('\', '/')
    
    Write-Host "Using SSH key: $terraform_key_path"
    terraform apply -var "private_pem_key=$terraform_key_path" -auto-approve

    # Get NATS server URL from Terraform output
    $nats_ip = terraform output -raw nats_instance_public_ip
    
    Write-Host "🔄 Configuring edge server with NATS URL..."
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/api/update-cloud-url" -Method Post -Body "nats://${nats_ip}:4222" -ContentType "application/x-www-form-urlencoded"
        if ($response.status -eq "success") {
            Write-Host "✅ Edge server configured with NATS URL"
        }
        else {
            Write-Host "⚠️ Failed to configure edge server: $($response.message)"
        }
    }
    catch {
        Write-Host "⚠️ Failed to configure edge server. You can manually set the NATS URL in the web interface."
        Write-Host "   NATS URL: nats://${nats_ip}:4222"
    }

    Pop-Location
}

function Destroy-Terraform {
    Write-Host "🛑 Destroying Terraform-managed infrastructure..."
    Push-Location "terraform"
    
    if (-not (Test-Path -Path $script:key_pair_path)) {
        Write-Host "❌ SSH key not found at: $script:key_pair_path"
        Write-Host "Nothing to destroy - no valid SSH key found"
        Pop-Location
        exit 1
    }

    $absolute_key_path = Resolve-Path $script:key_pair_path
    $terraform_key_path = $absolute_key_path.Path.Replace('\', '/')
    
    terraform destroy -var "private_pem_key=$terraform_key_path" -auto-approve
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
    Pop-Location
    return $null
}
