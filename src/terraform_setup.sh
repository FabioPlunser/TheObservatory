#!/usr/bin/env bash
# Exit on error
set -e

# Script-level variables
KEY_PAIR_NAME="theObservatory"
KEY_PAIR_PATH="$HOME/.ssh/theObservatory.pem"
SSH_DIR="$HOME/.ssh"

test_aws_credentials() {
    # Test basic AWS CLI access
    if ! /usr/local/bin/aws sts get-caller-identity >/dev/null 2>&1; then
        echo "❌ AWS CLI access failed"
        return 1
    fi

    # Check for token expiration using terraform
    pushd .terraform/ >/dev/null
    if terraform providers 2>&1 | grep -q "ExpiredToken"; then
        echo "❌ Token has expired"
        popd >/dev/null
        return 1
    fi
    popd >/dev/null
    return 0
}

setup_aws_credentials() {
    local aws_credentials_path="$HOME/.aws/credentials"
    
    if [ ! -f "$aws_credentials_path" ]; then
        echo "❌ AWS credentials file not found at: $aws_credentials_path"
        echo "Please configure your AWS credentials and try again."
        exit 1
    fi
    
    if ! test_aws_credentials; then
        echo "❌ AWS credentials are expired or invalid."
        echo "Please update your AWS credentials at: $aws_credentials_path"
        echo "You can use 'aws configure' to set up new credentials."
        echo "Note: If you're using temporary credentials or an AWS token, make sure it hasn't expired."
        exit 1
    fi
    
    echo "✅ AWS credentials are valid and not expired"
}

setup_ssh_keys() {
    # Create .ssh directory if it doesn't exist
    mkdir -p "$SSH_DIR"
    
    if [ ! -f "$KEY_PAIR_PATH" ]; then
        echo "Key pair file '$KEY_PAIR_PATH' not found. Checking if key pair exists in AWS..."
        if aws ec2 describe-key-pairs --key-names "$KEY_PAIR_NAME" 2>&1 | grep -q "$KEY_PAIR_NAME"; then
            echo "Key pair '$KEY_PAIR_NAME' exists in AWS but the local file is missing."
            read -p "Do you want to create a new key pair? (y/n): " create_new_key
            if [ "$create_new_key" = "y" ]; then
                aws ec2 delete-key-pair --key-name "$KEY_PAIR_NAME"
                aws ec2 create-key-pair --key-name "$KEY_PAIR_NAME" --query "KeyMaterial" --output text > "$KEY_PAIR_PATH"
                chmod 400 "$KEY_PAIR_PATH"
            else
                echo "Please ensure you have the correct key pair file at '$KEY_PAIR_PATH'."
                exit 1
            fi
        else
            echo "Key pair '$KEY_PAIR_NAME' not found in AWS. Creating key pair..."
            aws ec2 create-key-pair --key-name "$KEY_PAIR_NAME" --query "KeyMaterial" --output text > "$KEY_PAIR_PATH"
            chmod 400 "$KEY_PAIR_PATH"
        fi
    fi

    chmod 400 "$KEY_PAIR_PATH"
}

apply_terraform() {
    echo "🌍 Initializing Terraform..."
    
    # Check credentials before proceeding
    echo "Validating AWS credentials..."
    setup_aws_credentials
    
    pushd .terraform/ >/dev/null
    
    if [ ! -f "$KEY_PAIR_PATH" ]; then
        echo "❌ SSH key not found at: $KEY_PAIR_PATH"
        echo "Please run the script with 'launch' option first to create the SSH key"
        popd >/dev/null
        exit 1
    fi

    # Get absolute path to the key file
    KEY_PATH_ABS=$(readlink -f "$KEY_PAIR_PATH")
    
    echo "Using SSH key: $KEY_PATH_ABS"
    
    terraform init
    if ! terraform apply -var "private_pem_key=$KEY_PATH_ABS" -auto-approve; then
        echo "❌ Failed to apply Terraform configuration. Please check your AWS credentials and try again."
        popd >/dev/null
        exit 1
    fi

    # Get NATS server URL from Terraform output
    NATS_IP=$(terraform output -raw nats_instance_public_ip)
    if [ -n "$NATS_IP" ]; then
        echo "✅ Got NATS IP: $NATS_IP"
        echo "The edge server will be configured with this NATS URL once it's running."
        export NATS_URL="nats://${NATS_IP}:4222"
    fi

    popd >/dev/null
}

destroy_terraform() {
    echo "🛑 Destroying Terraform-managed infrastructure..."
    pushd terraform/ >/dev/null

    # Check credentials before proceeding
    echo "Validating AWS credentials..."
    setup_aws_credentials

    if [ ! -f "$KEY_PAIR_PATH" ]; then
        echo "❌ SSH key not found at: $KEY_PAIR_PATH"
        echo "Nothing to destroy - no valid SSH key found"
        popd >/dev/null
        exit 1
    fi

    KEY_PATH_ABS=$(readlink -f "$KEY_PAIR_PATH")

    if ! terraform destroy -var "private_pem_key=$KEY_PATH_ABS" -auto-approve; then
        echo "❌ Failed to destroy Terraform configuration. Please check your AWS credentials and try again."
        popd >/dev/null
        exit 1
    fi

    popd >/dev/null
    echo "✅ Terraform-managed infrastructure destroyed."
}

get_nats_ip() {
    #echo "🔍 Checking for existing NATS instance..."
    pushd terraform >/dev/null
    if NATS_IP=$(terraform output -raw nats_instance_public_ip 2>/dev/null); then
        #echo "✅ Found existing NATS instance"
        echo "$NATS_IP"
        return 0
    else
        echo "ℹ️ No existing NATS instance found"
        return 1
    fi
    popd >/dev/null
}

handle_terraform() {
    local action=$1
    
    # Store the starting location
    START_DIR=$(pwd)

    if [ "$action" = "l" ]; then
        setup_ssh_keys
        apply_terraform
    elif [ "$action" = "d" ]; then
        destroy_terraform
        read -p "Do you want to continue with the setup script? (y/n): " close
        if [ "$close" = "n" ]; then
            exit 0
        fi
    elif [ "$action" = "n" ]; then
        echo "Continuing with the setup script..."
        # Store NATS IP for later use if it exists
        if NATS_IP=$(get_nats_ip); then
            export NATS_URL="nats://${NATS_IP}:4222"
            echo "✅ Found existing NATS URL: $NATS_URL"
            echo "The edge server will be configured with this URL once it's running."
        fi
    else
        echo "🚫 Invalid action. Please enter 'l'aunch, 'd'estroy, or 'n'othing."
        exit 1
    fi

    # Always return to starting location
    cd "$START_DIR"
}

# Export functions to be used by other scripts
export -f handle_terraform
