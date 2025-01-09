#!/usr/bin/env bash
# Exit on error
set -e

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Animated loading function with better process handling
animate_loading() {
    local message=$1
    local frames='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    local delay=0.1
    local i=0
    
    while kill -0 $PPID 2>/dev/null; do
        printf "\r${frames:i++%${#frames}:1} %s" "$message"
        sleep $delay
    done
}

cleanup_spinner() {
    kill $spinner_pid 2>/dev/null || true
    wait $spinner_pid 2>/dev/null || true
    printf "\r%s\n" "$1"
}

cleanup() {
    echo "🛑 Stopping all processes..."
    # Kill any remaining spinner
    kill $spinner_pid 2>/dev/null || true
    # Kill camera processes
    for pid in "${camera_pids[@]}"; do
        kill $pid 2>/dev/null || true
    done
    # Kill alarm processes
    for pid in "${alarm_pids[@]}"; do
        kill $pid 2>/dev/null || true
    done
    # Kill simulated camera process group
    if [ "$use_simulated_data" = true ]; then
        # Kill the entire process group
        kill -- -$simulated_camera_pid 2>/dev/null || true
    fi
    # Kill server
    kill $server_pid 2>/dev/null || true
    
    echo "👋 Cleanup complete"
    rm -rf db.db 
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "🚀 Starting setup script..."

# Prompt the user to launch or destroy the Terraform server
read -p "Do you want to launch, destroy, or do nothing with the Terraform server? ([l]aunch/[d]estroy/[n]othing): " action
if [ "$action" = "l" ]; then

    # Check if AWS credentials file exists
    aws_credentials_path="$HOME/.aws/credentials"
    if [ ! -f "$aws_credentials_path" ]; then
        echo "AWS credentials file not found. Prompting for credentials..."
        read -p "Enter your AWS Access Key: " aws_access_key
        read -p "Enter your AWS Secret Key: " aws_secret_key
        read -p "Enter your AWS Region (e.g., us-east-1): " aws_region

        # Configure AWS CLI with the provided credentials
        aws configure set aws_access_key_id $aws_access_key
        aws configure set aws_secret_access_key $aws_secret_key
        aws configure set region $aws_region
    else
        echo "AWS credentials file found. Checking for credentials..."

        # Check if the credentials are present
        if ! grep -q "aws_access_key_id" "$aws_credentials_path" || ! grep -q "aws_secret_access_key" "$aws_credentials_path"; then
            echo "AWS credentials not found in the file. Prompting for credentials..."
            read -p "Enter your AWS Access Key: " aws_access_key
            read -p "Enter your AWS Secret Key: " aws_secret_key
            read -p "Enter your AWS Region (e.g., us-east-1): " aws_region

            # Configure AWS CLI with the provided credentials
            aws configure set aws_access_key_id $aws_access_key
            aws configure set aws_secret_access_key $aws_secret_key
            aws configure set region $aws_region
        else
            echo "AWS credentials found in the file."
        fi
    fi

    # Check if the key pair exists
    key_pair_name="theObservatory"
    if ! aws ec2 describe-key-pairs --key-names $key_pair_name 2>&1 | grep -q $key_pair_name; then
        echo "Key pair '$key_pair_name' not found. Creating key pair..."
        key_pair_path="$HOME/.ssh/theObservatory.pem"
        aws ec2 create-key-pair --key-name $key_pair_name --query "KeyMaterial" --output text > $key_pair_path
        chmod 400 $key_pair_path
        echo "Key pair created and saved to $key_pair_path"
    else
        echo "Key pair '$key_pair_name' found."
        key_pair_path="$HOME/.ssh/theObservatory.pem"
    fi

    echo "🌍 Initializing Terraform..."
    pushd terraform
    terraform init

    echo "🚀 Applying Terraform configuration..."
    terraform apply -var "private_pem_key=$key_pair_path" -auto-approve


elif [ "$action" = "d" ]; then
    echo "🛑 Destroying Terraform-managed infrastructure..."
    pushd terraform
    terraform destroy -var "private_pem_key=$key_pair_path" -auto-approve
    popd
    echo "✅ Terraform-managed infrastructure destroyed."
    read -p "Do you want to continue with the setup script? (y/n): " close
    if [ "$close" = "n" ]; then
        exit
    fi

elif [ "$action" = "n" ]; then
    echo "Continuing with the setup script..."
else
    echo "🚫 Invalid action. Please enter 'l', 'd', or 'n'."
    exit 1
fi

# Create and activate Python virtual environment
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv venv >/dev/null 2>&1
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Calculate checksum of requirements files
requirements_files=("onPremise/server/requirements.txt" "onPremise/devices/emulated/requirements.txt" "cloud/requirements.txt")
checksum=$(cat "${requirements_files[@]}" | sha256sum | awk '{print $1}')
checksum_file="requirements_checksum.txt"

# Check if checksum has changed or if virtual environment does not exist
run_pip_install=false
if [ ! -f "$checksum_file" ]; then
    run_pip_install=true
else
    stored_checksum=$(cat "$checksum_file" | tr -d '[:space:]')
    if [ "$checksum" != "$stored_checksum" ]; then
        run_pip_install=true
    fi
fi

# Install Python requirements if needed
if [ "$run_pip_install" = true ]; then
    echo "📦 Installing required Python packages..."
    pip install -r onPremise/server/requirements.txt
    pip install -r onPremise/devices/emulated/requirements.txt
    pip install -r cloud/requirements.txt
    echo -n "$checksum" > "$checksum_file"
else
    echo "📦 Python packages are already up-to-date."
fi

# Get number of devices to emulate
read -p "Enter number of cameras to emulate: " num_cameras
read -p "Enter number of alarms to emulate: " num_alarms
read -p "Do you want to use simulated data? (y/n): " use_simulated_data
if [ "$use_simulated_data" = "y" ]; then
    use_simulated_data=true
else
    use_simulated_data=false
fi

# Start frontend setup
animate_loading "Setting up frontend..." &
spinner_pid=$!

# Install frontend dependencies
if command_exists bun; then
    (cd onPremise/server/website && bun install && bun run build) >/dev/null 2>&1 && \
    cleanup_spinner "✅ Frontend setup complete!" || {
        cleanup_spinner "❌ Frontend setup failed!"
        exit 1
    }
elif command_exists npm; then
    (cd onPremise/server/website && npm install && npm run build) >/dev/null 2>&1 && \
    cleanup_spinner "✅ Frontend setup complete!" || {
        cleanup_spinner "❌ Frontend setup failed!"
        exit 1
    }
else
    cleanup_spinner "❌ Neither bun nor npm found. Please install one of them."
    exit 1
fi

# Start the server in the background
echo "🖥️ Starting edge server..."
python onPremise/server/edge_server.py &
server_pid=$!

# Wait for server to start
echo "⏳ Waiting for server to initialize..."
sleep 5

# Start emulated cameras
echo "📸 Starting $num_cameras camera(s)..."
camera_pids=()
for (( i=1; i<=$num_cameras; i++ ))
do
    python onPremise/devices/emulated/camera.py &
    camera_pids+=($!)
done

# Start simulated cameras
if [ "$use_simulated_data" = true ]; then
    echo "📸 Starting simulated cameras ..."
    # Start in a new process group
    setsid python onPremise/devices/emulated/simulatedCamera.py &
    simulated_camera_pid=$!
fi

# Start emulated alarms
echo "🚨 Starting $num_alarms alarm(s)..."
alarm_pids=()
for (( i=1; i<=$num_alarms; i++ ))
do
    python onPremise/devices/emulated/alarm.py &
    alarm_pids+=($!)
done

# Keep script running
echo "✅ System is running. Press Ctrl+C to stop all processes."
wait