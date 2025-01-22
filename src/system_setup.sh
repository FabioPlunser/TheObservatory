#!/usr/bin/env bash
# Exit on error
set -e

setup_python_environment() {
    # Check Python version and create virtual environment
    if ! command -v python3.12 &> /dev/null; then
        echo "❌ Python 3.12 is required but not found"
        exit 1
    fi

    # Create virtual environment with specific Python version
    if [ ! -d "venv" ]; then
        echo "🐍 Creating Python virtual environment..."
        python3.12 -m venv venv
        if [ $? -ne 0 ]; then
            echo "❌ Failed to create virtual environment"
            exit 1
        fi
    fi

    echo "🔄 Activating virtual environment..."
    source venv/bin/activate
    
    # Verify activation and Python version
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "❌ Failed to activate virtual environment"
        exit 1
    fi

    # Upgrade pip to latest version
    echo "📦 Upgrading pip..."
    python -m pip install --upgrade pip

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
        for req_file in "${requirements_files[@]}"; do
            echo "Installing requirements from: $req_file"
            if ! pip install -r "$req_file"; then
                echo "❌ Failed to install requirements from $req_file"
                exit 1
            fi
        done
        
        # Install additional required packages
        echo "📦 Installing additional required packages..."
        pip install fastapi uvicorn python-multipart python-jose[cryptography] passlib[bcrypt] pydantic email-validator python-jose sqlalchemy

        echo -n "$checksum" > "$checksum_file"
    else
        echo "📦 Python packages are already up-to-date."
    fi

    # Verify critical packages
    echo "🔍 Verifying critical packages..."
    python -c "import fastapi; import uvicorn; import http.client" || {
        echo "❌ Failed to import critical packages"
        exit 1
    }
}

setup_frontend() {
    # Get the root directory path
    root_dir=$(dirname "$(dirname "$0")")
    frontend_path="$root_dir/src/onPremise/server/website"

    echo "Looking for frontend at: $frontend_path"

    if [ ! -d "$frontend_path" ]; then
        # Try alternative path resolution
        alt_path="onPremise/server/website"
        echo "Frontend not found, trying alternative path: $alt_path"
        
        if [ -d "$alt_path" ]; then
            frontend_path="$alt_path"
            echo "✅ Found frontend at alternative path"
        else
            echo "❌ Frontend directory not found at either:"
            echo "   1. $frontend_path"
            echo "   2. $alt_path"
            echo "Please ensure the frontend directory exists in one of these locations."
            exit 1
        fi
    fi

    echo "Setting up frontend at: $frontend_path"
    
    pushd "$frontend_path" >/dev/null
    
    if command -v bun >/dev/null 2>&1; then
        bun install
        bun run build
        echo "✅ Frontend setup complete using bun!"
    elif command -v npm >/dev/null 2>&1; then
        echo "ℹ️ NPM requires sudo, please enter your password if prompted."
        sudo npm install
        sudo npm run build
        echo "✅ Frontend setup complete using npm!"
    else
        echo "❌ Neither bun nor npm found. Please install one of them."
        exit 1
    fi
    
    popd >/dev/null
}

start_server() {
    # Check if port 8000 is in use
    echo "🔍 Checking if port 8000 is available..."
    if lsof -i:8000 >/dev/null 2>&1; then
        echo "❌ Port 8000 is already in use. Attempting to free it..."
        if ! sudo lsof -i:8000 -t | xargs kill -9; then
            echo "⚠️ Failed to stop process. Please manually stop any process using port 8000"
            echo "   You can use 'lsof -i:8000' to find the process"
            cleanup
            exit 1
        fi
        # Wait for port to be freed
        sleep 2
    fi

    # Delete existing database and log files
    echo "🗑️ Cleaning up old database..."
    rm -f db.db log.log server_error.log server_output.log

    echo "🖥️ Starting edge server..."
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
    
    # Check Python environment
    echo "Python interpreter being used: $(which python)"
    echo "Python version: $(python --version)"
    echo "Virtual env path: $VIRTUAL_ENV"
    
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "❌ Virtual environment is not activated!"
        echo "Attempting to activate virtual environment..."
        if [ -f "venv/bin/activate" ]; then
            source venv/bin/activate
            echo "✅ Virtual environment activated"
        else
            echo "❌ Cannot find virtual environment at venv/bin/activate"
            exit 1
        fi
    fi
    
    # Get absolute path to server directory
    SERVER_DIR=$(readlink -f "onPremise/server")
    echo "Server directory: $SERVER_DIR"
    
    if [ ! -f "$SERVER_DIR/main.py" ]; then
        echo "❌ Server file not found at $SERVER_DIR/main.py"
        echo "Current directory structure:"
        ls -R
        exit 1
    fi
    
    # Add the server directory to PYTHONPATH
    export PYTHONPATH="$SERVER_DIR:$PYTHONPATH"
    
    echo "Starting server process with verbose output..."
    python -v "$SERVER_DIR/main.py" > server_output.log 2> server_error.log &
    server_pid=$!
    
    # Immediate check if process started
    if ! ps -p $server_pid > /dev/null; then
        echo "❌ Server process failed to start. Error log:"
        cat server_error.log
        exit 1
    fi

    echo "⏳ Checking if the server is already up..."
    if curl -s http://localhost:8000/api/get-company >/dev/null 2>&1; then
        echo "✅ Server is already responding. Skipping wait."
        return
    fi

    echo "⏳ Waiting for server to initialize..."
    max_wait_time=40
    waited=0
    server_started=false

    while [ $waited -lt $max_wait_time ] && [ "$server_started" = false ]; do
        sleep 1
        waited=$((waited + 1))
        printf "\rWaiting for server... %d/%d seconds" $waited $max_wait_time

        # Check if server process is still running
        if ! ps -p $server_pid > /dev/null; then
            echo -e "\n❌ Server process exited prematurely."
            echo "Server output log:"
            cat server_output.log
            echo -e "\nServer error log:"
            cat server_error.log
            cleanup
            exit 1
        fi
        
        # Check for common error patterns in the logs
        if grep -q "ModuleNotFoundError" server_error.log; then
            echo -e "\n❌ Missing Python module. Error details:"
            cat server_error.log
            cleanup
            exit 1
        fi
        
        if grep -q "ImportError" server_error.log; then
            echo -e "\n❌ Import error detected. Error details:"
            cat server_error.log
            cleanup
            exit 1
        fi
        
        # Check for successful startup in output log
        if grep -q "Application startup complete" server_output.log && \
           grep -q "Successfully registered mDNS service" server_output.log; then
            server_started=true
            break
        fi

        # Check for port conflict error
        if grep -q "error while attempting to bind on address" server_error.log; then
            echo -e "\n❌ Port 8000 is still in use. Please free the port and try again."
            cleanup
            exit 1
        fi

        # Add periodic status update from the logs
        if [ $((waited % 10)) -eq 0 ]; then
            echo -e "\nCurrent server status:"
            tail -n 5 server_output.log
        fi
    done

    # Verify server is responding
    if curl -s http://localhost:8000/api/get-company >/dev/null 2>&1; then
        echo -e "\n✅ Server is responding to requests"
    else
        echo -e "\n❌ Server is not responding to requests. Check server_error.log for details."
        cat server_error.log
        cleanup
        exit 1
    fi

    echo -e "\n✅ Server started successfully!"
    echo "ℹ️ Note: NATS warnings are normal when no cloud URL is configured."
}

configure_nats_url() {
    local nats_url="$1"

    if [ -z "$nats_url" ]; then
        echo "ℹ️ No NATS URL to configure"
        return
    fi

    echo "🔄 Configuring edge server with NATS URL: $nats_url"
    
    max_retries=5
    retry_count=0
    success=false

    # Try to verify server is actually running first
    if ! curl -s http://localhost:8000/api/get-company >/dev/null 2>&1; then
        echo "❌ Edge server is not running. Cannot configure NATS URL."
        echo "Please wait for the server to start or check server_error.log for issues."
        return
    fi

    while [ $retry_count -lt $max_retries ] && [ "$success" = false ]; do
        # URL encode the NATS URL
        encoded_url=$(python - <<EOF
import urllib.parse
nats_url = """$nats_url"""
print(urllib.parse.quote(nats_url))
EOF
        )
        
        response=$(curl -s -X POST "http://localhost:8000/api/update-cloud-url?cloud_url=$encoded_url")
        
        if echo "$response" | grep -q '"status":"success"'; then
            echo "✅ Edge server configured with NATS URL"
            success=true
            break
        fi
        
        echo "⚠️ Retry $((retry_count + 1)) of $max_retries..."
        ((retry_count++))
        sleep 2
    done

    if [ "$success" = false ]; then
        echo "⚠️ Failed to configure NATS URL automatically"
        echo "Manual configuration steps:"
        echo "1. Wait for the server to fully start"
        echo "2. Open http://localhost:8000 in your browser"
        echo "3. Click on 'Settings'"
        echo "4. Enter this NATS URL: $nats_url"
        echo "5. Click 'Save'"
    fi
}

test_docker_running() {
    if ! docker info >/dev/null 2>&1; then
        echo "❌ Docker is not running. Please start Docker Desktop and try again."
        return 1
    fi
    return 0
}

start_rtsp_server() {
    echo "🎥 Starting RTSP server..."
    
    if ! test_docker_running; then
        exit 1
    fi

    # Check if container already exists
    if docker ps -a --format '{{.Names}}' | grep -q "^rtsp$"; then
        # Remove existing container
        docker rm -f rtsp >/dev/null 2>&1
    fi

    if ! docker run -d --name rtsp -p 8555:8554 aler9/rtsp-simple-server >/dev/null 2>&1; then
        echo "❌ Failed to start RTSP server"
        exit 1
    fi
    
    echo "✅ RTSP server started successfully"
}

start_devices() {
    # Get number of devices to emulate
    read -p "Enter number of cameras to emulate: " num_cameras
    read -p "Enter number of alarms to emulate: " num_alarms
    read -p "Do you want to use simulated data? (y/n): " use_simulated_data

    if ["$num_cameras" -gt 0]; then
    echo "📸 Starting $num_cameras camera(s)..."
    camera_pids=()
    python onPremise/devices/emulated/camera.py &
    camera_pids+=($!)

    num_cameras=$((num_cameras-1))
    fi

    if [ "$use_simulated_data" = "y" ]; then
        echo "📸 Starting simulated cameras ..."
        # Start in a new process group
        setsid python onPremise/devices/emulated/simulatedCamera.py --streams "$num_cameras" &
          simulated_camera_pid=$!
    fi

    echo "🚨 Starting $num_alarms alarm(s)..."
    alarm_pids=()
    for ((i=1; i<=$num_alarms; i++)); do
        python onPremise/devices/emulated/alarm.py &
        alarm_pids+=($!)
    done
}

setup_system() {
    # Start RTSP server first
    start_rtsp_server

    # Then setup Python and start server
    setup_python_environment
    setup_frontend
    start_server

    # Configure NATS after server is running
    if [ -n "$NATS_URL" ]; then
        sleep 5  # Give the server a moment to fully initialize
        configure_nats_url "$NATS_URL"
    fi

    # Finally start devices
    start_devices
}

# Export functions to be used by other scripts
export -f setup_system
export -f configure_nats_url