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

echo "🚀 Starting setup script..."

# Create and activate Python virtual environment
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv venv >/dev/null 2>&1
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Get number of devices to emulate
read -p "Enter number of cameras to emulate: " num_cameras
read -p "Enter number of alarms to emulate: " num_alarms

# Start frontend setup
animate_loading "Setting up frontend..." &
spinner_pid=$!

# Install frontend dependencies
if command_exists bun; then
    (cd server/website && bun install && bun run build) >/dev/null 2>&1 && \
    cleanup_spinner "✅ Frontend setup complete!" || {
        cleanup_spinner "❌ Frontend setup failed!"
        exit 1
    }
elif command_exists npm; then
    (cd server/website && npm install && npm run build) >/dev/null 2>&1 && \
    cleanup_spinner "✅ Frontend setup complete!" || {
        cleanup_spinner "❌ Frontend setup failed!"
        exit 1
    }
else
    cleanup_spinner "❌ Neither bun nor npm found. Please install one of them."
    exit 1
fi

# Start Python dependencies installation
animate_loading "Installing Python dependencies..." &
spinner_pid=$!

# Install Python requirements
{
    pip install -r server/requirements.txt >/dev/null 2>&1
    pip install -r devices/emulated/requirements.txt >/dev/null 2>&1
} && cleanup_spinner "✅ Python dependencies installed!" || {
    cleanup_spinner "❌ Python dependencies installation failed!"
    exit 1
}

# Start the server in the background
echo "🖥️ Starting edge server..."
python server/edge_server.py &
server_pid=$!

# Wait for server to start
echo "⏳ Waiting for server to initialize..."
sleep 5

# Start emulated cameras
echo "📸 Starting $num_cameras camera(s)..."
for (( i=1; i<=$num_cameras; i++ ))
do
    python devices/emulated/camera.py &
    camera_pids+=($!)
done

# Start emulated alarms
echo "🚨 Starting $num_alarms alarm(s)..."
for (( i=1; i<=$num_alarms; i++ ))
do
    python devices/emulated/alarm.py &
    alarm_pids+=($!)
done

# Trap SIGINT and SIGTERM signals
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
    # Kill server
    kill $server_pid 2>/dev/null || true
    
    echo "👋 Cleanup complete"
    rm -rf db.db 
    exit 0
}

trap cleanup SIGINT SIGTERM

# Keep script running
echo "✅ System is running. Press Ctrl+C to stop all processes."
wait