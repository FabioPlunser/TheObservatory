#!/usr/bin/env bash

# Exit on error
set -e

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "🚀 Starting setup script..."

# Create and activate Python virtual environment
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Get number of devices to emulate
read -p "Enter number of cameras to emulate: " num_cameras
read -p "Enter number of alarms to emulate: " num_alarms

# Check for bun or npm and install dependencies
if command_exists bun; then
    echo "🔧 Installing frontend dependencies with bun..."
    cd server/website
    bun install
    bun run build
    cd ../..
elif command_exists npm; then
    echo "🔧 Installing frontend dependencies with npm..."
    cd server/website
    npm install
    npm run build
    cd ../..
else
    echo "❌ Neither bun nor npm found. Please install one of them."
    exit 1
fi

# Install Python requirements
echo "📦 Installing server requirements..."
pip install -r server/requirements.txt

echo "📦 Installing emulated devices requirements..."
pip install -r devices/emulated/requirements.txt

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
    exit 0

    rm -rf db.db 

}

trap cleanup SIGINT SIGTERM

# Keep script running
echo "✅ System is running. Press Ctrl+C to stop all processes."
wait