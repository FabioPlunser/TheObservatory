#!/usr/bin/env bash
# Exit on error
set -e

# Source the subscripts
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
source "$SCRIPT_DIR/terraform_setup.sh"
source "$SCRIPT_DIR/system_setup.sh"

# Initialize global variables
declare -a camera_pids=()
declare -a alarm_pids=()
server_pid=
simulated_camera_pid=
use_simulated_data=false
NATS_URL=

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Animated loading function
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

cleanup() {
    local exit_code=$?
    echo "🛑 Stopping all processes..."
    
    # Function to gracefully stop a process
    stop_process() {
        local pid=$1
        if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
            echo "Stopping process $pid..."
            kill -TERM $pid 2>/dev/null || kill -KILL $pid 2>/dev/null
            sleep 1
        fi
    }
    
    # Stop camera processes
    for pid in "${camera_pids[@]}"; do
        stop_process "$pid"
    done
    
    # Stop alarm processes
    for pid in "${alarm_pids[@]}"; do
        stop_process "$pid"
    done
    
    # Stop simulated camera process group
    if [ "$use_simulated_data" = true ] && [ -n "$simulated_camera_pid" ]; then
        stop_process "$simulated_camera_pid"
    fi
    
    # Stop server more gracefully
    if [ -n "$server_pid" ]; then
        echo "Stopping server (PID: $server_pid)..."
        kill -TERM $server_pid 2>/dev/null
        sleep 2
        # If still running, force kill
        if kill -0 $server_pid 2>/dev/null; then
            kill -KILL $server_pid 2>/dev/null
        fi
    fi
    
    # Stop any remaining Python processes
    pkill -f "python.*onPremise/server/main.py" 2>/dev/null || true
    
    # Clean up Docker containers
    if command -v docker >/dev/null 2>&1; then
        docker rm -f rtsp >/dev/null 2>&1 || true
    fi
    
    # Only clean up files if exiting successfully
    if [ $exit_code -eq 0 ]; then
        echo "🧹 Cleaning up files..."
        rm -f db.db server_output.log server_error.log
    else
        echo "⚠️ Exit code: $exit_code - preserving log files for inspection"
        echo "📝 Check server_output.log and server_error.log for details"
    fi
    
    echo "👋 Cleanup complete"
    exit $exit_code
}

# Set up error handling
set -E
trap 'handle_error $? $LINENO' ERR

handle_error() {
    local exit_code=$1
    local line_number=$2
    echo "❌ Error on line $line_number: Command exited with status $exit_code"
    cleanup
}

# Register the cleanup function for normal exit and signals
trap cleanup EXIT SIGINT SIGTERM

# Set up cleanup trap
trap cleanup SIGINT SIGTERM EXIT

echo "🚀 Starting setup script..."

# Handle Terraform operations
read -p "Do you want to launch, destroy, or do nothing with the Terraform server? ([l]aunch/[d]estroy/[n]othing): " action
handle_terraform "$action"

# Run system setup
setup_system

# Keep script running
echo "✅ System is running. Press Ctrl+C to stop all processes."
wait