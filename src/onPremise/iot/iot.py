import os
import threading
import subprocess
import signal

HOST_ADDR = "rtmp://localhost:1935"

STREAM_URLS = [
    HOST_ADDR + "/live/stream1",
    HOST_ADDR + "/live/stream2",
    HOST_ADDR + "/live/stream3",
    HOST_ADDR + "/live/stream4",
    HOST_ADDR + "/live/stream5",
    HOST_ADDR + "/live/stream6"
]

stop_flag = threading.Event()

def signal_handler(sig, frame):
    print('Stopping...')
    stop_flag.set()

signal.signal(signal.SIGINT, signal_handler)

def send_video(video_paths, stream_url):
    while not stop_flag.is_set():
        for video_path in video_paths:
            command = [
                'ffmpeg',
                '-re',                  # Read input at native frame rate
                '-i', video_path,       # Input file
                '-c:v', 'libx264',      # Video codec
                '-preset', 'veryfast',  # Encoding speed
                '-f', 'flv',            # Output format
                '-loglevel', 'error',   # Suppress warnings, show only errors
                stream_url              # Output URL
            ]
            process = subprocess.Popen(command)
            while process.poll() is None:
                if stop_flag.is_set():
                    process.terminate()
                    break
            process.wait()

def main():
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/video_sets'))

    threads = []

    for i in range(1, 7):  # Assuming 6 different cameras
        stream_url = STREAM_URLS[i - 1]
        video_paths = []
        for set_num in range(1, 12):  # Assuming 11 sets
            if i == 6 and set_num < 5:  # Skip sets 1 to 4 for the 6th camera
                continue
            video_path = os.path.join(dataset_dir, f'set_{set_num}', f'video{set_num}_{i}.avi')
            if os.path.exists(video_path):
                video_paths.append(video_path)
            else:
                print(f"Video {video_path} does not exist")
        
        if video_paths:
            print(f"Starting stream for {stream_url} with videos: {video_paths}")
            thread = threading.Thread(target=send_video, args=(video_paths, stream_url))
            thread.start()
            threads.append(thread)

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()