import os
import threading
import subprocess
import time

FRAME_INTERVAL = 1  # Process every n-th frame set to 1 to process all frames
STREAM_URLS = [
    "rtmp://44.204.173.180:1935/live/stream1",
    "rtmp://44.204.173.180:1935/live/stream2",
    "rtmp://44.204.173.180:1935/live/stream3",
    "rtmp://44.204.173.180:1935/live/stream4",
    "rtmp://44.204.173.180:1935/live/stream5",
    "rtmp://44.204.173.180:1935/live/stream6"
]

def send_video(video_paths, stream_url):
    while True:
        for video_path in video_paths:
            command = [
                'ffmpeg',
                '-re',  # Read input at native frame rate
                '-i', video_path,  # Input file
                '-c:v', 'libx264',  # Video codec
                '-preset', 'veryfast',  # Encoding speed
                '-f', 'flv',  # Output format
                '-loglevel', 'error',  # Suppress warnings, show only errors
                stream_url  # Output URL
            ]
            subprocess.run(command)

def main():
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/video_sets'))
    output_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), './output_frames'))
    os.makedirs(output_base_dir, exist_ok=True)

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
            time.sleep(1) 

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()