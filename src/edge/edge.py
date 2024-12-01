import os
import cv2
import threading

FRAME_INTERVAL = 30  # Process every n-th frame set to 1 to process all frames
STREAM_URLS = [
    "rtmp://44.204.173.180:1935/live/stream1",
    "rtmp://44.204.173.180:1935/live/stream2",
    "rtmp://44.204.173.180:1935/live/stream3",
    "rtmp://44.204.173.180:1935/live/stream4",
    "rtmp://44.204.173.180:1935/live/stream5",
    "rtmp://44.204.173.180:1935/live/stream6",
    "rtmp://44.204.173.180:1935/live/stream7"
]

def process_video(source, frame_interval, output_dir):
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {source}.")
        return
    
    frame_count = 0
    prev_frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        # Only process frames at the specified interval
        if frame_count % frame_interval == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                diff_frame = cv2.absdiff(prev_frame, gray_frame)
                if diff_frame is not None:
                    output_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
                    cv2.imwrite(output_path, frame)

            prev_frame = gray_frame
        
        frame_count += 1

    cap.release()

def main():
    output_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), './output_frames'))
    os.makedirs(output_base_dir, exist_ok=True)

    threads = []

    for stream_url in STREAM_URLS:
        output_dir = os.path.join(output_base_dir, stream_url.split('/')[-1])
        os.makedirs(output_dir, exist_ok=True)
        thread = threading.Thread(target=process_video, args=(stream_url, FRAME_INTERVAL, output_dir))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()