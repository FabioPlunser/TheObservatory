import cv2
import threading
import signal

FRAME_INTERVAL = 1  # Process every n-th frame
CHANGE_THRESHOLD = 1 # Threshold for detecting changes in frames

STREAM_URLS = [
    "rtmp://44.204.173.180:1935/live/stream1",
    "rtmp://44.204.173.180:1935/live/stream2",
    "rtmp://44.204.173.180:1935/live/stream3",
    "rtmp://44.204.173.180:1935/live/stream4",
    "rtmp://44.204.173.180:1935/live/stream5",
    "rtmp://44.204.173.180:1935/live/stream6",
    "rtmp://44.204.173.180:1935/live/stream7"
]

stop_flag = threading.Event()

def signal_handler(sig, frame):
    print('Stopping...')
    stop_flag.set()

signal.signal(signal.SIGINT, signal_handler)

def process_video(source, frame_interval, window_name):
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {source}.")
        return
    
    frame_count = 0
    prev_frame = None
    while cap.isOpened() and not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        # Only process frames at the specified interval
        if frame_count % frame_interval == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                diff_frame = cv2.absdiff(prev_frame, gray_frame)
                mean_diff = cv2.mean(diff_frame)[0]
                if mean_diff > CHANGE_THRESHOLD:
                    resized_frame = cv2.resize(frame, (640, 480))  
                    cv2.imshow(window_name, resized_frame)
                    print(f"Processed frame {frame_count} with mean difference {mean_diff}")
                    # processing logic here

            prev_frame = gray_frame
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag.set()
            break
        
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

def main():
    threads = []

    for stream_url in STREAM_URLS:
        window_name = stream_url.split('/')[-1]
        thread = threading.Thread(target=process_video, args=(stream_url, FRAME_INTERVAL, window_name))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()