import os
import requests
import cv2

FRAME_INTERVAL = 30 # Process every n-th frame
MEAN_DIFF_THRESHOLD = 4 # Threshold for mean difference between frames to detect a change
USER_LOCAL_DATA = True # Set this flag to True to use the local data, or False to download the video from the web, only works if USE_STREAM is False
USE_STREAM = True  # Set this flag to True to use the RTMP stream or webcam
STREAM_URL = 0  # Set to 0 for own webcam, or "rtmp://your_server_ip/live/stream" for RTMP stream

def download_video(url, filename):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
    else:
        print("Failed to download video")

def process_video(video_path, frame_interval, output_dir, use_stream=False):
    if use_stream:
        cap = cv2.VideoCapture(STREAM_URL)
    else:
        cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return
    
    frame_count = 0
    prev_frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process frames at the specified interval
        if frame_count % frame_interval == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_frame is not None:
                diff_frame = cv2.absdiff(prev_frame, gray_frame)
                mean_diff = cv2.mean(diff_frame)[0]
                if mean_diff > MEAN_DIFF_THRESHOLD:
                    print(f"Change detected at frame {frame_count} with mean difference {mean_diff}")
                    
                    frame_filename = os.path.join(output_dir, f"frame_{frame_count}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    print(f"Saved {frame_filename}")
            else:
                cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count}.jpg"), frame)
            
            prev_frame = gray_frame
        
        frame_count += 1
        yield frame 
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    base_url = "http://parataisa.at/observatory/"
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/video_sets'))

    output_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../output_frames'))
    os.makedirs(output_base_dir, exist_ok=True)

    if USE_STREAM:
        output_dir = os.path.join(output_base_dir, 'stream')
        os.makedirs(output_dir, exist_ok=True)
        for frame in process_video(None, FRAME_INTERVAL, output_dir, use_stream=True):
            #print("Frame was processed(video stream)")
            pass
    else:
        for set_dir in os.listdir(dataset_dir):
            set_path = os.path.join(dataset_dir, set_dir)
            if os.path.isdir(set_path):
                for video_file in os.listdir(set_path):
                    video_path = os.path.join(set_path, video_file)
                    if video_file.endswith(".avi"):
                        video_name = os.path.splitext(video_file)[0]
                        output_dir = os.path.join(output_base_dir, video_name)
                        os.makedirs(output_dir, exist_ok=True)
                        if USER_LOCAL_DATA:  # Process the video directly from the local directory
                            for frame in process_video(video_path, FRAME_INTERVAL, output_dir):
                                # print("Frame was processed(local video)")
                                pass
                        else:  # Download the video from the web
                            video_url = base_url + set_dir + "/" + video_file
                            video_filename = video_file
                            download_video(video_url, video_filename)
                            for frame in process_video(video_filename, FRAME_INTERVAL, output_dir):
                                # print("Frame was processed(downloaded video)")
                                pass

if __name__ == "__main__":
    main()