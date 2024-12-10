import os
import requests
import cv2
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from camera import Camera

def send_frame(camera, jpeg, session):
    try:
        response = session.post(
            f"{camera.edge_server_url}/frame",
            files={'frame': ('frame.jpg', jpeg.tobytes(), 'image/jpeg')},
            data={'camera_id': camera.camera_id},
            timeout=5  
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error sending frame: {e}")

def start_simulated_camera(video_path, session):
    print(f"Attempting to open video: {video_path}")
    camera = Camera(video_path=video_path)
    if not camera.start():
        return

    while True:
        ret, frame = camera.cap.read()
        if not ret:
            print(f"Failed to read frame from video: {video_path}")
            break

        _, jpeg = cv2.imencode('.jpg', frame)

        send_frame(camera, jpeg, session)

    camera.stop()

def main():
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/video_sets'))

    session = requests.Session()
    adapter = HTTPAdapter(max_retries=2)
    session.mount('http://', adapter)

    with ThreadPoolExecutor(max_workers=6) as executor:
        camera_count = 0
        for set_num in range(1, 12):  # Assuming 11 sets
            for i in range(1, 7):  # Assuming 6 different cameras
                if camera_count >= 6:
                    break
                if i == 6 and set_num < 5:  # Skip sets 1 to 4 for the 6th camera
                    continue
                video_path = os.path.join(dataset_dir, f'set_{set_num}', f'video{set_num}_{i}.avi')
                if os.path.exists(video_path):
                    print(f"Starting simulated camera {camera_count + 1} with video: {video_path}")
                    executor.submit(start_simulated_camera, video_path, session)
                    camera_count += 1
                    if camera_count >= 6:
                        break
            if camera_count >= 6:
                break

if __name__ == "__main__":
    main()