import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from camera import Camera
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def start_simulated_camera(video_path):
    camera = Camera()
    
    if not await camera.discover_edge_server():
        print("Failed to discover edge server")
        return

    if not await camera.register_with_edge():
        print("Failed to register with edge server")
        return

    await camera.start_streaming(video_path=video_path)
    await camera.stop()

def run_camera_in_thread(video_path):
    asyncio.run(start_simulated_camera(video_path))

def main():
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data/video_sets'))

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
                    executor.submit(run_camera_in_thread, video_path)
                    camera_count += 1
                    if camera_count >= 6:
                        break
            if camera_count >= 6:
                break

if __name__ == "__main__":
    main()