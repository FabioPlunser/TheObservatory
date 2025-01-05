import os
import asyncio
from multiprocessing import Process, Event
from camera import Camera
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def start_simulated_camera(video_paths, stop_event):
    camera = Camera()
    camera.frame_rate = 1
    
    if not await camera.discover_edge_server():
        logger.error("Failed to discover edge server")
        return

    if not await camera.register_with_edge():
        logger.error("Failed to register with edge server")
        return
    try:
        await camera.start_streaming(video_paths=video_paths, stop_event=stop_event)
    finally:
        await camera.stop()

def run_camera_in_process(video_paths, stop_event):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_simulated_camera(video_paths, stop_event))

def main():
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data/video_sets'))
    if not os.path.exists(dataset_dir):
        logger.error("Dataset not found")
        return

    stop_events = []
    processes = []
    
    for i in range(1, 7):  # Assuming 6 different cameras
        video_paths = []
        for set_num in range(1, 12):  # Assuming 11 sets
            if i == 6 and set_num < 5:  # Skip sets 1 to 4 for the 6th camera
                continue
            video_path = os.path.join(dataset_dir, f'set_{set_num}', f'video{set_num}_{i}.avi')
            if not os.path.exists(video_path):
                break
            video_paths.append(video_path)
        stop_event = Event()
        stop_events.append(stop_event)
        process = Process(target=run_camera_in_process, args=(video_paths, stop_event))
        processes.append(process)
        process.start()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        logger.info("Stopping all cameras...")
        for stop_event in stop_events:
            stop_event.set()
        for process in processes:
            process.join()

if __name__ == "__main__":
    main()