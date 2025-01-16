import os
import asyncio
import uuid
import aiohttp
import logging
import platform
import argparse
from edge_server_discover import EdgeServerDiscovery
from typing import List, Optional
from dataclasses import dataclass
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor


logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    set_name: str
    video_name: str
    full_path: str


class SimulatedCamera:
    def __init__(self):
        self.camera_id = str(uuid.uuid4())
        self.edge_server_url = None
        self.rtsp_url = None
        self.is_running = False
        self.frame_rate = 1
        self.frame_width = 640*2
        self.frame_height = 480*2
        self.os_type = platform.system().lower()
        self.gpu_vendor = self._detect_gpu()
        self.discovery = EdgeServerDiscovery()
        self.executor = ThreadPoolExecutor(max_workers=6)  # Limit to 6 threads

    def _detect_gpu(self):
        """Detect available GPU for encoding"""
        try:
            if self.os_type == "darwin":
                import subprocess
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                )
                return "apple_silicon" if "Apple" in result.stdout else None
            elif self.os_type in ["linux", "windows"]:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi"],
                    capture_output=True,
                    shell=(self.os_type == "windows"),
                )
                return "nvidia" if result.returncode == 0 else None
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
            return None

    async def discover_edge_server(self):
        """Discover edge server using EdgeServerDiscovery"""
        logger.info("Starting edge server discovery")
        self.edge_server_url = await self.discovery.discover_edge_server()
        if not self.edge_server_url:
            logger.error("Failed to discover edge server")
            return False
        logger.info(f"Discovered edge server at {self.edge_server_url}")
        return True

    @staticmethod
    def get_available_video_sets(base_path: str = "data/video_sets") -> List[str]:
        """Get all available video sets"""
        if not os.path.exists(base_path):
            logger.error(f"Video sets directory not found: {base_path}")
            return []
        return [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    @staticmethod
    def get_videos_in_set(set_path: str) -> List[str]:
        """Get all videos in a set"""
        if not os.path.exists(set_path):
            logger.error(f"Video set directory not found: {set_path}")
            return []
        return [f for f in os.listdir(set_path) if f.endswith(".avi")]

    def get_ffmpeg_input_args(self, video_path: str) -> List[str]:
        """Get optimized FFmpeg input arguments"""
        if not video_path:
            # Generate test pattern if no video file
            return [
                "-f", "lavfi",
                "-i", "testsrc=size=640x480:rate=1",  
                "-pix_fmt", "yuv420p",
            ]
        else:
            return [
                "-re",
                "-stream_loop", "-1",
                "-i", video_path,
                "-r", "1",  
                "-vf", "fps=1",  
            ]

    def get_ffmpeg_output_args(self, rtsp_url: str) -> List[str]:
        """Get optimized FFmpeg output arguments"""
        common_args = [
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            "-r", "1",  
        ]

        # Try to detect NVIDIA GPU capabilities
        if self.gpu_vendor == "nvidia":
            try:
                encoder_args = [
                    "-c:v", "h264_nvenc",
                    "-preset", "fast",  # Use a more efficient preset
                    "-tune", "ll",
                    "-zerolatency", "1",
                    "-b:v", "2M",  # Reduce bitrate
                    "-maxrate", "2M",
                    "-bufsize", "4M",
                    "-g", "30",
                ]
            except Exception:
                encoder_args = self._get_cpu_encoder_args()
        elif self.gpu_vendor == "apple_silicon":
            encoder_args = [
                "-c:v", "h264_videotoolbox",
                "-allow_sw", "1",
                "-realtime", "1",
            ]
        else:
            encoder_args = self._get_cpu_encoder_args()

        return encoder_args + common_args + [rtsp_url]

    def _get_cpu_encoder_args(self) -> List[str]:
        """Get CPU encoder arguments"""
        return [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-tune", "ll",
            "-profile:v", "baseline",
            "-x264-params", "nal-hrd=cbr:force-cfr=1",
            "-b:v", "2M",
            "-maxrate", "2M",
            "-bufsize", "2M",
            "-g", "30",
        ]

    async def start_streaming(self, video_path: str = "", stop_event: Optional[asyncio.Event] = None):
        """Start streaming with improved error handling"""
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return
            
        try:
            logger.info(f"Starting video stream for camera {self.camera_id} with video {video_path}")
            input_args = self.get_ffmpeg_input_args(video_path)
            output_args = self.get_ffmpeg_output_args(self.rtsp_url)

            ffmpeg_command = [
                "ffmpeg",
                "-hide_banner",
                *input_args,
                *output_args,
            ]

            logger.info(f"FFmpeg command: {' '.join(ffmpeg_command)}")
            process = await asyncio.create_subprocess_exec(
                *ffmpeg_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            async def log_output(stream, level):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    decoded_line = line.decode("utf-8").strip()
                    logger.log(level, decoded_line)

            stdout_task = asyncio.create_task(log_output(process.stdout, logging.INFO))
            stderr_task = asyncio.create_task(log_output(process.stderr, logging.ERROR))

            try:
                if stop_event:
                    while not stop_event.is_set():
                        if process.returncode is not None:
                            logger.error(f"FFmpeg process exited with code {process.returncode}")
                            break
                        await asyncio.sleep(1)
                    process.terminate()
                else:
                    await process.wait()
            finally:
                if process.returncode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        process.kill()

            await asyncio.gather(stdout_task, stderr_task)

        except Exception as e:
            logger.error(f"Streaming error for camera {self.camera_id}: {str(e)}")

    async def register_with_edge(self) -> bool:
        """Register camera with edge server"""
        if not self.edge_server_url:
            logger.error("No edge server URL available")
            return False

        parsed_url = urlparse(self.edge_server_url)
        server_ip = parsed_url.hostname
        rtsp_port = 8555

        self.rtsp_url = f"rtsp://{server_ip}:{rtsp_port}/{self.camera_id}"
        logger.info(f"RTSP URL for camera {self.camera_id}: {self.rtsp_url}")

        try:
            registration_data = {
                "camera_id": self.camera_id,
                "name": f"SimCam_{self.camera_id[:8]}",
                "rtsp_url": self.rtsp_url,
                "capabilities": {
                    "resolution": f"{self.frame_width}x{self.frame_height}",
                    "fps": self.frame_rate,
                    "night_vision": False,
                },
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.edge_server_url}/register/camera",
                    json=registration_data,
                    timeout=5,
                ) as response:
                    if response.status == 200:
                        logger.info(f"Camera {self.camera_id} registered successfully")
                        return True
                    else:
                        logger.error(
                            f"Registration failed for camera {self.camera_id} with status {response.status}"
                        )
                        return False

        except Exception as e:
            logger.error(f"Registration failed for camera {self.camera_id}: {e}")
            return False


async def run_camera(video_infos: List[VideoInfo], stop_event: asyncio.Event, camera_index: int):
    """Run a single camera instance"""
    camera = SimulatedCamera()
    try:
        if await camera.discover_edge_server():
            if await camera.register_with_edge():
                video_count = len(video_infos)
                if video_count == 0:
                    logger.error(f"No videos available for camera {camera_index}")
                    return

                while not stop_event.is_set():
                    # Play all videos in sequence
                    for video_info in video_infos:
                        if stop_event.is_set():
                            break
                        logger.info(f"Camera {camera_index} streaming video {video_info.video_name}")
                        await camera.start_streaming(video_info.full_path, stop_event)
                    
                    logger.info(f"Camera {camera_index} completed one full sequence, starting over")
                    # Loop will start over from the beginning after playing all videos
    except Exception as e:
        logger.error(f"Error running camera {camera.camera_id}: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Run multiple simulated cameras")
    parser.add_argument(
        "--streams", type=int, default=6, help="Number of video streams to create"
    )
    args = parser.parse_args()

    stop_event = asyncio.Event()
    try:
        # Initialize video paths for all cameras
        video_paths = [[] for _ in range(args.streams)]
        for set_num in range(1, 12):  # Sets 1 to 11
            for cam_num in range(args.streams):  # Use all cameras
                # Special handling for camera 6 (index 5)
                if cam_num == 5:  # Camera 6
                    if set_num >= 5:  # Only sets 5-11 for camera 6
                        video_paths[cam_num].append(f"set_{set_num}/video{set_num}_{cam_num + 1}.avi")
                else:  # Other cameras (1-5)
                    video_paths[cam_num].append(f"set_{set_num}/video{set_num}_{cam_num + 1}.avi")

        # Convert paths to VideoInfo objects and verify files exist
        selected_videos = []
        for cam_videos in video_paths:
            cam_video_info = []
            for video_path in cam_videos:
                set_name, video_name = video_path.split('/')
                full_path = os.path.join("data/video_sets", set_name, video_name)
                if os.path.exists(full_path):
                    cam_video_info.append(VideoInfo(set_name, video_name, full_path))
                else:
                    logger.warning(f"Video file not found: {full_path}")
            if cam_video_info:  # Only add if there are valid videos
                selected_videos.append(cam_video_info)

        if not selected_videos:
            logger.error("No valid videos available to stream")
            return

        # Create and run multiple camera instances
        camera_tasks = []
        for i, cam_videos in enumerate(selected_videos):
            logger.info(
                f"Starting camera {i} with {len(cam_videos)} videos: {[video.video_name for video in cam_videos]}"
            )
            task = asyncio.create_task(run_camera(cam_videos, stop_event, i))
            camera_tasks.append(task)

        # Wait for keyboard interrupt
        try:
            await asyncio.gather(*camera_tasks)
        except KeyboardInterrupt:
            logger.info("Shutting down cameras...")
            stop_event.set()
            await asyncio.gather(*camera_tasks, return_exceptions=True)

    except Exception as e:
        logger.error(f"Main error: {e}")
    finally:
        stop_event.set()


if __name__ == "__main__":
    asyncio.run(main())