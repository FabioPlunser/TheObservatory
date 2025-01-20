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
from pathlib import Path

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
        self.frame_rate = 15 
        self.frame_width = 640  
        self.frame_height = 480  
        self.os_type = platform.system().lower()
        self.gpu_vendor = self._detect_gpu()
        self.discovery = EdgeServerDiscovery()
        self.executor = ThreadPoolExecutor(max_workers=6)  # Limit to 6 threads

    def _detect_gpu(self):
        """Detect available GPU for encoding."""
        try:
            import subprocess
            if self.os_type == "darwin":
                result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], 
                                    capture_output=True, text=True)
                return "apple_silicon" if "Apple" in result.stdout else None
                
            if self.os_type in ["linux", "windows"]:
                result = subprocess.run(["nvidia-smi"], 
                                    capture_output=True, 
                                    shell=(self.os_type == "windows"))
                return "nvidia" if result.returncode == 0 else None
            return None
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
        return [
            d
            for d in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, d))
        ]

    @staticmethod
    def get_videos_in_set(set_path: str) -> List[str]:
        """Get all videos in a set"""
        if not os.path.exists(set_path):
            logger.error(f"Video set directory not found: {set_path}")
            return []
        return [f for f in os.listdir(set_path) if f.endswith(".avi")]

    def get_ffmpeg_input_args(self, video_paths: List[str]) -> tuple[List[str], str]:
            """Use FFmpeg's pipe protocol to read the concat list from memory.
            Returns tuple of (ffmpeg_args, concat_content)"""
            if not video_paths:
                # Fallback to test source if no videos provided
                return [
                    "-f",
                    "lavfi",
                    "-i",
                    f"testsrc=size={self.frame_width}x{self.frame_height}:rate={self.frame_rate}",
                    "-pix_fmt",
                    "yuv420p",
                ], ""
            
            concat_content = "".join(
                f"file '{str(Path(vp).resolve()).replace('\\', '/')}'\n"
                for vp in video_paths
            )
            
            return [
                "-fflags", "nobuffer",
                "-flags", "low_delay",
                "-safe", "0",
                "-protocol_whitelist", "pipe,file",
                "-f", "concat",
                "-re",
                "-stream_loop", "-1",
                "-i", "pipe:0",  # Read from stdin
                "-filter:v",
                f"setpts=PTS/{self.frame_rate}/TB",
            ], concat_content

    def get_ffmpeg_output_args(self, rtsp_url: str) -> List[str]:
        """Get optimized FFmpeg output arguments with GPU support."""
        encoder_configs = {
            "nvidia": [
                "-c:v", "h264_nvenc",
                "-preset", "fast",
                "-b:v", "2M",
                "-maxrate", "4M",
                "-bufsize", "4M",
            ],
            "apple_silicon": [
                "-c:v", "h264_videotoolbox",
                "-preset", "ultrafast",
                "-tune", "zerolatency",
                "-allow_sw", "1",
                "-realtime", "1",
            ],
            "default": [
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-threads", "4",
                "-tune", "zerolatency",
                "-x264-params", "keyint=30:min-keyint=30:scenecut=0:force-cfr=1",
            ]
        }

        base_args = [
            "-vf", f"scale={self.frame_width}:{self.frame_height}",
            "-r", f"{self.frame_rate}",
            "-profile:v", "baseline",
            "-bufsize", "1M",
            "-maxrate", "2M",
            "-g", "30",
            "-pix_fmt", "yuv420p",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
        ]

        encoder_args = encoder_configs.get(self.gpu_vendor, encoder_configs["default"])
        return base_args + encoder_args + [rtsp_url]

    async def _handle_ffmpeg_process(self, process, concat_content: str, stop_event: Optional[asyncio.Event]):
        """Handle FFmpeg process IO and monitoring."""
        async def log_output(stream, level):
            async for line in stream:
                logger.log(level, line.decode("utf-8").strip())

        # Write concat content if available
        if process.stdin and concat_content:
            try:
                process.stdin.write(concat_content.encode())
                await process.stdin.drain()
                process.stdin.close()
            except Exception as e:
                logger.error(f"Error writing to FFmpeg stdin: {e}")

        # Start output logging tasks
        log_tasks = [
            asyncio.create_task(log_output(process.stdout, logging.INFO)),
            asyncio.create_task(log_output(process.stderr, logging.ERROR))
        ]

        try:
            if stop_event:
                while not stop_event.is_set() and process.returncode is None:
                    await asyncio.sleep(1)
            else:
                await process.wait()
        finally:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    process.kill()
            
            await asyncio.gather(*log_tasks)

    async def start_streaming(self, video_infos: List[VideoInfo], stop_event: Optional[asyncio.Event] = None):
        """Start streaming videos using in-memory concat list."""
        if not video_infos or not all(Path(info.full_path).exists() for info in video_infos):
            logger.error("Invalid or missing video files.")
            return

        try:
            paths_to_stream = [info.full_path for info in video_infos]
            input_args, concat_content = self.get_ffmpeg_input_args(paths_to_stream)
            output_args = self.get_ffmpeg_output_args(self.rtsp_url)

            process = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-hide_banner",
                *input_args,
                *output_args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            await self._handle_ffmpeg_process(process, concat_content, stop_event)

        except Exception as e:
            logger.error(f"Streaming error for camera {self.camera_id}: {str(e)}")

    async def register_with_edge(self) -> bool:
        """Register camera with edge server"""
        if not self.edge_server_url:
            logger.error("No edge server URL available")
            return False

        parsed_url = urlparse(self.edge_server_url)
        self.rtsp_url = f"rtsp://{parsed_url.hostname}:8555/{self.camera_id}"
        logger.info(f"RTSP URL for camera {self.camera_id}: {self.rtsp_url}")

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

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.edge_server_url}/register/camera",
                    json=registration_data,
                    timeout=5,
                ) as response:
                    success = response.status == 200
                    log_level = logging.INFO if success else logging.ERROR
                    logger.log(
                        log_level,
                        f"Camera {self.camera_id} registration {'successful' if success else f'failed with status {response.status}'}"
                    )
                    return success

        except Exception as e:
            logger.error(f"Registration failed for camera {self.camera_id}: {e}")
            return False


async def run_camera(video_infos: List[VideoInfo], stop_event: asyncio.Event, camera_index: int):
    """Run a single camera instance"""
    if not video_infos:
        logger.error(f"No videos available for camera {camera_index}")
        return

    camera = SimulatedCamera()
    try:
        if not await camera.discover_edge_server() or not await camera.register_with_edge():
            return

        while not stop_event.is_set():
            for video_info in video_infos:
                if stop_event.is_set():
                    break
                    
                logger.info(f"Camera {camera_index} streaming {video_info.video_name}")
                await camera.start_streaming([video_info], stop_event)

            if not stop_event.is_set():
                logger.info(f"Camera {camera_index} completed sequence, restarting")
                
    except Exception as e:
        logger.error(f"Error running camera {camera.camera_id}: {e}")

def collect_video_paths(base_path: Path, num_streams: int) -> List[List[VideoInfo]]:
    """Collect video paths for each camera."""
    video_paths = [[] for _ in range(num_streams)]
    
    for set_num in range(1, 12):
        set_dir = base_path / f"set_{set_num}"
        
        for cam_num in range(num_streams):
            video_file = f"video{set_num}_{cam_num + 1}.avi"
            video_path = set_dir / video_file
            
            # Skip if video doesn't exist or if it's camera 6 with set < 5
            if not video_path.exists() or (cam_num == 5 and set_num < 5):
                continue
                
            video_paths[cam_num].append(
                VideoInfo(
                    set_name=set_dir.name,
                    video_name=video_file,
                    full_path=str(video_path.resolve())
                )
            )
    
    return [paths for paths in video_paths if paths]

def get_data_directory():
    # Go up four levels from this file to reach TheObservatory, then down to "data"
    data_dir = Path(__file__).resolve().parents[4] / "data"
    return data_dir

async def main():
    parser = argparse.ArgumentParser(description="Run multiple simulated cameras")
    parser.add_argument("--streams", type=int, default=6, 
                       help="Number of video streams to create")
    args = parser.parse_args()

    stop_event = asyncio.Event()
    try:
        data_dir = get_data_directory()
        base_path = data_dir / "video_sets"
        logger.info(f"Looking for videos in: {base_path}")
        
        selected_videos = collect_video_paths(base_path, args.streams)
        if not selected_videos:
            logger.error("No valid videos found in any of the sets")
            return

        # Create and run camera instances
        camera_tasks = [
            asyncio.create_task(run_camera(videos, stop_event, i))
            for i, videos in enumerate(selected_videos)
        ]

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