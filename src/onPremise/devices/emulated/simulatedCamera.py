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
            if self.os_type == "darwin":
                import subprocess

                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                )
                if "Apple" in result.stdout:
                    return "apple_silicon"
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
                "-protocol_whitelist", "pipe,file",  # Allow pipe protocol
                "-f", "concat",
                "-re",
                "-stream_loop", "-1",
                "-i", "pipe:0",  # Read from stdin
                "-filter:v",
                f"setpts=PTS/{self.frame_rate}/TB",
            ], concat_content

    def get_ffmpeg_output_args(self, rtsp_url: str) -> List[str]:
        """Get optimized FFmpeg output arguments with GPU support."""
        scaling_args = [
            "-vf",
            f"scale={self.frame_width}:{self.frame_height}",
            "-r",
            f"{self.frame_rate}",
        ]

        quality_args = [
            "-profile:v",
            "baseline",
            "-bufsize",
            "1M",
            "-maxrate",
            "2M",
            "-g",
            "30",
        ]

        if self.gpu_vendor == "nvidia":
            encoder_args = [
                "-c:v",
                "h264_nvenc",
                "-preset",
                "fast",
                "-b:v",
                "2M",
                "-maxrate",
                "4M",
                "-bufsize",
                "4M",
            ]
        elif self.gpu_vendor == "apple_silicon":
            encoder_args = [
                "-c:v",
                "-tune",
                "zerolatency",
                "h264_videotoolbox",
                "-preset",
                "ultrafast",
                "-allow_sw",
                "1",
                "-realtime",
                "1",
            ]
        else:
            encoder_args = [
                "-c:v", "libx264",
                "-preset", "ultrafast", 
                "-threads", "4"
                "-tune",
                "zerolatency",
                "-x264-params",
                "keyint=30:min-keyint=30:scenecut=0:force-cfr=1",
                ]

        output_args = [
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rtsp",
            "-rtsp_transport",
            "tcp",
            rtsp_url,
        ]

        return scaling_args + encoder_args + quality_args + output_args

    async def start_streaming(
        self,
        video_infos: List[VideoInfo],
        stop_event: Optional[asyncio.Event] = None
    ):
        """Start streaming videos using in-memory concat list."""
        if not video_infos:
            logger.error("No video files provided to stream.")
            return

        # Get paths from VideoInfo objects
        paths_to_stream = [info.full_path for info in video_infos]

        # Verify all paths exist
        for path in paths_to_stream:
            if not Path(path).exists():
                logger.error(f"Video file not found: {path}")
                return

        try:
            logger.info(f"Starting video stream for camera {self.camera_id}")
            input_args, concat_content = self.get_ffmpeg_input_args(paths_to_stream)
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
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Write concat content to stdin
            if process.stdin and concat_content:
                try:
                    process.stdin.write(concat_content.encode())
                    await process.stdin.drain()
                    process.stdin.close()
                except Exception as e:
                    logger.error(f"Error writing to FFmpeg stdin: {e}")

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
                        logger.info(
                            f"Camera {camera_index} streaming video {video_info.video_name}"
                        )
                        await camera.start_streaming([video_info], stop_event)

                    logger.info(
                        f"Camera {camera_index} completed one full sequence, starting over"
                    )
    except Exception as e:
        logger.error(f"Error running camera {camera.camera_id}: {e}")

def get_data_directory():
    # Go up four levels from this file to reach TheObservatory, then down to "data"
    data_dir = Path(__file__).resolve().parents[4] / "data"
    return data_dir

async def main():
    parser = argparse.ArgumentParser(description="Run multiple simulated cameras")
    parser.add_argument(
        "--streams", type=int, default=6, help="Number of video streams to create"
    )
    args = parser.parse_args()

    stop_event = asyncio.Event()
    try:
        data_dir = get_data_directory()
        base_path = data_dir / "video_sets"
        video_paths = [[] for _ in range(args.streams)]
        
        logger.info(f"Looking for videos in: {base_path}")
        
        for set_num in range(1, 12):  # Sets 1 to 11
            set_dir = base_path / f"set_{set_num}"
            logger.info(f"Checking set directory: {set_dir}")
            
            for cam_num in range(args.streams):
                # Create full path for the video file
                video_file = f"video{set_num}_{cam_num + 1}.avi"
                video_path = set_dir / video_file
                
                logger.info(f"Checking for video: {video_path}")
                
                # Special handling for camera 6 (index 5)
                if cam_num == 5:  # Camera 6
                    if set_num >= 5:  # Only sets 5-11 for camera 6
                        if video_path.exists():
                            logger.info(f"Found video for camera 6: {video_path}")
                            video_paths[cam_num].append(str(video_path.resolve()))
                else:  # Other cameras (1-5)
                    if video_path.exists():
                        logger.info(f"Found video: {video_path}")
                        video_paths[cam_num].append(str(video_path.resolve()))

        # Log the found videos for each camera
        for cam_num, paths in enumerate(video_paths):
            logger.info(f"Camera {cam_num + 1} has {len(paths)} videos: {paths}")

        # Filter out empty camera paths
        selected_videos = [paths for paths in video_paths if paths]

        if not selected_videos:
            logger.error("No valid videos found in any of the sets")
            return
        # Convert strings to VideoInfo objects (single selection)
        selected_videos = []
        for cam_num, paths in enumerate(video_paths):
            if paths:  # if there are videos for this camera
                video_infos = []
                for path in paths:
                    path_obj = Path(path)
                    video_infos.append(VideoInfo(
                        set_name=path_obj.parent.name,
                        video_name=path_obj.name,
                        full_path=str(path_obj)
                    ))
                selected_videos.append(video_infos)
                logger.info(f"Camera {cam_num + 1} prepared with {len(video_infos)} videos")

        if not selected_videos:
            logger.error("No valid videos found in any of the sets")
            return

        # Create and run camera instances
        camera_tasks = []
        for i, video_infos in enumerate(selected_videos):
            logger.info(f"Starting camera {i} with {len(video_infos)} videos")
            task = asyncio.create_task(run_camera(video_infos, stop_event, i))
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
        # Log the full traceback for debugging
        import traceback
        logger.error(traceback.format_exc())
    finally:
        stop_event.set()

if __name__ == "__main__":
    asyncio.run(main())