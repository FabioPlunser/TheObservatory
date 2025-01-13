import os
import asyncio
import uuid
import aiohttp
import logging
import platform
import random
import argparse
from edge_server_discover import EdgeServerDiscovery
from typing import List, Optional
from dataclasses import dataclass
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
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
        self.frame_rate = 30
        self.frame_width = 640
        self.frame_height = 480
        self.os_type = platform.system().lower()
        self.gpu_vendor = self._detect_gpu()
        self.discovery = EdgeServerDiscovery()

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
    def get_available_video_sets(
        base_path: str = "data/videos/video_sets",
    ) -> List[str]:
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

    @staticmethod
    def choose_random_videos(
        base_path: str = "data/videos/video_sets", num_videos: int = 1
    ) -> List[VideoInfo]:
        """Choose random videos from available sets"""
        video_sets = SimulatedCamera.get_available_video_sets(base_path)
        if not video_sets:
            logger.error("No video sets found")
            return []

        selected_videos = []
        while len(selected_videos) < num_videos and video_sets:
            set_name = random.choice(video_sets)
            set_path = os.path.join(base_path, set_name)
            videos = SimulatedCamera.get_videos_in_set(set_path)

            if videos:
                video_name = random.choice(videos)
                full_path = os.path.join(set_path, video_name)
                selected_videos.append(VideoInfo(set_name, video_name, full_path))
            else:
                video_sets.remove(set_name)

        return selected_videos

    def get_ffmpeg_input_args(self, video_path: str) -> List[str]:
        """Get optimized FFmpeg input arguments"""
        return [
            "-re",
            "-stream_loop",
            "-1",  # Loop the video indefinitely
            "-i",
            video_path,
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-strict",
            "experimental",
            "-fps_mode",
            "vfr",
        ]

    def get_ffmpeg_output_args(self, rtsp_url: str) -> List[str]:
        """Get optimized FFmpeg output arguments with improved compatibility"""
        # Try to detect NVIDIA GPU capabilities
        try:
            if self.gpu_vendor == "nvidia":
                import subprocess

                result = subprocess.run(
                    ["ffmpeg", "-encoders"], capture_output=True, text=True
                )
                if "h264_nvenc" in result.stdout:
                    logger.info(
                        f"Using NVIDIA GPU encoding for camera {self.camera_id}"
                    )
                    return [
                        "-c:v",
                        "h264_nvenc",
                        "-preset",
                        "p1",  # Use P1 preset instead of ultrafast
                        "-tune",
                        "ll",  # Low latency tune
                        "-b:v",
                        "2M",  # Target bitrate
                        "-maxrate",
                        "4M",
                        "-bufsize",
                        "8M",
                        "-g",
                        "30",
                        "-f",
                        "rtsp",
                        "-rtsp_transport",
                        "tcp",
                        rtsp_url,
                    ]
        except Exception as e:
            logger.warning(f"NVIDIA GPU encoding not available: {e}")

        # Fallback to CPU encoding with optimized settings
        logger.info(f"Using CPU encoding for camera {self.camera_id}")
        return [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",  # Use veryfast instead of ultrafast for better stability
            "-tune",
            "zerolatency",
            "-profile:v",
            "baseline",
            "-x264-params",
            "nal-hrd=cbr:force-cfr=1",
            "-b:v",
            "1M",
            "-maxrate",
            "2M",
            "-bufsize",
            "2M",
            "-g",
            "30",
            "-f",
            "rtsp",
            "-rtsp_transport",
            "tcp",
            rtsp_url,
        ]

        output_args = [
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rtsp",
            "-rtsp_transport",
            "tcp",
            "-muxdelay",
            "0.1",
        ]

        # Add downscale filter
        scale_filter = ["-vf", f"scale={self.frame_width}:{self.frame_height}"]
        return encoder_args + quality_args + scale_filter + output_args + [rtsp_url]

    async def start_streaming(
        self, video_path: str, stop_event: Optional[asyncio.Event] = None
    ):
        """Start streaming a single video with improved error handling"""
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return

        try:
            logger.info(
                f"Starting video stream for camera {self.camera_id}: {video_path}"
            )
            # Add input format detection
            probe_command = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate",
                "-of",
                "json",
                video_path,
            ]

            try:
                process = await asyncio.create_subprocess_exec(
                    *probe_command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await process.communicate()
                if process.returncode == 0:
                    logger.info(
                        f"Successfully probed video file for camera {self.camera_id}"
                    )

            except Exception as e:
                logger.warning(
                    f"Failed to probe video file for camera {self.camera_id}: {e}"
                )

            ffmpeg_command = [
                "ffmpeg",
                "-hide_banner",  # Reduce logging noise
                *self.get_ffmpeg_input_args(video_path),
                *self.get_ffmpeg_output_args(self.rtsp_url),
            ]

            # Log the full command for debugging
            logger.info(
                f"FFmpeg command for camera {self.camera_id}: {' '.join(ffmpeg_command)}"
            )

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
                    log_line = line.decode("utf-8").strip()
                    if level == logging.ERROR and "Error" in log_line:
                        logger.error(f"FFmpeg error: {log_line}")
                    elif level == logging.INFO and (
                        "fps" in log_line or "bitrate" in log_line
                    ):
                        logger.info(f"FFmpeg status: {log_line}")

            stdout_task = asyncio.create_task(log_output(process.stdout, logging.INFO))
            stderr_task = asyncio.create_task(log_output(process.stderr, logging.ERROR))

            try:
                if stop_event:
                    while not stop_event.is_set():
                        if process.returncode is not None:
                            break
                        await asyncio.sleep(1)
                else:
                    await process.wait()
            except asyncio.CancelledError:
                logger.info(f"Streaming cancelled for camera {self.camera_id}")
            finally:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    process.kill()
                await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

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


async def run_camera(video_info: VideoInfo, stop_event: asyncio.Event):
    """Run a single camera instance"""
    camera = SimulatedCamera()
    try:
        if await camera.discover_edge_server():
            if await camera.register_with_edge():
                await camera.start_streaming(video_info.full_path, stop_event)
    except Exception as e:
        logger.error(f"Error running camera: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Run multiple simulated cameras")
    parser.add_argument(
        "--streams", type=int, default=3, help="Number of video streams to create"
    )
    args = parser.parse_args()

    stop_event = asyncio.Event()
    try:
        # Select random videos
        selected_videos = SimulatedCamera.choose_random_videos(num_videos=args.streams)
        if not selected_videos:
            logger.error("No videos available to stream")
            return

        # Create and run multiple camera instances
        camera_tasks = []
        for video_info in selected_videos:
            logger.info(
                f"Starting camera with video: {video_info.video_name} from set {video_info.set_name}"
            )
            task = asyncio.create_task(run_camera(video_info, stop_event))
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
