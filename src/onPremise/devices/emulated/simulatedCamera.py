import os
import asyncio
import uuid
import aiohttp
import logging
import platform
import random
from multiprocessing import Process, Event

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    def _detect_gpu(self):
        """Detect available GPU for encoding"""
        try:
            if self.os_type == "darwin":
                import subprocess
                result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], 
                                     capture_output=True, text=True)
                return "apple_silicon" if "Apple" in result.stdout else None
            elif self.os_type in ["linux", "windows"]:
                import subprocess
                result = subprocess.run(["nvidia-smi"], 
                                     capture_output=True, 
                                     shell=(self.os_type == "windows"))
                return "nvidia" if result.returncode == 0 else None
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
            return None

    async def discover_edge_server(self):
        """Discover edge server - using dummy URL for simulation"""
        self.edge_server_url = "http://localhost:8080"  # Replace with your actual edge server discovery logic
        return True

    def choose_videos(self, base_path="data/videos/video_sets", num_sets=3, num_videos=3):
        """Pick random sets and videos."""
        import os
        all_sets = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        chosen_sets = random.sample(all_sets, min(num_sets, len(all_sets)))
        selected_files = []
        for s in chosen_sets:
            set_path = os.path.join(base_path, s)
            videos_in_set = [f for f in os.listdir(set_path) if f.endswith(".avi")]
            chosen_videos = random.sample(videos_in_set, min(num_videos, len(videos_in_set)))
            for v in chosen_videos:
                selected_files.append(os.path.join(set_path, v))
        return selected_files

    def get_ffmpeg_input_args(self, video_path):
        """Get optimized FFmpeg input arguments for MP4 files"""
        return [
            "-re",  # Read input at native frame rate
            "-i", video_path,
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-strict", "experimental",
            "-fps_mode", "vfr"
        ]

    def get_ffmpeg_output_args(self, rtsp_url):
        """Get optimized FFmpeg output arguments with GPU support"""
        quality_args = [
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-profile:v", "baseline",
            "-x264-params", "keyint=30:min-keyint=30:scenecut=0:force-cfr=1",
            "-bufsize", "1M",
            "-maxrate", "2M",
            "-g", "30"
        ]

        if self.gpu_vendor == "nvidia":
            encoder_args = [
                "-c:v", "h264_nvenc",
                "-rc", "cbr_ld_hq",
                "-zerolatency", "1",
                "-gpu", "0"
            ]
        elif self.gpu_vendor == "apple_silicon":
            encoder_args = [
                "-c:v", "h264_videotoolbox",
                "-allow_sw", "1",
                "-realtime", "1"
            ]
        else:
            encoder_args = ["-c:v", "libx264", "-threads", "4"]

        output_args = [
            "-pix_fmt", "yuv420p",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            "-muxdelay", "0.1"
        ]

        # Add downscale filter
        scale_filter = ["-vf", "scale=320:240"]
        return encoder_args + quality_args + scale_filter + output_args + [rtsp_url]

    async def start_streaming(self, video_path=None, stop_event=None):
        """Start streaming multiple random videos with downscale."""
        if not self.edge_server_url:
            logger.error("No edge server URL available")
            return

        if not video_path:
            videos_to_stream = self.choose_videos()
        else:
            videos_to_stream = [video_path]

        while not (stop_event and stop_event.is_set()):
            for vid in videos_to_stream:
                if stop_event and stop_event.is_set():
                    break
                    
                if not os.path.exists(vid):
                    logger.error(f"Video file not found: {vid}")
                    continue

                try:
                    logger.info(f"Starting video stream: {vid}")
                    ffmpeg_command = [
                        "ffmpeg",
                        *self.get_ffmpeg_input_args(vid),
                        *self.get_ffmpeg_output_args(self.rtsp_url)
                    ]

                    process = await asyncio.create_subprocess_exec(
                        *ffmpeg_command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
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

                    # Wait for process to complete
                    await process.wait()
                    
                except asyncio.CancelledError:
                    logger.info("Streaming cancelled")
                    if process:
                        process.terminate()
                except Exception as e:
                    logger.error(f"Streaming error: {str(e)}")
                finally:
                    if process:
                        try:
                            process.terminate()
                            await asyncio.wait_for(process.wait(), timeout=5.0)
                        except asyncio.TimeoutError:
                            process.kill()
                    await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

    async def register_with_edge(self):
        """Register camera with edge server"""
        if not self.edge_server_url:
            logger.error("No edge server URL available")
            return False

        from urllib.parse import urlparse

        parsed_url = urlparse(self.edge_server_url)
        server_ip = parsed_url.hostname
        rtsp_port = 8555

        self.rtsp_url = f"rtsp://{server_ip}:{rtsp_port}/{self.camera_id}"
        logger.info(f"RTSP URL: {self.rtsp_url}")

        try:
            registration_data = {
                "camera_id": self.camera_id,
                "name": f"SimulatedCam {self.camera_id[:8]}",
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
                        await response.json()
                        logger.info(f"Camera {self.camera_id} registered successfully")
                        return True
                    else:
                        logger.error(f"Registration failed with status {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return False

    async def stop(self):
        """Stop the camera"""
        self.is_running = False
        logger.info("Simulated camera stopped")

async def main():
    camera = SimulatedCamera()
    try:
        if not await camera.discover_edge_server():
            logger.error("No edge server found")
            return

        if await camera.register_with_edge():
            await camera.start_streaming()
    except KeyboardInterrupt:
        logger.info("Shutting down camera...")
    except Exception as e:
        logger.error(f"Camera error: {e}")
    finally:
        await camera.stop()

if __name__ == "__main__":
    asyncio.run(main())
