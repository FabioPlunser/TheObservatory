import cv2
import uuid
import aiohttp
import logging
import asyncio
import websockets
import base64
import json
import platform
from datetime import datetime
from edge_server_discover import EdgeServerDiscovery

logging.basicConfig(
    level=logging.INFO,
    format="Camera: %(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("log.log")],
)

logger = logging.getLogger(__name__)


class Camera:
    def __init__(self):
        self.camera_id = str(uuid.uuid4())
        self.edge_server_url = None
        self.rtsp_url = None
        self.cap = None
        self.is_running = False
        self.discovery = EdgeServerDiscovery()

        # Default to a commonly supported configuration
        self.frame_rate = 30
        self.frame_width = 640
        self.frame_height = 480
        self.os_type = platform.system().lower()
        self.gpu_vendor = self._detect_gpu()

        logger.info(f"Detected operating system: {self.os_type}")
        logger.info(f"Detected GPU: {self.gpu_vendor}")

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
        """Discover edge server"""
        logger.info("Starting edge server discovery")
        self.edge_server_url = await self.discovery.discover_edge_server()
        if not self.edge_server_url:
            logger.error("Failed to discover edge server")
            return False
        logger.info(f"Successfully discovered edge server at {self.edge_server_url}")
        return True

    def get_ffmpeg_input_args(self):
        """Get optimized FFmpeg input arguments for webcam"""
        base_args = [
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-framerate",
            str(self.frame_rate),
            "-video_size",
            f"{self.frame_width}x{self.frame_height}",
        ]

        if self.os_type == "darwin":
            return base_args + [
                "-f",
                "avfoundation",
                "-capture_cursor",
                "0",
                "-capture_mouse_clicks",
                "0",
                "-pixel_format",
                "uyvy422",
                "-i",
                "0:none",
            ]
        elif self.os_type == "linux":
            return base_args + [
                "-f",
                "v4l2",
                "-input_format",
                "mjpeg",
                "-ts",
                "monotonic",
                "-i",
                "/dev/video0",
            ]
        else:  # Windows
            return base_args + [
                "-f",
                "dshow",
                "-rtbufsize",
                "100M",
                "-i",
                "video=Integrated Webcam",
            ]

    def get_ffmpeg_output_args(self, rtsp_url):
        """Get optimized FFmpeg output arguments with GPU support"""
        # Base quality settings
        quality_args = [
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-profile:v",
            "baseline",
            "-x264-params",
            "keyint=30:min-keyint=30:scenecut=0:force-cfr=1",
            "-bufsize",
            "1M",
            "-maxrate",
            "2M",
            "-g",
            "30",
        ]

        # GPU-specific encoder settings
        if self.gpu_vendor == "nvidia":
            encoder_args = [
                "-c:v",
                "h264_nvenc",
                "-rc",
                "cbr_ld_hq",
                "-zerolatency",
                "1",
                "-gpu",
                "0",
            ]
        elif self.gpu_vendor == "apple_silicon":
            encoder_args = [
                "-c:v",
                "h264_videotoolbox",
                "-allow_sw",
                "1",
                "-realtime",
                "1",
            ]
        else:
            encoder_args = ["-c:v", "libx264", "-threads", "4"]

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

        return encoder_args + quality_args + output_args + [rtsp_url]

    async def start_streaming(self, stop_event=None):
        """Start webcam streaming"""
        if not self.edge_server_url:
            logger.error("No edge server URL available")
            return

        try:
            logger.info(f"Starting webcam stream on {self.os_type}")

            ffmpeg_command = [
                "ffmpeg",
                *self.get_ffmpeg_input_args(),
                *self.get_ffmpeg_output_args(self.rtsp_url),
            ]

            process = await asyncio.create_subprocess_exec(
                *ffmpeg_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            logger.info(f"FFmpeg process started for webcam {self.camera_id}")

            async def log_output(stream, level):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    decoded_line = line.decode("utf-8").strip()
                    logger.log(level, decoded_line)

                    # Check for common webcam errors
                    if "Permission denied" in decoded_line:
                        if self.os_type == "darwin":
                            logger.error(
                                "Webcam access permission denied. Please check System Preferences -> Security & Privacy -> Camera"
                            )
                        elif self.os_type == "linux":
                            logger.error(
                                "Webcam access permission denied. Check user permissions for /dev/video0"
                            )
                        else:
                            logger.error("Webcam access permission denied")
                        process.terminate()
                    elif "Invalid data found" in decoded_line:
                        logger.error(
                            "Error accessing webcam. Please ensure no other applications are using it"
                        )
                        process.terminate()

            stdout_task = asyncio.create_task(log_output(process.stdout, logging.INFO))
            stderr_task = asyncio.create_task(log_output(process.stderr, logging.ERROR))

            try:
                if stop_event:
                    while not stop_event.is_set():
                        if process.returncode is not None:
                            logger.error(
                                f"FFmpeg process exited with code {process.returncode}"
                            )
                            break
                        await asyncio.sleep(1)
                    process.terminate()
                else:
                    await process.wait()
            except asyncio.CancelledError:
                logger.info("Streaming cancelled")
                process.terminate()
            finally:
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning(
                        "FFmpeg process didn't terminate gracefully, forcing..."
                    )
                    process.kill()

            await asyncio.gather(stdout_task, stderr_task)

        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            raise
        finally:
            self.is_running = False
            logger.info(f"Webcam streaming stopped for camera {self.camera_id}")

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
                "name": f"Webcam {self.camera_id[:8]}",
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
                        logger.info(f"Webcam {self.camera_id} registered successfully")
                        return True
                    else:
                        logger.error(
                            f"Registration failed with status {response.status}"
                        )
                        return False

        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return False

    async def stop(self):
        """Stop the camera"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None


async def main():
    camera = Camera()
    try:
        if not await camera.discover_edge_server():
            logger.error("No edge server found")
            return

        if await camera.register_with_edge():
            await camera.start_streaming()
    except KeyboardInterrupt:
        logger.info("Shutting down webcam...")
    except Exception as e:
        logger.error(f"Camera error: {e}")
    finally:
        await camera.stop()


if __name__ == "__main__":
    asyncio.run(main())
