import os
import asyncio
import uuid
import aiohttp
import logging
import platform
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

        return encoder_args + quality_args + output_args + [rtsp_url]

    async def start_streaming(self, video_path, stop_event):
        """Start streaming from MP4 file"""
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return

        try:
            logger.info(f"Starting MP4 stream: {video_path}")
            
            ffmpeg_command = [
                "ffmpeg",
                *self.get_ffmpeg_input_args(video_path),
                *self.get_ffmpeg_output_args(self.rtsp_url)
            ]

            process = await asyncio.create_subprocess_exec(
                *ffmpeg_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            logger.info(f"FFmpeg process started for video: {video_path}")

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
                while not stop_event.is_set():
                    if process.returncode is not None: