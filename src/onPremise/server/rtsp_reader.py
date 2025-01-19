import ffmpeg
import numpy as np
import multiprocessing as mp
import subprocess
import logging
import time
import queue

logger = logging.getLogger("RTSPReader")


class RTSPReader(mp.Process):
    def __init__(
        self,
        rtsp_url: str,
        output_queue: mp.Queue,
        width: int = 680,
        height: int = 480,
        fps: int = 30,
    ):
        super().__init__(daemon=True)
        self.rtsp_url = rtsp_url
        self.output_queue = output_queue
        self.width = width
        self.height = height
        self.target_fps = fps
        self.stop_event = mp.Event()

        # FFmpeg process
        self.process = None

        # FFmpeg command
        self.command = [
            "ffmpeg",
            "-hwaccel",
            "auto",  # Let FFmpeg choose best hardware acceleration
            "-rtsp_transport",
            "tcp",
            "-rtsp_flags",
            "prefer_tcp",
            "-i",
            self.rtsp_url,
            "-an",  # Disable audio
            "-vf",
            f"scale={width}:{height}",
            "-fps_mode",
            "vfr",
            "-framerate",
            str(fps),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:",
        ]

        # Error handling
        self.reconnect_delay = 1.0
        self.max_reconnect_delay = 30.0
        self.consecutive_failures = 0
        self.max_consecutive_failures = 10

    def run(self):
        """Start the main frame reading loop in a separate process"""
        frame_size = self.width * self.height * 3
        last_frame_time = 0
        frame_interval = 1.0 / self.target_fps

        while not self.stop_event.is_set():
            try:
                if self.process is None:
                    self._start_ffmpeg()
                    continue

                current_time = time.time()
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.001)
                    continue

                raw_frame = self.process.stdout.read(frame_size)
                if len(raw_frame) == 0:
                    raise EOFError("FFmpeg process ended")

                if len(raw_frame) < frame_size:
                    raise ValueError("Incomplete frame")

                frame = np.frombuffer(raw_frame, np.uint8).reshape(
                    (self.height, self.width, 3)
                )

                if not self.output_queue.full():
                    self.output_queue.put(frame)
                    last_frame_time = current_time
                    self.consecutive_failures = 0
                    self.reconnect_delay = 1.0

            except (EOFError, ValueError) as e:
                logger.error(f"Stream error: {e}")
                self._cleanup_process()
                self._handle_reconnect()
            except Exception as e:
                logger.error(f"Error reading stream: {e}")
                self._cleanup_process()
                self._handle_reconnect()

    def stop(self):
        """Stop the process and clean up resources"""
        self.stop_event.set()
        self._cleanup_process()

    def _start_ffmpeg(self):
        """Start FFmpeg subprocess"""
        try:
            logger.info(f"Starting FFmpeg process for {self.rtsp_url}")
            logger.info(" ".join(self.command))
            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**9,
            )
        except Exception as e:
            logger.error(f"Failed to start FFmpeg: {e}")
            self._handle_reconnect()

    def _cleanup_process(self):
        """Clean up FFmpeg subprocess"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            finally:
                self.process = None

    def _handle_reconnect(self):
        """Handle reconnection logic with exponential backoff"""
        self.consecutive_failures += 1
        if self.consecutive_failures > self.max_consecutive_failures:
            logger.error("Too many consecutive failures, stopping reader")
            self.stop_event.set()
            return

        time.sleep(self.reconnect_delay)
        self.reconnect_delay = min(self.reconnect_delay * 1.5, self.max_reconnect_delay)
