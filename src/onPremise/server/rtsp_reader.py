from typing import Optional

import cv2
import numpy as np
import threading
import subprocess
import queue
import platform
import logging


logger = logging.getLogger(__name__)


class RTSPReader:
    def __init__(self, url: str):
        self.url = url
        self.running = False
        self.last_frame = None
        self.lock = threading.Lock()
        self.frame_queue = queue.Queue(maxsize=10)
        self.process: Optional[subprocess.Popen] = None
        self.os_type = platform.system().lower()

    def get_ffmpeg_read_args(self):
        base_args = ["ffmpeg", "-rtsp_transport", "tcp", "-stimeout", "5000000"]

        input_args = ["-re", "-i", self.url]

        output_args = [
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-flags",
            "low_delay",
            "-fflags",
            "nobuffer+discardcorrupt",
            "-vf",
            "scale=1280:720",
            "-an",
            "pipe:1",
        ]

        return base_args + input_args + output_args

    async def start(self):
        try:
            self.running = True
            self.process = subprocess.Popen(
                self.get_ffmpeg_read_args(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8,
            )

            # Start worker threads
            for worker in [self._read_frames, self._log_stderr]:
                thread = threading.Thread(target=worker)
                thread.daemon = True
                thread.start()

        except Exception as e:
            logger.error(f"Error in RTSP reader: {str(e)}")
            self.stop()

    def _read_frames(self):
        try:
            while self.running and self.process and self.process.poll() is None:
                raw_frame = self.process.stdout.read(1280 * 720 * 3)
                if not raw_frame:
                    break

                frame = np.frombuffer(raw_frame, np.uint8).reshape((720, 1280, 3))

                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    continue

        except Exception as e:
            logger.error(f"Error reading frames: {str(e)}")

    def _log_stderr(self):
        while self.running and self.process and self.process.poll() is None:
            line = self.process.stderr.readline()
            if line:
                logger.debug(f"FFmpeg: {line.decode().strip()}")

    def get_frame(self):
        with self.lock:
            return self.last_frame.copy() if self.last_frame is not None else None

    def stop(self):
        self.running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
