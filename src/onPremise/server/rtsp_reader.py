import subprocess
import numpy as np
import multiprocessing as mp
import logging
import time
import queue
import signal
import os
from typing import Optional

logger = logging.getLogger("RTSPReader")


class RTSPReader:
    def __init__(
        self,
        rtsp_url: str,
        output_queue: mp.Queue,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self.rtsp_url = rtsp_url
        self.output_queue = output_queue
        self.width = width
        self.height = height
        self.target_fps = fps

        # Process management
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        self.stop_event = mp.Event()

        # Connection settings
        self.reconnect_delay = 1.0
        self.max_reconnect_delay = 30.0
        self.consecutive_failures = 0
        self.max_consecutive_failures = 10
        self.last_frame_time = 0

        # Frame size calculation
        self.frame_size = width * height * 3  # BGR format

        # FFmpeg command optimization
        self.ffmpeg_command = [
            "ffmpeg",
            "-hide_banner",  # Reduce logging noise
            "-loglevel",
            "error",  # Only show errors
            "-fflags",
            "+discardcorrupt",  # Skip corrupted frames
            "-hwaccel",
            "auto",  # Enable hardware acceleration
            "-rtsp_transport",
            "tcp",  # Use TCP for better reliability
            "-use_wallclock_as_timestamps",
            "1",  # Use system clock for timestamps
            "-i",
            self.rtsp_url,
            "-an",  # Disable audio
            "-sn",  # Disable subtitles
            "-dn",  # Disable data streams
            "-vf",
            f"scale={width}:{height}",  # Scale video
            "-r",
            str(fps),  # Set target FPS
            "-pix_fmt",
            "bgr24",  # Set pixel format
            "-f",
            "rawvideo",  # Output format
            "pipe:",  # Output to pipe
        ]

        # Advanced options for problematic streams
        self.advanced_options = [
            "-reorder_queue_size",
            "5",  # Buffer for packet reordering
            "-max_delay",
            "500000",  # Max processing delay (microseconds)
            "-stimeout",
            "5000000",  # Socket timeout (microseconds)
        ]

        self.reader_thread = None

    def start(self):
        """Start the RTSP reader in a separate thread"""
        if not self.is_running:
            self.is_running = True
            self.reader_thread = mp.Process(target=self._read_stream, daemon=True)
            self.reader_thread.start()

    def stop(self):
        """Stop the RTSP reader and cleanup resources"""
        self.is_running = False
        self.stop_event.set()

        if self.process:
            self._cleanup_process()

        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.terminate()
            self.reader_thread.join(timeout=5)

    def _start_ffmpeg(self, use_advanced_options=False):
        """Start FFmpeg process with error handling"""
        try:
            command = self.ffmpeg_command.copy()

            # Add advanced options if needed
            if use_advanced_options:
                # Insert advanced options before output settings
                insert_pos = -4
                for option in reversed(self.advanced_options):
                    command.insert(insert_pos, option)

            logger.debug(f"Starting FFmpeg with command: {' '.join(command)}")

            # Start FFmpeg process
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,  # Explicitly close stdin
                bufsize=self.frame_size * 4,  # Buffer 4 frames worth of data
                preexec_fn=os.setsid,  # Create new process group
            )

            return True

        except Exception as e:
            logger.error(f"Failed to start FFmpeg: {e}")
            return False

    def _cleanup_process(self):
        """Cleanup FFmpeg process with proper signal handling"""
        if self.process:
            try:
                # Try graceful termination first
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)

                # Wait for process to terminate
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if process doesn't terminate
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    self.process.wait(timeout=1)

            except ProcessLookupError:
                pass  # Process already terminated
            except Exception as e:
                logger.error(f"Error cleaning up FFmpeg process: {e}")
            finally:
                self.process = None

    def _read_stream(self):
        """Main stream reading loop with improved error handling"""
        use_advanced_options = False
        frame_interval = 1.0 / self.target_fps
        buffer = b""

        while not self.stop_event.is_set():
            if not self.process:
                if not self._start_ffmpeg(use_advanced_options):
                    self._handle_reconnect()
                    continue

            try:
                # Read exactly one frame worth of data
                while len(buffer) < self.frame_size:
                    chunk = self.process.stdout.read(self.frame_size - len(buffer))
                    if not chunk:
                        raise EOFError("FFmpeg process ended unexpectedly")
                    buffer += chunk

                # Process complete frame
                if len(buffer) >= self.frame_size:
                    # Convert to numpy array
                    frame_data = buffer[: self.frame_size]
                    buffer = buffer[self.frame_size :]

                    frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(
                        (self.height, self.width, 3)
                    )

                    current_time = time.time()
                    if current_time - self.last_frame_time >= frame_interval:
                        try:
                            self.output_queue.put_nowait(frame)
                            self.last_frame_time = current_time
                            self.consecutive_failures = 0
                            self.reconnect_delay = 1.0
                        except queue.Full:
                            pass  # Skip frame if queue is full

            except (EOFError, ValueError) as e:
                logger.error(f"Stream error: {e}")
                self._cleanup_process()
                # Try advanced options after multiple failures
                use_advanced_options = self.consecutive_failures >= 3
                self._handle_reconnect()

            except Exception as e:
                logger.error(f"Error reading stream: {e}")
                self._cleanup_process()
                self._handle_reconnect()

    def _handle_reconnect(self):
        """Handle reconnection with exponential backoff"""
        self.consecutive_failures += 1

        if self.consecutive_failures > self.max_consecutive_failures:
            logger.error(f"Too many consecutive failures for {self.rtsp_url}")
            self.stop_event.set()
            return

        delay = min(
            self.reconnect_delay * (1.5 ** (self.consecutive_failures - 1)),
            self.max_reconnect_delay,
        )

        logger.info(
            f"Reconnecting in {delay:.1f} seconds... (Attempt {self.consecutive_failures})"
        )
        time.sleep(delay)
