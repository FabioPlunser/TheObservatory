"""Video processing transformation for WebRTC and YOLO detection"""

from aiortc import MediaStreamTrack
from av import VideoFrame

import numpy as np
import time
import cv2
import logging

logger = logging.getLogger(__name__)


class VideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, rtsp_reader, camera_id, model, device):
        super().__init__()
        self.rtsp_reader = rtsp_reader
        self.camera_id = camera_id
        self.model = model
        self.device = device
        self._last_frame_time = time.monotonic() * 1000
        self._start = time.time()
        self._timestamp = 0

    async def next_timestamp(self):
        if self._timestamp == 0:
            self._start = time.time()
            self._timestamp = 90
        else:
            self._timestamp += int(1 / 30 * 1000000)
        return self._timestamp, 1000000

    async def recv(self):
        frame = self.rtsp_reader.get_frame() or np.zeros((720, 1280, 3), dtype=np.uint8)

        current_time = time.monotonic() * 1000
        if current_time - self._last_frame_time >= 100:
            try:
                # Perform YOLO detection
                results = self.model(frame, device=self.device)

                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        if int(box.cls) == 0:  # Person class
                            conf = float(box.conf)
                            x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(
                                frame,
                                f"Person {conf:.2f}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2,
                            )

                self._last_frame_time = current_time

            except Exception as e:
                logger.error(f"Error in YOLO processing: {str(e)}")

        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        pts, time_base = await self.next_timestamp()
        video_frame.pts = pts
        video_frame.time_base = time_base

        return video_frame
