import time
import logging
import psutil
import numpy as np
import torch
import cv2
import asyncio

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class BenchmarkMetrics:
    fps: float = 0.0
    detection_time: float = 0.0
    reid_time: float = 0.0
    memory_usage: float = 0.0
    gpu_memory: float = 0.0
    cpu_usage: float = 0.0
    detection_accuracy: float = 0.0
    reid_accuracy: float = 0.0
    end_to_end_latency: float = 0.0


class SystemBenchmark:
    def __init__(self, video_processor, reid_manager):
        self.video_processor = video_processor
        self.reid_manager = reid_manager
        self.metrics = defaultdict(list)
        self.setup_logging()

    def setup_logging(self):
        """Setup benchmark logging"""
        self.logger = logging.getLogger("Benchmark")
        self.logger.setLevel(logging.INFO)

        # Create benchmark-specific log file
        fh = logging.FileHandler("benchmark_results.log")
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def measure_system_metrics(self) -> Dict[str, float]:
        """Measure current system resource usage"""
        metrics = {}

        # CPU Usage
        metrics["cpu_usage"] = psutil.cpu_percent(interval=1)

        # Memory Usage
        memory = psutil.Process().memory_info()
        metrics["memory_usage"] = memory.rss / 1024 / 1024  # Convert to MB

        # GPU Memory if available
        if torch.cuda.is_available():
            metrics["gpu_memory"] = (
                torch.cuda.memory_allocated() / 1024 / 1024
            )  # Convert to MB
            metrics["gpu_utilization"] = torch.cuda.utilization()

        return metrics

    def benchmark_detection(self, test_frames: List[np.ndarray]) -> Dict[str, float]:
        """Benchmark YOLO detection performance"""
        metrics = {}
        detection_times = []

        try:
            for frame in test_frames:
                start_time = time.time()
                with torch.inference_mode():
                    results = self.video_processor.model.track(
                        source=frame,
                        conf=0.5,
                        iou=0.7,
                        persist=True,
                        tracker="bytetrack.yaml",
                        verbose=False,
                    )
                detection_times.append(time.time() - start_time)

            metrics["avg_detection_time"] = np.mean(detection_times)
            metrics["detection_fps"] = 1.0 / metrics["avg_detection_time"]

        except Exception as e:
            self.logger.error(f"Error in detection benchmark: {e}")

        return metrics

    def benchmark_reid(
        self, person_crops: List[Tuple[int, np.ndarray]]
    ) -> Dict[str, float]:
        """Benchmark ReID performance"""
        metrics = {}
        reid_times = []

        try:
            for _ in range(5):  # Multiple runs for stability
                start_time = time.time()
                self.reid_manager._extract_features_batch(
                    [crop for _, crop in person_crops]
                )
                reid_times.append(time.time() - start_time)

            metrics["avg_reid_time"] = np.mean(reid_times)
            metrics["reid_fps"] = 1.0 / metrics["avg_reid_time"]

        except Exception as e:
            self.logger.error(f"Error in ReID benchmark: {e}")

        return metrics

    async def benchmark_multi_camera(
        self, num_cameras: int, test_video_path: str, duration: int = 60
    ) -> Dict[str, BenchmarkMetrics]:
        """Benchmark system with multiple camera streams"""
        camera_metrics = {}
        start_time = time.time()

        try:
            # Add test cameras
            for i in range(num_cameras):
                camera_id = f"test_camera_{i}"
                self.video_processor.add_camera(
                    camera_id, test_video_path, "test_company"
                )
                camera_metrics[camera_id] = []

            # Collect metrics for specified duration
            while time.time() - start_time < duration:
                system_metrics = self.measure_system_metrics()

                for camera_id in camera_metrics:
                    frames_processed = self.video_processor.fps_counters[camera_id][
                        "frames"
                    ]
                    camera_metrics[camera_id].append(
                        {
                            "timestamp": time.time(),
                            "fps": frames_processed,
                            **system_metrics,
                        }
                    )

                await asyncio.sleep(1)  # Measure every second

        finally:
            # Cleanup test cameras
            for i in range(num_cameras):
                self.video_processor.remove_camera(f"test_camera_{i}")

        return self.analyze_metrics(camera_metrics)

    def analyze_metrics(
        self, raw_metrics: Dict[str, List[Dict]]
    ) -> Dict[str, BenchmarkMetrics]:
        """Analyze collected metrics and generate report"""
        analyzed_metrics = {}

        for camera_id, measurements in raw_metrics.items():
            metrics = BenchmarkMetrics()

            if measurements:
                metrics.fps = np.mean([m["fps"] for m in measurements])
                metrics.cpu_usage = np.mean([m["cpu_usage"] for m in measurements])
                metrics.memory_usage = np.mean(
                    [m["memory_usage"] for m in measurements]
                )

                if "gpu_memory" in measurements[0]:
                    metrics.gpu_memory = np.mean(
                        [m["gpu_memory"] for m in measurements]
                    )

            analyzed_metrics[camera_id] = metrics

        return analyzed_metrics

    def generate_report(self, metrics: Dict[str, BenchmarkMetrics]) -> str:
        """Generate human-readable benchmark report"""
        report = ["System Benchmark Report", "=" * 50, ""]

        # System Information
        report.append("System Information:")
        report.append(f"CPU Count: {psutil.cpu_count()}")
        report.append(
            f"Total Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB"
        )
        if torch.cuda.is_available():
            report.append(f"GPU: {torch.cuda.get_device_name(0)}")
        report.append("")

        # Per-Camera Metrics
        for camera_id, m in metrics.items():
            report.append(f"Camera: {camera_id}")
            report.append("-" * 30)
            report.append(f"Average FPS: {m.fps:.2f}")
            report.append(f"CPU Usage: {m.cpu_usage:.1f}%")
            report.append(f"Memory Usage: {m.memory_usage:.1f} MB")
            if m.gpu_memory > 0:
                report.append(f"GPU Memory: {m.gpu_memory:.1f} MB")
            report.append("")

        # Save report
        report_str = "\n".join(report)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"benchmark_report_{timestamp}.txt", "w") as f:
            f.write(report_str)

        return report_str


def benchmark_accuracy(
    self,
    ground_truth_path: str,
    test_video_path: str,
    output_path: Optional[str] = None,
) -> Dict[str, float]:
    """Benchmark detection and ReID accuracy against labeled ground truth"""
    metrics = {}

    try:
        # Load ground truth annotations
        gt_data = self.load_ground_truth(ground_truth_path)

        # Process test video
        cap = cv2.VideoCapture(test_video_path)
        frame_idx = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        reid_matches = 0
        reid_total = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection
            results = self.video_processor.model.track(
                source=frame,
                conf=0.5,
                iou=0.7,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )

            # Compare with ground truth
            gt_boxes = gt_data.get(frame_idx, [])
            pred_boxes = results.boxes.xyxy.cpu().numpy()

            # Calculate detection metrics
            for gt_box in gt_boxes:
                matched = False
                for pred_box in pred_boxes:
                    if self.calculate_iou(gt_box, pred_box) > 0.5:
                        true_positives += 1
                        matched = True
                        break
                if not matched:
                    false_negatives += 1

            false_positives += len(pred_boxes) - true_positives

            # ReID metrics
            if len(gt_boxes) > 0:
                reid_results = self.reid_manager.update(
                    "test",
                    [
                        (i, frame[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])])
                        for i, box in enumerate(pred_boxes)
                    ],
                )

                reid_matches += sum(
                    1
                    for gt_id, pred_id in zip(
                        gt_data["ids"][frame_idx], reid_results.values()
                    )
                    if gt_id == pred_id
                )
                reid_total += len(gt_data["ids"][frame_idx])

            frame_idx += 1

        # Calculate final metrics
        metrics["detection_precision"] = true_positives / (
            true_positives + false_positives
        )
        metrics["detection_recall"] = true_positives / (
            true_positives + false_negatives
        )
        metrics["detection_f1"] = (
            2
            * (metrics["detection_precision"] * metrics["detection_recall"])
            / (metrics["detection_precision"] + metrics["detection_recall"])
        )
        metrics["reid_accuracy"] = reid_matches / reid_total if reid_total > 0 else 0

        # Save detailed results if output path provided
        if output_path:
            self.save_accuracy_results(metrics, output_path)

    except Exception as e:
        self.logger.error(f"Error in accuracy benchmark: {e}")

    return metrics


@staticmethod
def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate Intersection over Union between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection / (area1 + area2 - intersection)


def save_accuracy_results(self, metrics: Dict[str, float], output_path: str):
    """Save detailed accuracy results to file"""
    with open(output_path, "w") as f:
        f.write("Accuracy Benchmark Results\n")
        f.write("=========================\n\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
