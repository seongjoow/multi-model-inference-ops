"""추론 백엔드.

yolo  : Ultralytics YOLO. 이미지 리스트를 한 번에 넘기면 한 배치로 추론한다.
dummy : 모델 없이 고정 시간만 쓴다. 큐·배치 동작을 모델과 분리해 검증하거나 CI 스모크 테스트에 쓴다.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from common.jsonlog import log_event
from workers.cv.settings import Settings

log = logging.getLogger("cv-worker")

Detections = list[dict[str, Any]]


class DummyBackend:
    def __init__(self, base_ms: float, per_item_ms: float) -> None:
        self.name = "dummy"
        self.base_s = base_ms / 1000
        self.per_item_s = per_item_ms / 1000

    def decode(self, data: bytes) -> bytes:
        return data

    def infer(self, images: list[Any]) -> list[Detections]:
        time.sleep(self.base_s + self.per_item_s * len(images))
        return [[] for _ in images]


class YoloBackend:
    def __init__(self, s: Settings) -> None:
        import cv2  # 무거운 의존성은 yolo 백엔드에서만 불러온다
        import numpy as np
        import torch
        from ultralytics import YOLO
        from ultralytics.utils.downloads import attempt_download_asset

        self._cv2, self._np = cv2, np
        if s.torch_threads > 0:
            torch.set_num_threads(s.torch_threads)

        weights = Path(s.model_dir) / s.model
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights = Path(attempt_download_asset(weights))  # 없으면 Ultralytics 공개 릴리스에서 받는다
        self.model = YOLO(str(weights))
        self.name = weights.stem
        self.kw = dict(imgsz=s.imgsz, device=s.device, half=s.half, conf=s.conf, max_det=s.max_det, verbose=False)

        t0 = time.perf_counter()
        self.model.predict(np.zeros((s.imgsz, s.imgsz, 3), dtype=np.uint8), **self.kw)  # 워밍업
        log_event(
            log,
            "model_loaded",
            model=self.name,
            weights=str(weights),
            device=s.device,
            half=s.half,
            torch=torch.__version__,
            cuda=torch.cuda.is_available(),
            warmup_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    def decode(self, data: bytes):
        img = self._cv2.imdecode(self._np.frombuffer(data, dtype=self._np.uint8), self._cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cannot decode image")
        return img  # BGR, Ultralytics가 numpy 입력으로 기대하는 형식

    def infer(self, images: list[Any]) -> list[Detections]:
        results = self.model.predict(images, **self.kw)
        out: list[Detections] = []
        for res in results:
            boxes = res.boxes
            out.append(
                [
                    {"label": res.names[int(c)], "conf": round(float(p), 4), "xyxy": [round(v, 1) for v in xyxy]}
                    for xyxy, p, c in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())
                ]
            )
        return out


def make_backend(s: Settings) -> DummyBackend | YoloBackend:
    if s.backend == "dummy":
        return DummyBackend(s.dummy_base_ms, s.dummy_per_item_ms)
    if s.backend == "yolo":
        return YoloBackend(s)
    raise ValueError(f"unknown CV_BACKEND: {s.backend!r} (yolo | dummy)")
