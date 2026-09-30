import threading
import time
from collections import deque
from typing import Any, Dict

import cv2

from app.alerts import AlertManager
from app.config import CAMERA_SOURCE, SPEED_SOURCE
from app.face_monitor import FaceMonitor
from app.gps_input import GPSSpeed
from app.lane_detector import detect_lane_departure
from app.risk_model import RiskModel
from flask import Flask, Response, jsonify, render_template

app = Flask(__name__, template_folder="templates", static_folder="static")

_frame_lock = threading.Lock()
_latest_frame = None
_status: Dict[str, Any] = {
    "risk": 0.0,
    "label": "alert",
    "reason": "Loading...",
    "ear": None,
    "mar": None,
    "lane_deviation": 0.0,
    "speed_kmph": 50.0,
    "bpm": None,
    "yawns": 0,
    "face_detected": False,
}
_camera_started = False


def _build_status(frame_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "risk": round(float(frame_data.get("risk", 0.0)), 3),
        "label": frame_data.get("label", "alert"),
        "reason": frame_data.get("reason", "Initializing..."),
        "ear": frame_data.get("ear"),
        "mar": frame_data.get("mar"),
        "lane_deviation": round(float(frame_data.get("lane_deviation", 0.0)), 3),
        "speed_kmph": round(float(frame_data.get("speed_kmph", 50.0)), 1),
        "bpm": frame_data.get("bpm"),
        "yawns": int(frame_data.get("yawns", 0)),
        "face_detected": bool(frame_data.get("face_detected", False)),
    }


def _camera_loop() -> None:
    global _latest_frame, _status

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        _status["reason"] = "Camera unavailable. Check the webcam connection."
        return

    face = FaceMonitor()
    model = RiskModel()
    speed = GPSSpeed(SPEED_SOURCE)
    alert = AlertManager()

    yawn_times = deque()
    yawn_window = 30
    yawn_high_count = 3

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue

            frame, dev, cross = detect_lane_departure(frame)
            features = face.process(frame)
            data = {
                **features,
                "lane_deviation": dev,
                "speed_kmph": speed.get_speed(),
            }

            risk, label, probs = model.predict(data)
            reason = model.explain(data)
            now = time.time()
            mar = features.get("mar")

            if mar is not None and mar > 0.60:
                if not yawn_times or now - yawn_times[-1] > 3:
                    yawn_times.append(now)

            while yawn_times and now - yawn_times[0] > yawn_window:
                yawn_times.popleft()

            yawn_count = len(yawn_times)
            if yawn_count >= yawn_high_count:
                risk = max(risk, 0.90)
                label = "high_impairment_risk"
                reason = f"Frequent yawning ({yawn_count} yawns in {yawn_window}s)"

            alert.trigger(label, reason, risk)
            frame = alert.draw(frame, risk)

            metrics = {
                "risk": risk,
                "label": label,
                "reason": reason,
                "ear": features.get("ear"),
                "mar": features.get("mar"),
                "lane_deviation": dev,
                "speed_kmph": data["speed_kmph"],
                "bpm": features.get("bpm"),
                "yawns": yawn_count,
                "face_detected": bool(features.get("face_detected", False)),
            }

            with _frame_lock:
                _latest_frame = frame.copy()
                _status.update(_build_status(metrics))

            time.sleep(0.03)
    finally:
        face.close()
        cap.release()


def start_camera() -> None:
    global _camera_started
    if _camera_started:
        return
    _camera_started = True
    thread = threading.Thread(target=_camera_loop, daemon=True)
    thread.start()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/status")
def status():
    with _frame_lock:
        payload = dict(_status)
    return jsonify(payload)


@app.get("/video_feed")
def video_feed():
    def generate():
        while True:
            with _frame_lock:
                frame = None if _latest_frame is None else _latest_frame.copy()

            if frame is not None:
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + encoded.tobytes()
                        + b"\r\n"
                    )

            time.sleep(0.033)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    start_camera()
    app.run(host="0.0.0.0", port=5000, debug=False)
