from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
CAMERA_SOURCE = 0
SPEED_SOURCE = ROOT / 'data' / 'speed.json'
FACE_MODEL = ROOT / 'models' / 'face_landmarker.task'
RISK_MODEL = ROOT / 'models' / 'risk_model_real.joblib'
EAR_CLOSED_THRESHOLD = 0.21
MAR_YAWN_THRESHOLD = 0.60
EYE_CLOSED_SECONDS = 1.2
YAWN_SECONDS = 1.0
LANE_DEVIATION_THRESHOLD = 0.18
HIGH_SPEED_KMPH = 80.0
