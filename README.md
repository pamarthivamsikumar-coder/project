# Explainable Federated Multimodal Driver Safety System

Final CSE project implementation for **camera + vehicle speed + lane-departure** based driver monitoring. EEG and machine/industrial sensors are intentionally excluded.

## What is included
- Live camera face landmarks, EAR, MAR, blink-duration proxy, gaze proxy
- Experimental camera rPPG heart-rate estimate
- OpenCV lane-departure signal
- Configurable vehicle-speed input
- Explainable risk factors
- Federated Averaging demonstration
- Real-dataset ingestion and training pipeline
- Saved model/report generation

## Real datasets
The raw third-party datasets are **not bundled** because they are large and remain subject to their own access/licensing terms.

Recommended sources:
- UTA-RLDD: https://sites.google.com/view/utarldd/home
- NTHU-DDD: https://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/
- YawDD: https://ieee-dataport.org/datasets/yawdd-yawning-detection-dataset
- Drive&Act: https://driveandact.com/

### Dataset roles
- UTA-RLDD: primary real-world RGB drowsiness data (alert / low vigilance / drowsy).
- NTHU-DDD: drowsiness behaviors including yawning, blinking and nodding.
- YawDD: yawning/talking/normal driver video.
- Drive&Act: distracted-driver activity recognition.

Do not claim that these datasets directly contain drug impairment. The project's `high_impairment_risk` output is an inferred multimodal risk category, not a drug-identification result.

## Folder placement
After downloading datasets according to their official terms:
```
datasets/
  uta_rldd/
  yawdd/
  drive_act/
```

You may also use a real, already-extracted feature CSV. Required columns:
```
ear,mar,blink_duration,gaze_deviation,lane_deviation,speed_kmph,heart_rate,label
```
Labels supported by the final model:
```
alert,drowsy,distracted,high_impairment_risk
```

## One-time real-data preparation
Activate the project environment first, then:
```
python -m tools.download_face_model
python -m tools.prepare_real_dataset --videos datasets/uta_rldd
python -m tools.train_real_model
```

For a processed real feature CSV:
```
python -m tools.prepare_real_dataset --csv path\to\real_features.csv
python -m tools.train_real_model
```

The model and evaluation report are written to:
```
models/risk_model_real.joblib
outputs/real_dataset_report.json
```

## Important data-integrity rule
The real drowsiness datasets do not automatically provide the exact synchronized vehicle-speed, lane-departure and rPPG channels required by this project's final multimodal vector. Do **not** invent those values and call them measured dataset data. The live application obtains speed and lane information separately, while rPPG is estimated from the live camera.

## Python 3.14
This project pins MediaPipe 1.0.1, which currently publishes a Windows x86-64 wheel tagged `py3` on PyPI. If your Python 3.14 environment cannot install the wheel, use Python 3.13 for this project rather than changing the project code. See the installation commands below.

## Installation
```
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the live system
After training the real model, copy/rename the real model as `models/risk_model.joblib` or update `app/config.py` to point to `risk_model_real.joblib`, then:
```
python -m app.main
```
Press `q` to quit and `s` to save a snapshot.

## Research/viva wording
Use: "The model is trained/validated on public driver-monitoring datasets for drowsiness and distraction. Vehicle speed and lane-departure signals are treated as separate live vehicle-context modalities. The final system fuses these signals for explainable risk estimation."
Do not state that a public drowsiness dataset is a drug-impairment dataset unless its documentation explicitly says so.
