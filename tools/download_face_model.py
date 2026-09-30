from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'models' / 'face_landmarker.task'
URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
OUT.parent.mkdir(parents=True, exist_ok=True)
print('Downloading face landmark model...')
with urlopen(URL, timeout=60) as r, open(OUT, 'wb') as f:
    while True:
        chunk = r.read(1024 * 1024)
        if not chunk:
            break
        f.write(chunk)
print(f'Saved: {OUT}')
