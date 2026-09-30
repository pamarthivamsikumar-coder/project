from pathlib import Path
import subprocess, sys

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'models' / 'face_landmarker.task'
RISK = ROOT / 'models' / 'risk_model.joblib'


def run_module(module):
    subprocess.check_call([sys.executable, '-m', module], cwd=ROOT)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'desktop':
        if not MODEL.exists():
            run_module('tools.download_face_model')
        if not RISK.exists():
            run_module('tools.train_risk_model')
        run_module('app.main')
    else:
        if not MODEL.exists():
            run_module('tools.download_face_model')
        if not RISK.exists():
            run_module('tools.train_risk_model')
        run_module('web_app')
