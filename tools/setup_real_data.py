from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
print('REAL DATASET SETUP')
print('1. Put UTA-RLDD videos in datasets/uta_rldd/')
print('2. Put YawDD videos/annotations in datasets/yawdd/')
print('3. Put Drive&Act videos/annotations in datasets/drive_act/')
print('4. Then run: python -m tools.prepare_real_dataset --videos datasets/uta_rldd')
print('   or import a processed real feature CSV: python -m tools.prepare_real_dataset --csv path/to/features.csv')
print('5. Then run: python -m tools.train_real_model')
