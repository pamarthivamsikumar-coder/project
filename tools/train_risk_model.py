from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'demo_multimodal_dataset.csv'
OUT = ROOT / 'models' / 'risk_model.joblib'
FEATURES = ['ear','mar','blink_duration','gaze_deviation','lane_deviation','speed_kmph','heart_rate']
LABELS = ['alert','drowsy','distracted','high_impairment_risk']


def make_dataset(n=2400, seed=42):
    rng = np.random.default_rng(seed)
    rows=[]
    for _ in range(n):
        label = rng.choice(LABELS, p=[.30,.30,.25,.15])
        if label == 'alert':
            ear=rng.normal(.30,.035); mar=rng.normal(.25,.08); blink=rng.uniform(.05,.35); gaze=rng.normal(.10,.08); lane=rng.normal(.04,.04); speed=rng.normal(48,12); hr=rng.normal(78,9)
        elif label == 'drowsy':
            ear=rng.normal(.18,.035); mar=rng.normal(.55,.16); blink=rng.uniform(.7,2.2); gaze=rng.normal(.15,.12); lane=rng.normal(.12,.08); speed=rng.normal(50,14); hr=rng.normal(75,12)
        elif label == 'distracted':
            ear=rng.normal(.28,.04); mar=rng.normal(.25,.10); blink=rng.uniform(.05,.5); gaze=rng.normal(.55,.18); lane=rng.normal(.18,.10); speed=rng.normal(55,16); hr=rng.normal(82,12)
        else:
            ear=rng.normal(.22,.05); mar=rng.normal(.45,.18); blink=rng.uniform(.4,1.8); gaze=rng.normal(.45,.22); lane=rng.normal(.25,.12); speed=rng.normal(78,18); hr=rng.normal(95,16)
        rows.append([np.clip(ear,.08,.45),np.clip(mar,.05,1.1),np.clip(blink,0,3),np.clip(abs(gaze),0,1),np.clip(abs(lane),0,.8),np.clip(speed,0,140),np.clip(hr,45,150),label])
    return pd.DataFrame(rows, columns=FEATURES+['label'])


def main():
    DATA.parent.mkdir(parents=True, exist_ok=True); OUT.parent.mkdir(parents=True, exist_ok=True)
    if DATA.exists():
        df=pd.read_csv(DATA)
    else:
        df=make_dataset(); df.to_csv(DATA,index=False)
    X=df[FEATURES]; y=df['label']
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    pipe=Pipeline([('scaler',StandardScaler()),('model',LogisticRegression(max_iter=1500,multi_class='auto'))])
    pipe.fit(Xtr,ytr)
    pred=pipe.predict(Xte)
    print('Dataset:', len(df), 'rows')
    print('Validation accuracy:', round(accuracy_score(yte,pred),3))
    print(classification_report(yte,pred,zero_division=0))
    joblib.dump({'pipeline':pipe,'features':FEATURES,'labels':LABELS,'dataset_note':'Synthetic demonstration dataset. Replace/augment with licensed public datasets for research evaluation.'},OUT)
    print('Saved:',OUT)

if __name__=='__main__': main()
