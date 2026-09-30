from pathlib import Path
import json, joblib, pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'real_multimodal_dataset.csv'; OUT=ROOT/'models'/'risk_model_real.joblib'; REPORT=ROOT/'outputs'/'real_dataset_report.json'
FEATURES=['ear','mar','blink_duration','gaze_deviation','lane_deviation','speed_kmph','heart_rate']
LABELS=['alert','drowsy','distracted','high_impairment_risk']

def main():
    if not DATA.exists(): raise SystemExit(f'Missing {DATA}. Run tools/prepare_real_dataset.py first.')
    df=pd.read_csv(DATA); df=df.dropna(subset=FEATURES+['label'])
    X=df[FEATURES]; y=df['label']
    if y.nunique()<2: raise SystemExit('Need at least two real classes to train.')
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    pipe=Pipeline([('scaler',StandardScaler()),('model',LogisticRegression(max_iter=2000,class_weight='balanced'))])
    pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
    acc=float(accuracy_score(yte,pred)); report=classification_report(yte,pred,zero_division=0,output_dict=True)
    OUT.parent.mkdir(parents=True,exist_ok=True); REPORT.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({'pipeline':pipe,'features':FEATURES,'labels':LABELS,'dataset_note':'Model trained from user-supplied real dataset features. Speed/heart-rate/lane values are only real when supplied by the dataset; otherwise use the live sensors separately.'},OUT)
    REPORT.write_text(json.dumps({'rows':len(df),'classes':sorted(y.unique().tolist()),'accuracy':acc,'classification_report':report,'confusion_matrix':confusion_matrix(yte,pred).tolist()},indent=2))
    print(f'Validation accuracy: {acc:.3f}')
    print(f'Saved: {OUT}')
    print(f'Report: {REPORT}')
if __name__=='__main__': main()
