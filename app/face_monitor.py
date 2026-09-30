import time, cv2, numpy as np, mediapipe as mp
from .features import ear, mar, RPPGEstimator
from .config import FACE_MODEL, EAR_CLOSED_THRESHOLD, MAR_YAWN_THRESHOLD, EYE_CLOSED_SECONDS, YAWN_SECONDS

LEFT_EYE=[33,160,158,133,153,144]; RIGHT_EYE=[362,385,387,263,373,380]; MOUTH=[61,13,14,291,17,84]

class FaceMonitor:
    def __init__(self):
        vision=mp.tasks.vision
        opts=vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_MODEL)), running_mode=vision.RunningMode.IMAGE, num_faces=1, output_face_blendshapes=False)
        self.landmarker=vision.FaceLandmarker.create_from_options(opts)
        self.rppg=RPPGEstimator(); self.closed_since=None; self.yawn_since=None
    def close(self): self.landmarker.close()
    def process(self,frame):
        h,w=frame.shape[:2]; rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
        res=self.landmarker.detect(image)
        out={'face_detected':False,'ear':None,'mar':None,'drowsy':False,'yawning':False,'gaze_proxy':0.0,'bpm':None}
        if not res.face_landmarks: return out
        pts=[(lm.x*w,lm.y*h) for lm in res.face_landmarks[0]]; out['face_detected']=True
        out['ear']=(ear([pts[i] for i in LEFT_EYE])+ear([pts[i] for i in RIGHT_EYE]))/2
        out['mar']=mar([pts[i] for i in MOUTH]); now=time.time()
        if out['ear']<EAR_CLOSED_THRESHOLD: self.closed_since=self.closed_since or now
        else: self.closed_since=None
        if out['mar']>MAR_YAWN_THRESHOLD: self.yawn_since=self.yawn_since or now
        else: self.yawn_since=None
        out['drowsy']=(self.closed_since is not None and now-self.closed_since>=EYE_CLOSED_SECONDS) or (self.yawn_since is not None and now-self.yawn_since>=YAWN_SECONDS)
        out['yawning']=self.yawn_since is not None and now-self.yawn_since>=YAWN_SECONDS
        a=np.mean([pts[33],pts[133]],axis=0); b=np.mean([pts[362],pts[263]],axis=0); nose=np.array(pts[1]); mid=(a+b)/2; width=max(abs(b[0]-a[0]),1)
        out['gaze_proxy']=float(np.clip((nose[0]-mid[0])/width,-1,1))
        x1,x2=int(.35*w),int(.65*w); y1,y2=int(.15*h),int(.40*h); self.rppg.add(frame[y1:y2,x1:x2]); out['bpm']=self.rppg.bpm()
        return out
