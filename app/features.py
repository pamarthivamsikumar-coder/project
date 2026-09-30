import math
from collections import deque
import numpy as np

def dist(a,b): return math.hypot(a[0]-b[0],a[1]-b[1])
def ear(p): return (dist(p[1],p[5])+dist(p[2],p[4]))/(2*max(dist(p[0],p[3]),1e-6))
def mar(p): return (dist(p[1],p[5])+dist(p[2],p[4]))/(2*max(dist(p[0],p[3]),1e-6))

class RPPGEstimator:
    def __init__(self,fps=20): self.fps=fps; self.samples=deque(maxlen=fps*12)
    def add(self,roi):
        if roi is not None and roi.size:
            self.samples.append(float(np.mean(roi[:,:,1])))
    def bpm(self):
        if len(self.samples)<self.fps*5: return None
        x=np.asarray(self.samples,float); x=(x-x.mean())/(x.std()+1e-8)
        f=np.fft.rfftfreq(len(x),1/self.fps); p=np.abs(np.fft.rfft(x))**2
        band=(f>=.75)&(f<=3.0)
        if not band.any(): return None
        return float(f[band][np.argmax(p[band])]*60)
