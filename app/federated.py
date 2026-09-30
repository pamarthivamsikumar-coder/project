import numpy as np

def fedavg(local_weights,counts):
    total=float(sum(counts)); out=np.zeros_like(local_weights[0],dtype=float)
    for w,n in zip(local_weights,counts): out+=(n/total)*w
    return out

def demo():
    global_w=np.array([.2,.4,.1,.3,.2,.1]); rng=np.random.default_rng(42)
    local=[global_w+rng.normal(0,.02,global_w.size) for _ in range(3)]
    return fedavg(local,[100,125,150])
