import json
from pathlib import Path
class GPSSpeed:
    def __init__(self,source,demo_speed=50): self.source=Path(source); self.demo_speed=float(demo_speed)
    def get_speed(self):
        try: return float(json.loads(self.source.read_text()).get('speed_kmph',self.demo_speed))
        except Exception: return self.demo_speed
