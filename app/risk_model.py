import joblib
import pandas as pd

from .config import RISK_MODEL


FEATURES = [
    "ear",
    "mar",
    "blink_duration",
    "gaze_deviation",
    "lane_deviation",
    "speed_kmph",
    "heart_rate",
]


class RiskModel:

    def __init__(self):
        bundle = joblib.load(RISK_MODEL)

        self.pipeline = bundle["pipeline"]
        self.features = bundle.get("features", FEATURES)

        print("Loaded risk model:", RISK_MODEL)
        print("Model classes:", list(self.pipeline.classes_))

    def vector(self, f):

        values = {
            "ear": (
                f.get("ear")
                if f.get("ear") is not None
                else 0.30
            ),

            "mar": (
                f.get("mar")
                if f.get("mar") is not None
                else 0.25
            ),

            "blink_duration": f.get(
                "blink_duration", 0.1
            ),

            "gaze_deviation": abs(
                f.get("gaze_proxy", 0)
            ),

            "lane_deviation": abs(
                f.get("lane_deviation", 0)
            ),

            "speed_kmph": f.get(
                "speed_kmph", 50.0
            ),

            "heart_rate": (
                f.get("bpm")
                if f.get("bpm") is not None
                else 78.0
            ),
        }

        return pd.DataFrame(
            [[
                values.get(feature, 0)
                for feature in self.features
            ]],
            columns=self.features
        )

    def predict(self, f):

        x = self.vector(f)

        ml_label = self.pipeline.predict(x)[0]

        probs = self.pipeline.predict_proba(x)[0]

        classes = list(self.pipeline.classes_)

        probabilities = dict(
            zip(classes, probs)
        )

        # Start in SAFE state.
        # The real UTA-RLDD model is used as
        # the facial drowsiness component.
        risk = 0.20

        # Live features
        ear = f.get("ear")

        mar = f.get("mar")

        gaze = abs(
            f.get("gaze_proxy", 0)
        )

        lane = abs(
            f.get("lane_deviation", 0)
        )

        blink = f.get(
            "blink_duration", 0
        )

        speed = f.get(
            "speed_kmph", 0
        )

        # Safety cues

        eye_closure = (
            ear is not None
            and ear < 0.21
        )

        prolonged_closure = (
            blink >= 0.8
        )

        yawning = (
            mar is not None
            and mar > 0.60
        )

        gaze_distraction = (
            gaze > 0.35
        )

        lane_departure = (
            lane > 0.18
        )

        high_speed = (
            speed > 80
        )

        # Increase risk according to live cues

        if eye_closure:
            risk = max(
                risk,
                0.85
            )

        if prolonged_closure:
            risk = max(
                risk,
                0.90
            )

        if yawning:
            risk = max(
                risk,
                0.65
            )

        if gaze_distraction:
            risk = max(
                risk,
                0.80
            )

        if lane_departure:
            risk = max(
                risk,
                0.85
            )

        if high_speed:
            risk = max(
                risk,
                0.70
            )

        # Keep risk between 0 and 1

        risk = max(
            0.0,
            min(1.0, risk)
        )

        # Final system classification

        severe_cues = sum([
            eye_closure,
            prolonged_closure,
            lane_departure,
            gaze_distraction,
        ])

        if risk >= 0.85 or severe_cues >= 2:

            final_label = (
                "high_impairment_risk"
            )

        elif gaze_distraction or lane_departure:

            final_label = "distracted"

        elif yawning:

            final_label = "drowsy"

        else:

            final_label = "alert"

        return (
            risk,
            final_label,
            probabilities
        )

    def explain(self, f):

        reasons = []

        ear = f.get("ear")

        mar = f.get("mar")

        gaze = abs(
            f.get("gaze_proxy", 0)
        )

        lane = abs(
            f.get("lane_deviation", 0)
        )

        blink = f.get(
            "blink_duration", 0
        )

        speed = f.get(
            "speed_kmph", 0
        )

        if (
            ear is not None
            and ear < 0.21
        ):
            reasons.append(
                "eye closure"
            )

        if blink >= 0.8:
            reasons.append(
                "prolonged eye closure"
            )

        if (
            mar is not None
            and mar > 0.60
        ):
            reasons.append(
                "yawning"
            )

        if gaze > 0.35:
            reasons.append(
                "gaze distraction"
            )

        if lane > 0.18:
            reasons.append(
                "lane departure"
            )

        if speed > 80:
            reasons.append(
                "high speed"
            )

        if reasons:

            return (
                "Factors: "
                + ", ".join(
                    reasons[:3]
                )
            )

        return (
            "Factors: no strong abnormal cue"
        )