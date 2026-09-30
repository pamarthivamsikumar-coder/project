import cv2
import time

from collections import deque

from .config import (
    CAMERA_SOURCE,
    SPEED_SOURCE
)

from .face_monitor import FaceMonitor

from .lane_detector import (
    detect_lane_departure
)

from .risk_model import RiskModel

from .gps_input import GPSSpeed

from .alerts import AlertManager

from .federated import demo as fed_demo


def put(
    frame,
    text,
    y
):

    cv2.putText(
        frame,
        str(text),
        (15, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


def main():

    print(
        "Starting Explainable "
        "Multimodal Driver Safety System..."
    )

    # -----------------------------------------
    # FEDERATED LEARNING DEMO
    # -----------------------------------------

    print(
        "Federated aggregation demo:",
        [
            round(x, 3)
            for x in fed_demo()
        ]
    )

    # -----------------------------------------
    # CAMERA
    # -----------------------------------------

    cap = cv2.VideoCapture(
        CAMERA_SOURCE
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Camera unavailable. "
            "Change CAMERA_SOURCE "
            "in app/config.py."
        )

    # -----------------------------------------
    # MODULES
    # -----------------------------------------

    face = FaceMonitor()

    model = RiskModel()

    speed = GPSSpeed(
        SPEED_SOURCE
    )

    alert = AlertManager()

    # -----------------------------------------
    # FREQUENT YAWNING SETTINGS
    # -----------------------------------------

    yawn_times = deque()

    # Time window
    YAWN_WINDOW = 30

    # Number of yawns required
    # for high alert
    YAWN_HIGH_COUNT = 3

    # -----------------------------------------
    # MAIN LOOP
    # -----------------------------------------

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            # ---------------------------------
            # LANE DETECTION
            # ---------------------------------

            frame, dev, cross = (
                detect_lane_departure(
                    frame
                )
            )

            # ---------------------------------
            # FACE PROCESSING
            # ---------------------------------

            f = face.process(
                frame
            )

            # ---------------------------------
            # SENSOR DATA
            # ---------------------------------

            data = {

                **f,

                "lane_deviation": dev,

                "speed_kmph":
                    speed.get_speed()
            }

            # ---------------------------------
            # ML PREDICTION
            # ---------------------------------

            risk, label, probs = (
                model.predict(data)
            )

            reason = (
                model.explain(data)
            )

            # ---------------------------------
            # FREQUENT YAWNING DETECTION
            # ---------------------------------

            now = time.time()

            mar = f.get("mar")

            # A yawn is detected when
            # MAR crosses the threshold
            if (
                mar is not None
                and mar > 0.60
            ):

                # Don't count the same
                # continuous yawn repeatedly
                if (
                    not yawn_times
                    or
                    now - yawn_times[-1] > 3
                ):

                    yawn_times.append(
                        now
                    )

            # Remove old yawns
            while (
                yawn_times
                and
                now - yawn_times[0]
                > YAWN_WINDOW
            ):

                yawn_times.popleft()

            # Current yawn count
            yawn_count = len(
                yawn_times
            )

            # ---------------------------------
            # FREQUENT YAWNING = HIGH ALERT
            # ---------------------------------

            if yawn_count >= YAWN_HIGH_COUNT:

                risk = max(
                    risk,
                    0.90
                )

                label = (
                    "high_impairment_risk"
                )

                reason = (
                    f"Frequent yawning "
                    f"({yawn_count} yawns "
                    f"in {YAWN_WINDOW}s)"
                )

            # ---------------------------------
            # HIGH ALERT
            # ---------------------------------

            alert.trigger(
                label,
                reason,
                risk
            )

            # ---------------------------------
            # DISPLAY
            # ---------------------------------

            put(
                frame,
                f"Risk: {risk:.2f} | {label}",
                30
            )

            put(
                frame,
                f"Speed: "
                f"{data['speed_kmph']:.1f} km/h",
                60
            )

            # EAR
            if f.get("ear") is not None:

                put(
                    frame,
                    f"EAR: "
                    f"{f['ear']:.3f}",
                    90
                )

            else:

                put(
                    frame,
                    "EAR: --",
                    90
                )

            # MAR
            if f.get("mar") is not None:

                put(
                    frame,
                    f"MAR: "
                    f"{f['mar']:.3f}",
                    120
                )

            else:

                put(
                    frame,
                    "MAR: --",
                    120
                )

            # Lane
            put(
                frame,
                f"Lane: {dev:.2f} | "
                f"Departure: {cross}",
                150
            )

            # rPPG
            if f.get("bpm"):

                put(
                    frame,
                    f"rPPG: "
                    f"{f['bpm']:.1f} BPM",
                    180
                )

            else:

                put(
                    frame,
                    "rPPG: collecting...",
                    180
                )

            # Explanation
            put(
                frame,
                reason,
                215
            )

            # Yawn counter
            put(
                frame,
                f"Yawns: "
                f"{yawn_count}/"
                f"{YAWN_HIGH_COUNT} "
                f"in {YAWN_WINDOW}s",
                245
            )

            # ---------------------------------
            # RISK BORDER + ALERT BANNER
            # ---------------------------------

            frame = alert.draw(
                frame,
                risk
            )

            # ---------------------------------
            # SHOW CAMERA
            # ---------------------------------

            cv2.imshow(
                "Driver Safety - "
                "Explainable Multimodal Prototype",
                frame
            )

            # ---------------------------------
            # KEYBOARD
            # ---------------------------------

            k = (
                cv2.waitKey(1)
                & 0xFF
            )

            # Q = quit
            if k == ord("q"):

                break

            # S = snapshot
            if k == ord("s"):

                filename = (
                    f"outputs/"
                    f"snapshot_"
                    f"{int(time.time())}.jpg"
                )

                cv2.imwrite(
                    filename,
                    frame
                )

                print(
                    "Snapshot saved:",
                    filename
                )

    finally:

        face.close()

        cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":

    main()