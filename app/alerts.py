import time
import winsound
import cv2


class AlertManager:

    def __init__(self):

        self.last_alert = 0

        # Minimum time between sounds
        self.cooldown = 3

    def trigger(
        self,
        label,
        reason,
        risk
    ):

        # Only high risk produces
        # sound/alert
        if risk < 0.70:
            return

        now = time.time()

        # Prevent continuous sound
        if now - self.last_alert < self.cooldown:
            return

        print()
        print(
            "===================================="
        )
        print(
            "       HIGH DRIVER RISK ALERT"
        )
        print(
            "===================================="
        )

        print(
            "Condition:",
            label
        )

        print(
            "Reason:",
            reason
        )

        print(
            "Risk:",
            round(risk, 2)
        )

        print(
            "===================================="
        )

        # Windows warning sound
        winsound.PlaySound(
            "SystemExclamation",
            winsound.SND_ALIAS |
            winsound.SND_ASYNC
        )

        self.last_alert = now

    def draw(
        self,
        frame,
        risk
    ):

        # ---------------------------------
        # SAFE
        # ---------------------------------

        if risk < 0.40:

            color = (
                0,
                255,
                0
            )

        # ---------------------------------
        # MODERATE RISK
        # ---------------------------------

        elif risk < 0.70:

            color = (
                0,
                255,
                255
            )

        # ---------------------------------
        # HIGH RISK
        # ---------------------------------

        else:

            color = (
                0,
                0,
                255
            )

        # Thin border
        cv2.rectangle(
            frame,
            (3, 3),
            (
                frame.shape[1] - 4,
                frame.shape[0] - 4
            ),
            color,
            3
        )

        # ---------------------------------
        # HIGH ALERT BANNER
        # ---------------------------------

        if risk >= 0.70:

            cv2.rectangle(
                frame,
                (0, 0),
                (
                    frame.shape[1],
                    40
                ),
                (0, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                "!!! HIGH DRIVER RISK ALERT !!!",
                (20, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        return frame