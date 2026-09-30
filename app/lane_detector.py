import cv2
import numpy as np


def detect_lane_departure(frame):
    """
    Clean lane detection for the Driver Safety prototype.

    Returns:
        output     : processed frame
        deviation : normalized lane deviation (-1 to +1)
        crossing   : True if significant lane departure is detected
    """

    output = frame.copy()

    height, width = frame.shape[:2]

    # ---------------------------------------------------------
    # 1. Convert to grayscale and detect edges
    # ---------------------------------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 60, 150)

    # ---------------------------------------------------------
    # 2. Region of Interest
    #    Only look at the road area
    # ---------------------------------------------------------
    mask = np.zeros_like(edges)

    polygon = np.array([[
        (int(width * 0.10), height),
        (int(width * 0.42), int(height * 0.58)),
        (int(width * 0.58), int(height * 0.58)),
        (int(width * 0.90), height)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)

    roi = cv2.bitwise_and(edges, mask)

    # ---------------------------------------------------------
    # 3. Detect lines
    # ---------------------------------------------------------
    lines = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi / 180,
        threshold=55,
        minLineLength=50,
        maxLineGap=40
    )

    left_lines = []
    right_lines = []

    if lines is not None:

        for line in lines:
            x1, y1, x2, y2 = np.asarray(line).reshape(-1)[:4]

            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            dx = x2 - x1
            dy = y2 - y1

            if dx == 0:
                continue

            slope = dy / float(dx)

            # Remove horizontal/noisy lines
            if abs(slope) < 0.45:
                continue

            # Remove extremely steep/noisy lines
            if abs(slope) > 3.5:
                continue

            # Ignore lines too high in the image
            if max(y1, y2) < int(height * 0.60):
                continue

            # Left and right classification
            if slope < 0:
                left_lines.append((x1, y1, x2, y2))
            else:
                right_lines.append((x1, y1, x2, y2))

    # ---------------------------------------------------------
    # 4. Average each lane boundary
    # ---------------------------------------------------------
    def average_line(line_list):

        if not line_list:
            return None

        slopes = []
        intercepts = []

        for x1, y1, x2, y2 in line_list:

            if x2 == x1:
                continue

            slope = (y2 - y1) / float(x2 - x1)

            intercept = y1 - slope * x1

            slopes.append(slope)
            intercepts.append(intercept)

        if not slopes:
            return None

        return float(np.mean(slopes)), float(np.mean(intercepts))

    left_lane = average_line(left_lines)
    right_lane = average_line(right_lines)

    # ---------------------------------------------------------
    # 5. Calculate lane positions
    # ---------------------------------------------------------
    target_y = int(height * 0.88)

    left_x = None
    right_x = None

    if left_lane is not None:
        slope, intercept = left_lane

        if abs(slope) > 0.01:
            left_x = (target_y - intercept) / slope

    if right_lane is not None:
        slope, intercept = right_lane

        if abs(slope) > 0.01:
            right_x = (target_y - intercept) / slope

    # ---------------------------------------------------------
    # 6. Validate lane boundaries
    # ---------------------------------------------------------
    lane_center = width / 2.0

    if left_x is not None and right_x is not None:

        # Valid lane width
        lane_width = right_x - left_x

        if width * 0.20 < lane_width < width * 0.90:

            lane_center = (left_x + right_x) / 2.0

            # Draw ONLY the averaged lane boundaries
            left_top_y = int(height * 0.60)
            right_top_y = int(height * 0.60)

            left_top_x = int(
                (left_top_y - left_lane[1]) / left_lane[0]
            )

            right_top_x = int(
                (right_top_y - right_lane[1]) / right_lane[0]
            )

            # Keep coordinates inside image
            left_top_x = int(np.clip(left_top_x, 0, width - 1))
            right_top_x = int(np.clip(right_top_x, 0, width - 1))

            left_x_draw = int(np.clip(left_x, 0, width - 1))
            right_x_draw = int(np.clip(right_x, 0, width - 1))

            # Draw clean lane boundaries
            cv2.line(
                output,
                (left_top_x, left_top_y),
                (left_x_draw, target_y),
                (0, 255, 0),
                4
            )

            cv2.line(
                output,
                (right_top_x, right_top_y),
                (right_x_draw, target_y),
                (0, 255, 0),
                4
            )

            # Draw lane center
            cv2.line(
                output,
                (int(lane_center), int(height * 0.60)),
                (int(lane_center), target_y),
                (255, 255, 0),
                3
            )

    # ---------------------------------------------------------
    # 7. Calculate deviation
    # ---------------------------------------------------------
    deviation = (
        (lane_center - width / 2.0)
        / (width / 2.0)
    )

    deviation = float(np.clip(deviation, -1.0, 1.0))

    # ---------------------------------------------------------
    # 8. Lane departure
    # ---------------------------------------------------------
    crossing = abs(deviation) > 0.35

    # ---------------------------------------------------------
    # 9. Draw camera center
    # ---------------------------------------------------------
    cv2.line(
        output,
        (width // 2, int(height * 0.75)),
        (width // 2, height),
        (255, 255, 0),
        2
    )

    # ---------------------------------------------------------
    # 10. Draw lane center point
    # ---------------------------------------------------------
    cv2.circle(
        output,
        (int(lane_center), target_y),
        8,
        (0, 0, 255) if crossing else (0, 255, 0),
        -1
    )

    # ---------------------------------------------------------
    # 11. Status text
    # ---------------------------------------------------------
    if crossing:
        status = "LANE DEPARTURE"
        status_color = (0, 0, 255)
    else:
        status = "LANE OK"
        status_color = (0, 255, 0)

    cv2.putText(
        output,
        status,
        (20, height -25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        status_color,
        2,
        cv2.LINE_AA
    )

    return output, deviation, crossing