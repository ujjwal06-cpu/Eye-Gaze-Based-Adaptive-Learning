import cv2
import mediapipe as mp
import numpy as np
import time
import math


# ============================================================
# SETTINGS
# ============================================================

DWELL_TIME = 1.5
CAMERA_ID = 0


# ============================================================
# MEDIAPIPE
# ============================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("ERROR: Webcam could not be opened.")
    print("Check System Settings -> Privacy & Security -> Camera.")
    exit()

print()
print("======================================")
print("       EYE GAZE INTERFACE")
print("======================================")
print("Webcam started.")
print("Look at an option for 1.5 seconds.")
print("Press Q to quit.")
print()


# ============================================================
# OPTIONS
# ============================================================

OPTIONS = [
    "OPTION 1",
    "OPTION 2",
    "OPTION 3",
    "OPTION 4",
    "EXIT"
]


# ============================================================
# GAZE FUNCTION
# ============================================================

def calculate_gaze(landmarks):

    # LEFT EYE

    left_outer = landmarks[33]
    left_inner = landmarks[133]

    left_top = landmarks[159]
    left_bottom = landmarks[145]

    left_iris = np.mean(
        [
            [landmarks[i].x, landmarks[i].y]
            for i in range(468, 473)
        ],
        axis=0
    )


    # RIGHT EYE

    right_outer = landmarks[362]
    right_inner = landmarks[263]

    right_top = landmarks[386]
    right_bottom = landmarks[374]

    right_iris = np.mean(
        [
            [landmarks[i].x, landmarks[i].y]
            for i in range(473, 478)
        ],
        axis=0
    )


    # --------------------------------------------------------
    # LEFT EYE
    # --------------------------------------------------------

    left_x1 = left_outer.x
    left_x2 = left_inner.x

    left_y1 = left_top.y
    left_y2 = left_bottom.y


    left_gaze_x = (
        left_iris[0] - left_x1
    ) / (
        left_x2 - left_x1 + 1e-6
    )


    left_gaze_y = (
        left_iris[1] - left_y1
    ) / (
        left_y2 - left_y1 + 1e-6
    )


    # --------------------------------------------------------
    # RIGHT EYE
    # --------------------------------------------------------

    right_x1 = right_outer.x
    right_x2 = right_inner.x

    right_y1 = right_top.y
    right_y2 = right_bottom.y


    right_gaze_x = (
        right_iris[0] - right_x1
    ) / (
        right_x2 - right_x1 + 1e-6
    )


    right_gaze_y = (
        right_iris[1] - right_y1
    ) / (
        right_y2 - right_y1 + 1e-6
    )


    # --------------------------------------------------------
    # AVERAGE BOTH EYES
    # --------------------------------------------------------

    gaze_x = (
        left_gaze_x +
        right_gaze_x
    ) / 2


    gaze_y = (
        left_gaze_y +
        right_gaze_y
    ) / 2


    gaze_x = float(
        np.clip(gaze_x, 0, 1)
    )

    gaze_y = float(
        np.clip(gaze_y, 0, 1)
    )


    # --------------------------------------------------------
    # BLINK
    # --------------------------------------------------------

    left_eye_height = abs(
        left_bottom.y -
        left_top.y
    )

    right_eye_height = abs(
        right_bottom.y -
        right_top.y
    )


    left_eye_width = abs(
        left_inner.x -
        left_outer.x
    )

    right_eye_width = abs(
        right_inner.x -
        right_outer.x
    )


    left_ratio = (
        left_eye_height /
        (left_eye_width + 1e-6)
    )

    right_ratio = (
        right_eye_height /
        (right_eye_width + 1e-6)
    )


    eye_openness = (
        left_ratio +
        right_ratio
    ) / 2


    blink = eye_openness < 0.16


    return (
        gaze_x,
        gaze_y,
        eye_openness,
        blink
    )


# ============================================================
# OPTION DETECTION
# ============================================================

def get_option(x, y):

    if y < 0.45:

        if x < 0.5:
            return "OPTION 1"
        else:
            return "OPTION 2"


    elif y < 0.70:

        if x < 0.5:
            return "OPTION 3"
        else:
            return "OPTION 4"


    else:

        if 0.30 < x < 0.70:
            return "EXIT"

    return None


# ============================================================
# DRAW GRID
# ============================================================

def draw_grid(
    frame,
    current_option,
    progress
):

    h, w = frame.shape[:2]


    boxes = {

        "OPTION 1": (
            30,
            60,
            w // 2 - 10,
            h // 3
        ),

        "OPTION 2": (
            w // 2 + 10,
            60,
            w - 30,
            h // 3
        ),

        "OPTION 3": (
            30,
            h // 3 + 15,
            w // 2 - 10,
            2 * h // 3 - 15
        ),

        "OPTION 4": (
            w // 2 + 10,
            h // 3 + 15,
            w - 30,
            2 * h // 3 - 15
        ),

        "EXIT": (
            w // 3,
            2 * h // 3 + 15,
            2 * w // 3,
            h - 40
        )
    }


    for option, box in boxes.items():

        x1, y1, x2, y2 = box


        thickness = 3

        if option == current_option:
            thickness = 8


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            thickness
        )


        text_size = cv2.getTextSize(
            option,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            2
        )[0]


        tx = x1 + (
            x2 - x1 -
            text_size[0]
        ) // 2


        ty = y1 + (
            y2 - y1 +
            text_size[1]
        ) // 2


        cv2.putText(
            frame,
            option,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


    # --------------------------------------------------------
    # PROGRESS BAR
    # --------------------------------------------------------

    if current_option:

        bar_width = int(w * 0.6)

        bar_x = (
            w - bar_width
        ) // 2

        bar_y = h - 20


        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (
                bar_x + bar_width,
                bar_y + 12
            ),
            (100, 100, 100),
            -1
        )


        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (
                bar_x +
                int(bar_width * progress),
                bar_y + 12
            ),
            (0, 255, 0),
            -1
        )


# ============================================================
# VARIABLES
# ============================================================

current_option = None

fixation_start = None

last_gaze = None

last_time = time.time()

last_print = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam.")
        break


    frame = cv2.flip(frame, 1)


    h, w = frame.shape[:2]


    # --------------------------------------------------------
    # MEDIAPIPE
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    result = face_mesh.process(rgb)


    gaze_x = None
    gaze_y = None

    eye_openness = None

    blink = False

    face_detected = False

    gaze_velocity = 0.0

    fixation_duration = 0.0

    confidence = 0.0


    # --------------------------------------------------------
    # FACE
    # --------------------------------------------------------

    if result.multi_face_landmarks:

        face_detected = True


        landmarks = (
            result.multi_face_landmarks[0]
            .landmark
        )


        (
            gaze_x,
            gaze_y,
            eye_openness,
            blink
        ) = calculate_gaze(
            landmarks
        )


        now = time.time()


        # ----------------------------------------------------
        # GAZE VELOCITY
        # ----------------------------------------------------

        if last_gaze is not None:

            dx = (
                gaze_x -
                last_gaze[0]
            )

            dy = (
                gaze_y -
                last_gaze[1]
            )


            distance = math.sqrt(
                dx * dx +
                dy * dy
            )


            dt = now - last_time


            if dt > 0:

                gaze_velocity = (
                    distance / dt
                )


        last_gaze = (
            gaze_x,
            gaze_y
        )


        last_time = now


        # ----------------------------------------------------
        # BLINK
        # ----------------------------------------------------

        if not blink:

            detected_option = get_option(
                gaze_x,
                gaze_y
            )


            # ------------------------------------------------
            # NEW OPTION
            # ------------------------------------------------

            if detected_option != current_option:

                current_option = (
                    detected_option
                )


                if current_option:

                    fixation_start = now

                else:

                    fixation_start = None


            # ------------------------------------------------
            # FIXATION
            # ------------------------------------------------

            if (
                current_option
                and fixation_start
            ):

                fixation_duration = (
                    now -
                    fixation_start
                )


            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            stability = max(
                0,
                1 -
                min(
                    gaze_velocity,
                    1
                )
            )


            fixation_confidence = min(
                fixation_duration /
                DWELL_TIME,
                1
            )


            confidence = (
                0.5 * stability +
                0.5 *
                fixation_confidence
            )


            # ------------------------------------------------
            # SELECTION
            # ------------------------------------------------

            if (
                current_option
                and fixation_duration >=
                DWELL_TIME
            ):

                print()
                print(
                    "================================"
                )

                print(
                    "       SELECTION DETECTED"
                )

                print(
                    "================================"
                )

                print(
                    "Option       :",
                    current_option
                )

                print(
                    "X coordinate :",
                    round(gaze_x, 3)
                )

                print(
                    "Y coordinate :",
                    round(gaze_y, 3)
                )

                print(
                    "Confidence   :",
                    round(confidence, 3)
                )

                print(
                    "Blink        :",
                    blink
                )

                print(
                    "Fixation     :",
                    round(
                        fixation_duration,
                        3
                    )
                )

                print(
                    "Gaze velocity:",
                    round(
                        gaze_velocity,
                        3
                    )
                )

                print(
                    "Eye openness :",
                    round(
                        eye_openness,
                        3
                    )
                )

                print(
                    "================================"
                )


                if current_option == "EXIT":
                    break


                # Prevent immediate repeated selection

                fixation_start = now + 1.0


    # ========================================================
    # FIXATION PROGRESS
    # ========================================================

    if (
        fixation_start is not None
        and fixation_start <= time.time()
    ):

        fixation_duration = (
            time.time() -
            fixation_start
        )


        progress = min(
            fixation_duration /
            DWELL_TIME,
            1
        )

    else:

        progress = 0


    # ========================================================
    # DRAW GAZE
    # ========================================================

    if gaze_x is not None:

        px = int(
            gaze_x * w
        )

        py = int(
            gaze_y * h
        )


        cv2.circle(
            frame,
            (px, py),
            12,
            (0, 255, 0),
            -1
        )


    # ========================================================
    # DRAW GRID
    # ========================================================

    draw_grid(
        frame,
        current_option,
        progress
    )


    # ========================================================
    # STATUS
    # ========================================================

    if not face_detected:

        status = "NO FACE"

    elif blink:

        status = "BLINK"

    elif current_option:

        status = (
            "LOOKING: " +
            current_option
        )

    else:

        status = "LOOK AT AN OPTION"


    cv2.putText(
        frame,
        status,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # VECTOR OUTPUT
    # ========================================================

    now = time.time()


    if now - last_print >= 0.2:

        vector = [

            None if gaze_x is None
            else round(gaze_x, 3),

            None if gaze_y is None
            else round(gaze_y, 3),

            current_option,

            round(confidence, 3),

            bool(blink),

            round(fixation_duration, 3),

            round(gaze_velocity, 3),

            None if eye_openness is None
            else round(
                float(eye_openness),
                3
            ),

            face_detected,

            round(now, 3)
        ]


        print(
            "VECTOR:",
            vector
        )


        last_print = now


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        "Eye Gaze Interface",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

face_mesh.close()

print()
print("Eye gaze interface stopped.")
