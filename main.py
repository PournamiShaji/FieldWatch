import cv2
import winsound

# ---------------- ALERT SOUND ----------------
def play_alert_sound():
    for _ in range(3):
        winsound.Beep(1000, 400)

# ---------------- LIVE CAMERA ----------------
cap = cv2.VideoCapture(0)  # LIVE webcam feed

if not cap.isOpened():
    print("❌ Webcam not opened")
    exit()

print("✅ Live webcam feed started")

# ---------------- MOTION DETECTION SETUP ----------------
first_frame = None
alert_triggered = False

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale & blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Initialize background model
    if first_frame is None:
        first_frame = gray.copy().astype("float")
        continue

    # Update background slowly
    cv2.accumulateWeighted(gray, first_frame, 0.01)

    # Compute difference
    frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(first_frame))

    # Threshold for motion
    thresh = cv2.threshold(frame_delta, 15, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(
        thresh.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    animal_detected = False

    # Process contours
    for contour in contours:
        if cv2.contourArea(contour) < 1000:  # sensitivity
            continue

        animal_detected = True
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Alert logic
    if animal_detected:
        cv2.putText(
            frame,
            "⚠️ ANIMAL DETECTED ⚠️",
            (40, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            4
        )

        if not alert_triggered:
            play_alert_sound()
            alert_triggered = True
    else:
        alert_triggered = False

    # Show live feed
    cv2.imshow("FieldWatch - Live Feed", frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
