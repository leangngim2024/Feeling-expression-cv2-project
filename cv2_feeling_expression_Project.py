import cv2
import os

# Load Haar cascades
haar_path = cv2.data.haarcascades
face_cascade = cv2.CascadeClassifier(os.path.join(haar_path, 'haarcascade_frontalface_default.xml'))
smile_cascade = cv2.CascadeClassifier(os.path.join(haar_path, 'haarcascade_smile.xml'))

# Draw various emoji faces above the head
def draw_emoji_above_head(frame, x, y, w, h, emotion):
    radius = w // 2
    center_x = x + w // 2
    center_y = y - radius - 10  # Position above the head with small gap

    # Skip if emoji would be out of the frame
    if center_y - radius < 0:
        return

    # Yellow face circle
    cv2.circle(frame, (center_x, center_y), radius, (0, 255, 255), -1)

    # Eyes (always drawn)
    eye_radius = w // 15
    eye_y = center_y - h // 6
    cv2.circle(frame, (center_x - w // 6, eye_y), eye_radius, (0, 0, 0), -1)
    cv2.circle(frame, (center_x + w // 6, eye_y), eye_radius, (0, 0, 0), -1)

    # Define emotions
    if emotion == "smile":
        # Smiling mouth
        mouth_center = (center_x, center_y + h // 8)
        cv2.ellipse(frame, mouth_center, (w // 4, h // 10), 0, 0, 180, (0, 0, 0), 3)
    elif emotion == "sad":
        # Sad mouth
        mouth_center = (center_x, center_y + h // 8)
        cv2.line(frame, (center_x - w // 6, mouth_center[1] + 5),
                 (center_x + w // 6, mouth_center[1] + 5), (0, 0, 0), 3)
    elif emotion == "angry":
        # Angry mouth
        mouth_center = (center_x, center_y + h // 8)
        cv2.ellipse(frame, mouth_center, (w // 4, h // 10), 0, 180, 180, (0, 0, 0), 3)
    elif emotion == "love":
        # Love mouth (heart shape)
        mouth_center = (center_x, center_y + h // 8)
        cv2.ellipse(frame, mouth_center, (w // 6, h // 10), 0, 0, 180, (0, 0, 255), 3)
    elif emotion == "confuse":
        # Confused mouth (wiggle)
        mouth_center = (center_x, center_y + h // 8)
        cv2.line(frame, (center_x - w // 6, mouth_center[1]),
                 (center_x + w // 6, mouth_center[1] + 5), (0, 0, 0), 3)
    elif emotion == "surprise":
        # Surprised mouth (wide open)
        mouth_center = (center_x, center_y + h // 8)
        cv2.ellipse(frame, mouth_center, (w // 3, h // 5), 0, 0, 360, (0, 0, 0), 3)
    elif emotion == "cool":
        # Cool mouth (straight line)
        mouth_center = (center_x, center_y + h // 8)
        cv2.line(frame, (center_x - w // 4, mouth_center[1]),
                 (center_x + w // 4, mouth_center[1]), (0, 0, 0), 3)
    elif emotion == "wink":
        # Winking (one eye closed)
        eye_y = center_y - h // 6
        cv2.circle(frame, (center_x - w // 6, eye_y), eye_radius, (0, 0, 0), -1)
        cv2.circle(frame, (center_x + w // 6, eye_y), eye_radius, (0, 0, 0), -1)
        cv2.line(frame, (center_x + w // 6 - eye_radius, eye_y - eye_radius),
                 (center_x + w // 6 + eye_radius, eye_y + eye_radius), (0, 0, 0), 3)
    elif emotion == "tired":
        # Tired (drooping eyes)
        eye_y = center_y - h // 6
        cv2.circle(frame, (center_x - w // 6, eye_y), eye_radius, (0, 0, 0), -1)
        cv2.circle(frame, (center_x + w // 6, eye_y), eye_radius, (0, 0, 0), -1)
        cv2.line(frame, (center_x - w // 6, eye_y + 5),
                 (center_x + w // 6, eye_y + 5), (0, 0, 0), 3)

# Start webcam
cap = cv2.VideoCapture(0)

# Set resolution for webcam (adjust as needed for stability)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Frame rate control (set to 30 FPS)
frame_rate = 30
prev_frame_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Control frame rate
    current_time = cv2.getTickCount()
    time_elapsed = (current_time - prev_frame_time) / cv2.getTickFrequency()
    if time_elapsed < 1 / frame_rate:
        continue

    prev_frame_time = current_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)


        # Draw a square around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Example of detecting a smile (emotion can be dynamically assigned)
        is_smiling = len(smiles) > 0
        emotion = "smile" if is_smiling else "cool"  # Example: assign "smile" if smiling

        # Draw emoji based on emotion
        draw_emoji_above_head(frame, x, y, w, h, emotion)

    # Show resized frame
    cv2.imshow('Emoji Reaction Mirror', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
