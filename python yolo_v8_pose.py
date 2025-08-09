from ultralytics import YOLO
import cv2
import time

# Load model
model = YOLO("yolov8n-pose.pt")

# Video source
video_path = r"D:\Research\Promotion\8\dance.mp4"
cap = cv2.VideoCapture(video_path)

# Get FPS for sync
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Face keypoints to skip (nose, eyes, ears)
face_indices = {0, 1, 2, 3, 4}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO pose detection
    results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)

    # Get YOLO speed values
    speed_data = results[0].speed
    preprocess_ms = speed_data['preprocess']
    inference_ms = speed_data['inference']
    postprocess_ms = speed_data['postprocess']

    # Draw keypoints and full skeleton connections
    for r in results:
        keypoints = r.keypoints.xy  # Shape: (num_people, num_kpts, 2)

        for person in keypoints:
            person = person.cpu().numpy().astype(int)

            # Unpack keypoint coordinates
            LS = tuple(person[5])   # Left Shoulder
            RS = tuple(person[6])   # Right Shoulder
            LE = tuple(person[7])   # Left Elbow
            RE = tuple(person[8])   # Right Elbow
            LW = tuple(person[9])   # Left Wrist
            RW = tuple(person[10])  # Right Wrist
            LH = tuple(person[11])  # Left Hip
            RH = tuple(person[12])  # Right Hip
            LK = tuple(person[13])  # Left Knee
            RK = tuple(person[14])  # Right Knee
            LA = tuple(person[15])  # Left Ankle
            RA = tuple(person[16])  # Right Ankle

            # Arm connections
            cv2.line(frame, LS, LE, (0, 255, 0), 2)
            cv2.line(frame, LE, LW, (0, 255, 0), 2)
            cv2.line(frame, RS, RE, (0, 255, 0), 2)
            cv2.line(frame, RE, RW, (0, 255, 0), 2)

            # Torso connections
            cv2.line(frame, LS, RS, (255, 255, 0), 2)  # Shoulders
            cv2.line(frame, LH, RH, (255, 255, 0), 2)  # Hips
            cv2.line(frame, LS, LH, (255, 0, 255), 2)  # Left side
            cv2.line(frame, RS, RH, (255, 0, 255), 2)  # Right side

            # Leg connections
            cv2.line(frame, LH, LK, (255, 0, 0), 2)
            cv2.line(frame, LK, LA, (255, 0, 0), 2)
            cv2.line(frame, RH, RK, (255, 0, 0), 2)
            cv2.line(frame, RK, RA, (255, 0, 0), 2)

            # Draw all keypoints except face ones
            for idx, (x, y) in enumerate(person):
                if idx not in face_indices:
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    # Overlay model name and detection speeds in black
    cv2.putText(frame, "yolov8n-pose", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, f"Pre: {preprocess_ms:.1f}ms  Inf: {inference_ms:.1f}ms  Post: {postprocess_ms:.1f}ms",
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Show frame
    cv2.imshow("Pose Detection", frame)

    # Keep original video speed
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
