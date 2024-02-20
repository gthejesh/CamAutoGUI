import cv2
import mediapipe as mp
import pyautogui as gui

# Initialize Mediapipe FaceMesh module
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open a connection to the webcam (you may need to change the index based on your setup)
cap = cv2.VideoCapture(1)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe FaceMesh
    results = face_mesh.process(rgb_frame)

    # Check if face landmarks are present
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            # Extract the nose landmark
            nose_landmark = landmarks.landmark[2]  # Index for the nose
            h, w, c = frame.shape
            nose_x, nose_y = int(nose_landmark.x * w), int(nose_landmark.y * h)

            # Print the coordinates of the nose position
            print(f"Nose Position: ({nose_x}, {nose_y})")
            gui.moveTo(nose_x,nose_y)

            # Draw landmarks on the face
            for landmark in landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), 1)

    # Display the result
    cv2.imshow('Nose Tracking', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
