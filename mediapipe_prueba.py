import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils

mp_pose = mp.solutions.pose

# Set up the videocapture device
# 0 is the number asigned to the cam device
cap = cv2.VideoCapture(0)
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:  #static_image_mode: false
    while cap.isOpened():
        # Frame = gives us the image
        ret, frame = cap.read()

        # Recolor image to RGB (reordering RGB data)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        ###################################################
        # Make detection
        ###################################################
        results = pose.process(image)
        




        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               

        # Visualize the image
        cv2.imshow('Mediapipe Feed', image)

        # 10 = close window
        # or press 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release webcam
cap.release()

# Close windows
cv2.destroyAllWindows()