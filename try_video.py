import os 
import cv2
import mediapipe as mp
import numpy as np
path = 'Videos Jugadores Profesionales\Kevin De Bruyne - tiro Libre.mp4' 


#NO FUNCIONAN:
'''
barcelona 1
barcelona 2
barcelona 3
real madrid 1
real madrid 7
real madrid 9
juninho ( va tan lento que un momento se rompe)

'''
#FUNCIONAN EN TIEMPO DADO:
'''
real madrid 2 (agarra al final al estatua)
real madrid 4 (al principio y al final se rompe)
real madrid 6 (al final)
real madrid 8 (al comienzo)
real madrid 10 (al comienzo y al final)
Kevin De Bruyne (al principio CORTARLO DE NUEVO)
Cristiano Ronald 1 (al principio no hay luz y después va re lento, se vuelve a romper)
'''

#FUNCIONAN:
'''
real madrid 5
real madrid 11
riquelme-recorte
cristiano 2
inker 
Beckham
'''
#no funciona el barcelona 3, tampoco 2 y 1 tampoco, real madrid 1 se rompe también, el 2 agarra en el íultimo frame valores de las estatuas,
# el 4 lo toma después y al final toma la estatua. 
#
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
i = 0
    
video = cv2.VideoCapture(path)

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while video.isOpened():
            i = i +1
            ret, frame = video.read()

            if(ret):
                 
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                # Make detection
                results = pose.process(image)
            
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )          

                            
            
                cv2.imshow('Mediapipe Feed', image)
                # Get coordinates
                
                if(results.pose_landmarks != None):
                    landmarks = results.pose_landmarks.landmark
                
                
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
                    print('Frame:',i)
                    print('Right Shoulder:',right_shoulder)
                    print('Right Hip:',right_hip)
                    print('Right Knee:',right_knee)
                    print('Right Ankle:',right_ankle)
                    print('Right Foot:',right_foot_index)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                
            else:
                 break
          
           

# Release webcam
video.release()

# Close windows
cv2.destroyAllWindows()

