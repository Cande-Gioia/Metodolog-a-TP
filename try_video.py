import os 
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
#path = 'Videos Jugadores Profesionales\Real Madrid 3 corto.mp4' 
path = 'Videos Bruno\\bruno_30grad_cortado.mp4' 

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
Cristiano Ronaldo 1 (al principio no hay luz y después va re lento, se vuelve a romper)
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

#point1 es el de izquierda 
#point 2 es el de derecha
#el ángulo lo da en sentido horario
def angles_calc(point1, center, point2):
     vector1 = [point1[0] - center[0], point1[1] - center[1]]
     vector2 = [point2[0] - center[0], point2[1] - center[1]]
     mod_vector1 = np.sqrt(vector1[0] * vector1[0] + vector1[1]*vector1[1] )
     mod_vector2 = np.sqrt(vector2[0] * vector2[0] + vector2[1]*vector2[1] )
     cos_angle = (vector1[0] * vector2[0] + vector1[1]*vector2[1])/(  mod_vector1 * mod_vector2 )
     angle = np.arccos(cos_angle)
     return angle/(np.pi) * 180

#no funciona el barcelona 3, tampoco 2 y 1 tampoco, real madrid 1 se rompe también, el 2 agarra en el íultimo frame valores de las estatuas,
# el 4 lo toma después y al final toma la estatua. 
#
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
i = 0


angle_A = []
angle_B = []
angle_C = []
video = cv2.VideoCapture(path)

## Setup mediapipe instance
with mp_pose.Pose(static_image_mode = False, smooth_landmarks = True, min_detection_confidence = 0.5, min_tracking_confidence = 0.5, model_complexity = 2) as pose:
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

                window_name = 'Mi Ventana'
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

                cv2.resizeWindow(window_name, 800, 600)
                cv2.imshow(window_name, image)
                # Get coordinates
                
                if(results.pose_landmarks != None):
                    landmarks = results.pose_landmarks.landmark
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
                    angle_A.append(angles_calc(right_shoulder, right_hip, right_knee))
                    angle_B.append(angles_calc(right_hip, right_knee, right_ankle))
                    angle_C.append(angles_calc(right_knee, right_ankle, right_foot_index))
                    print('Frame:',i)
                    print('Right Shoulder:',right_shoulder)
                    print('Right Hip:',right_hip)
                    print('Right Knee:',right_knee)
                    print('Right Ankle:',right_ankle)
                    print('Right Foot:',right_foot_index)
                    
                    
                    x = [right_shoulder[0], right_hip[0], right_knee[0], right_ankle[0], right_foot_index[0]]
                    y = [-right_shoulder[1], -right_hip[1], -right_knee[1], -right_ankle[1], -right_foot_index[1]]
                    plt.scatter(x, y, color = 'black')
                    plt.gca().set_aspect(aspect = 9/16, adjustable=None, anchor=None, share=False)
                    plt.show()
                    while(cv2.waitKey(10) & 0xFF == ord('q')):
                        break
                 
            else:
                 break
          
           

# Release webcam
video.release()

# Close windows
cv2.destroyAllWindows()


t = np.arange(1, len(angle_A) + 1, step = 1 )

plt.plot(t, angle_A, color = 'black')
plt.plot(t, angle_B, color = 'blue')
plt.plot(t, angle_C, color = 'red')
plt.show()


