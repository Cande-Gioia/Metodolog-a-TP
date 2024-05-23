import os 
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

path = 'Videos Bruno\\Bruno 2.mp4' 

# Point1 es el de izquierda 
# Point 2 es el de derecha
# El ángulo lo da en sentido horario
def angles_calc(point1, center, point2):
    vector1 = [point1[0] - center[0], point1[1] - center[1]]
    vector2 = [point2[0] - center[0], point2[1] - center[1]]
    mod_vector1 = np.sqrt(vector1[0] * vector1[0] + vector1[1]*vector1[1] )
    mod_vector2 = np.sqrt(vector2[0] * vector2[0] + vector2[1]*vector2[1] )
    cos_angle = (vector1[0] * vector2[0] + vector1[1]*vector2[1]) / (mod_vector1 * mod_vector2)

    angle = np.arccos(cos_angle)

    angle_deg = angle * 180 / np.pi

     # Determinar el signo del ángulo usando producto cruzado
    cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]
    if cross_product < 0:
        angle_deg = 360 - angle_deg

    return angle_deg


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
i = 0

angle_A = []
angle_B = []
angle_C = []
video = cv2.VideoCapture(path)

fps = 240

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
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,-landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,-landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,-landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,-landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,-landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
                    
                    angle_A.append(angles_calc(right_shoulder, right_hip, right_knee))
                    angle_B.append(angles_calc(right_hip, right_knee, right_ankle))
                    angle_C.append(angles_calc(right_foot_index, right_ankle, right_knee))
                    
                    '''print('Frame:',i)
                    print('Right Shoulder:',right_shoulder)
                    print('Right Hip:',right_hip)
                    print('Right Knee:',right_knee)
                    print('Right Ankle:',right_ankle)
                    print('Right Foot:',right_foot_index)'''
                    
                    
                    '''
                    if (i == 20):
                        x = [right_shoulder[0], right_hip[0], right_knee[0], right_ankle[0], right_foot_index[0]]
                        y = [right_shoulder[1], right_hip[1], right_knee[1], right_ankle[1], right_foot_index[1]]
                        plt.scatter(x, y, color = 'black')
                        plt.gca().set_aspect(aspect = 9/16, adjustable=None, anchor=None, share=False)
                        plt.show()
                        while(not (cv2.waitKey(10) & 0xFF == ord('q'))):
                            print(angles_calc(right_shoulder, right_hip, right_knee)) 
                            pass
                    '''        
                    '''while(not cv2.waitKey(10) & 0xFF == ord('q')):
                         pass'''
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                 
            else:
                 break
          
           
# Release webcam
video.release()

# Close windows
cv2.destroyAllWindows()

'''
angle_A1 = savgol_filter(angle_A, window_length=10, polyorder=1)
angle_B1 = savgol_filter(angle_B, window_length=10, polyorder=1)
angle_C1 = savgol_filter(angle_C, window_length=10, polyorder=1)
'''
angle_A = savgol_filter(angle_A, window_length=20, polyorder=1)
angle_B = savgol_filter(angle_B, window_length=20, polyorder=1)
angle_C = savgol_filter(angle_C, window_length=20, polyorder=1)


vel_angle_A = [(angle_A[i] - angle_A[i-1]) * fps for i in range(1, len(angle_A))]
vel_angle_B = [(angle_B[i] - angle_B[i-1]) * fps for i in range(1, len(angle_B))]
vel_angle_C = [(angle_C[i] - angle_C[i-1]) * fps for i in range(1, len(angle_C))]

'''
vel_angle_A1 = [(angle_A1[i] - angle_A1[i-1]) * fps for i in range(1, len(angle_A1))]
vel_angle_B1 = [(angle_B1[i] - angle_B1[i-1]) * fps for i in range(1, len(angle_B1))]
vel_angle_C1 = [(angle_C1[i] - angle_C1[i-1]) * fps for i in range(1, len(angle_C1))]


vel_angle_A1 = savgol_filter(vel_angle_A1, window_length=20, polyorder=2)
vel_angle_B1 = savgol_filter(vel_angle_B1, window_length=20, polyorder=2)
vel_angle_C1 = savgol_filter(vel_angle_C1, window_length=20, polyorder=2)
'''

vel_angle_A = savgol_filter(vel_angle_A, window_length=20, polyorder=1)
vel_angle_B = savgol_filter(vel_angle_B, window_length=20, polyorder=1)
vel_angle_C = savgol_filter(vel_angle_C, window_length=20, polyorder=1)


t = np.arange(1, len(angle_A) + 1, step = 1 )

plt.plot(t, angle_A, color = 'black', label='Cadera')
plt.plot(t, angle_B, color = 'black', label='Rodilla')
#plt.plot(t, angle_C, color = 'red', label="Tobillo")
'''
plt.plot(t, angle_A1, color = 'blue', label='Cadera')
plt.plot(t, angle_B1, color = 'blue', label='Rodilla')
'''
plt.xlabel("Frame")
plt.ylabel("Ángulo [°]")
plt.legend()

plt.grid()

plt.show()

t = np.arange(1, len(vel_angle_C) + 1, step = 1 )

plt.plot(t, vel_angle_A, color = 'black', label='Cadera')
plt.plot(t, vel_angle_B, color = 'blue', label='Rodilla')
#plt.plot(t, vel_angle_C, color = 'red', label="Tobillo")

'''plt.plot(t, vel_angle_A1, color = 'blue', label='Cadera')
plt.plot(t, vel_angle_B1, color = 'blue', label='Rodilla')
'''
plt.xlabel("Frame")
plt.ylabel("Ángulo [°]")
plt.legend()

plt.grid()

plt.show()