import os 
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from dtaidistance import dtw



# Point1 es el de izquierda 
# Point 2 es el de derecho
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




def get_points(path):

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    i = 0

    angle_A = []
    angle_B = []
    angle_C = []


    right_shoulder_x = []
    right_shoulder_y = []

    right_hip_x = []
    right_hip_y = []

    right_knee_x = []
    right_knee_y = []

    right_ankle_x = []
    right_ankle_y = []

    right_foot_index_x = []
    right_foot_index_y = []

    video = cv2.VideoCapture(path)


    ## Setup mediapipe instance
    with mp_pose.Pose(static_image_mode = False, smooth_landmarks = True, min_detection_confidence = 0.4, min_tracking_confidence = 0.7, model_complexity = 2) as pose:
        while video.isOpened():
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
                        right_shoulder_x.append(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x)
                        right_shoulder_y.append(-landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
                        right_hip_x.append( landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x)
                        right_hip_y.append(-landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y)
                        right_knee_x.append(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x)
                        right_knee_y.append(-landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y)
                        right_ankle_x.append(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x)
                        right_ankle_y.append(-landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)
                        right_foot_index_x.append(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x)
                        right_foot_index_y.append(-landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y)
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
                i = i +1
    
            
    # Release webcam
    video.release()

    # Close windows
    cv2.destroyAllWindows()

    right_shoulder_x = savgol_filter(right_shoulder_x, window_length=20, polyorder=1)
    right_shoulder_y = savgol_filter(right_shoulder_y, window_length=20, polyorder=1)
    right_hip_x = savgol_filter(right_hip_x, window_length=20, polyorder=1)
    right_hip_y = savgol_filter(right_hip_y, window_length=20, polyorder=1) 
    right_ankle_x = savgol_filter(right_ankle_x, window_length=20, polyorder=1) 
    right_ankle_y = savgol_filter(right_ankle_y, window_length=20, polyorder=1) 
    right_knee_x = savgol_filter(right_knee_x, window_length=20, polyorder=1) 
    right_knee_y = savgol_filter(right_knee_y, window_length=20, polyorder=1) 
    right_foot_index_x = savgol_filter(right_foot_index_x, window_length=20, polyorder=1) 
    right_foot_index_y = savgol_filter(right_foot_index_y, window_length=20, polyorder=1) 

    for i in range(0,len(right_shoulder_x)):
        
        angle_A.append( angles_calc([right_shoulder_x[i], right_shoulder_y[i]], [right_hip_x[i],right_hip_y[i]],[right_knee_x[i], right_knee_y[i]]))
        angle_B.append(angles_calc([right_hip_x[i],right_hip_y[i]],[right_knee_x[i], right_knee_y[i]], [right_ankle_x[i], right_ankle_y[i]]))
        angle_C.append(angles_calc([right_foot_index_x[i], right_foot_index_y[i] ],[right_ankle_x[i], right_ankle_y[i]], [right_knee_x[i], right_knee_y[i]]))
                    


    angle_A = savgol_filter(angle_A, window_length=20, polyorder=1)
    angle_B = savgol_filter(angle_B, window_length=20, polyorder=1)
    angle_C = savgol_filter(angle_C, window_length=20, polyorder=1)

    return angle_A, angle_B, angle_C



def get_vel_angular( angle_A, angle_B, angle_C, fps):
    vel_angle_A = [(angle_A[i] - angle_A[i-1]) * fps for i in range(1, len(angle_A))]
    vel_angle_B = [(angle_B[i] - angle_B[i-1]) * fps for i in range(1, len(angle_B))]
    vel_angle_C = [(angle_C[i] - angle_C[i-1]) * fps for i in range(1, len(angle_C))]

    vel_angle_A = savgol_filter(vel_angle_A, window_length=20, polyorder=1)
    vel_angle_B = savgol_filter(vel_angle_B, window_length=20, polyorder=1)
    vel_angle_C = savgol_filter(vel_angle_C, window_length=20, polyorder=1)

    return vel_angle_A, vel_angle_B, vel_angle_C





path_1 = 'Videos Bruno\\Bruno 3m0.mp4'
path_2 = 'Videos Bruno\\Bruno -3m67_5.mp4'
angle_A_1, angle_B_1, angle_C_1 = get_points(path_1)
angle_A_2, angle_B_2, angle_C_2 = get_points(path_2)


vel_angle_A_1, vel_angle_B_1, vel_angle_C_1 =  get_vel_angular( angle_A_1, angle_B_1, angle_C_1, 240)
vel_angle_A_2, vel_angle_B_2, vel_angle_C_2 =  get_vel_angular( angle_A_2, angle_B_2, angle_C_2, 240)


dist_angle_A= dtw.distance(angle_A_1,angle_A_2)
dist_angle_B= dtw.distance(angle_B_1,angle_B_2)
dist_angle_C= dtw.distance(angle_C_1,angle_C_2)

dist_vel_angle_A = dtw.distance(vel_angle_A_1,vel_angle_A_2)
dist_vel_angle_B= dtw.distance(vel_angle_B_1,vel_angle_B_2)
dist_vel_angle_C= dtw.distance(vel_angle_C_1,vel_angle_C_2)





print(dist_angle_A)
print(dist_angle_B)
print(dist_angle_C)
print(dist_vel_angle_A)
print(dist_vel_angle_B)
print(dist_vel_angle_C)



t = np.arange(1, len(angle_A_1) + 1, step = 1 )

plt.plot(t, angle_A_1, color = 'black', label='Ángulo de Cadera 1')
plt.plot(t, angle_B_1, color = 'red', label=' Ángulo de Rodilla 1')
plt.plot(t, angle_C_1, color = 'blue', label=' Ángulo de Tobillo 1')
plt.xlabel("Frame")
plt.ylabel("Ángulo [°]")
plt.legend()

plt.grid()

plt.show()

t = np.arange(1, len(angle_A_2) + 1, step = 1 )

plt.plot(t, angle_A_2, color = 'black', label='Ángulo de Cadera 2')
plt.plot(t, angle_B_2, color = 'red', label=' Ángulo de Rodilla 2')
plt.plot(t, angle_C_2, color = 'blue', label=' Ángulo de Tobillo 2')
plt.xlabel("Frame")
plt.ylabel("Ángulo [°]")
plt.legend()

plt.grid()

plt.show()


t = np.arange(1, len(vel_angle_A_1) + 1, step = 1 )

plt.plot(t, vel_angle_A_1, color = 'black', label='Cadera 1')
plt.plot(t, vel_angle_B_1, color = 'red', label='Rodilla 1')
plt.plot(t, vel_angle_C_1, color = 'blue', label="Tobillo 1")



plt.xlabel("Frame")
plt.ylabel("Velocidad angular [°/s]")
plt.legend()

plt.grid()

plt.show()


t = np.arange(1, len(vel_angle_A_2) + 1, step = 1 )

plt.plot(t, vel_angle_A_2, color = 'black', label='Cadera 2')
plt.plot(t, vel_angle_B_2, color = 'red', label='Rodilla 2')
plt.plot(t, vel_angle_C_2, color = 'blue', label="Tobillo 2")



plt.xlabel("Frame")
plt.ylabel("Velocidad angular [°/s]")
plt.legend()

plt.grid()

plt.show()