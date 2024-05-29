"""
proces_videos.py

En este archivo, se definen las funciones necesarias para 
obtener los datos de los videos y comparar el tiro libre del usuario 
con el del profecional

"""
import os 
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def angles_calc(point1, center, point2):
    """
    Retorna el ángulo formado por los puntos indicados, en sentido antihorario

    Parámetros
    ----------
    pont1 : float
        Primer punto
    center : float
        Punto central
    pont2 : int, optional
        Último punto, en sentido antihorario

    Retorna
    ----------
    angle_deg : float
        Ángulo en grados
    """

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


def process_Video(path, foot):
    """
    Analiza el video del path ingresado y devueve los ángulos de la Cadera, Rodilla y Tobillo
    por frame. Además, devuelve el frame en que se realiza el disparo.

    Parámetros
    ----------
    path : string
        Path del video a analizar

    Retorna
    ----------
    angle_A : array of floats
        Ángulos de la cadera por frame
    angle_B : array of floats
        Ángulos de la rodilla por frame
    angle_C : array of floats
        Ángulos de la rodilla por frame
    shoot_index : int
        Frame del disparo
    """

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    
    # Contador de frames
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

    left_ankle_x = []
    left_ankle_y = []

    right_foot_index_x = []
    right_foot_index_y = []


    video = cv2.VideoCapture(path)

    # Setup mediapipe instance
    with mp_pose.Pose(static_image_mode = False,
                      smooth_landmarks = True,
                      min_detection_confidence = 0.4,
                      min_tracking_confidence = 0.7,
                      model_complexity = 2)\
                    as pose:
        
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

                    # Se almacenan los puntos
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

                    left_ankle_x.append(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x)
                    left_ankle_y.append(-landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y)
                    
                    # Si se desea cancelar la operación
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                
            else:
                break

            # Se actualiza el número de frames
            i = i + 1
            
    # Release webcam
    video.release()

    # Close windows
    cv2.destroyAllWindows()
    
    # Se suavizan los puntos para reducir el ruido en los valores
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

    left_ankle_x = savgol_filter(left_ankle_x, window_length=20, polyorder=1) 
    left_ankle_y = savgol_filter(left_ankle_y, window_length=20, polyorder=1) 

    # Se calculan todos los ángulos por frame
    for i in range(0,len(right_shoulder_x)):
        angle_A.append(angles_calc([right_shoulder_x[i], right_shoulder_y[i]], [right_hip_x[i],right_hip_y[i]],[right_knee_x[i], right_knee_y[i]]))

        angle_B.append(angles_calc([right_hip_x[i],right_hip_y[i]],[right_knee_x[i], right_knee_y[i]], [right_ankle_x[i], right_ankle_y[i]]))

        angle_C.append(angles_calc([right_foot_index_x[i], right_foot_index_y[i] ],[right_ankle_x[i], right_ankle_y[i]], [right_knee_x[i], right_knee_y[i]]))
                    
    # Se suavizan estos datos                
    angle_A = savgol_filter(angle_A, window_length=20, polyorder=1)
    angle_B = savgol_filter(angle_B, window_length=20, polyorder=1)
    angle_C = savgol_filter(angle_C, window_length=20, polyorder=1)

    # Se calcula el frame en que la distancia entre ambos tobillos es mínima, lo cual indica que la pelota ha sido pateada
    distances = [(right_ankle_x[i] - left_ankle_x[i])**2 + (right_ankle_y[i] - left_ankle_y[i])**2 for i in range(len(right_ankle_x))]

    shoot_index = np.argmin(distances)

    return angle_A, angle_B, angle_C, shoot_index


def get_vel_angular(angle_A, fps):
    """
    Calcula las velocidades angulares de un conjunto de ángulos por frame
    en función de los fps

    Parámetros
    ----------
    angle_A : array of floats
        Ángulos de la cadera por frame
    fps : float
        Frames por segundo

    Retorna
    ----------
    vel_angle_A : array of floats
        Las velocidades angulares por frame (a partir del segundo frame)
    """
    vel_angle_A = [(angle_A[i] - angle_A[i-1]) * fps for i in range(1, len(angle_A))]

    # Se suavizan los resultados
    vel_angle_A = savgol_filter(vel_angle_A, window_length=20, polyorder=1)

    return vel_angle_A

def process_angulos(angle_A_1, angle_B_1, angle_C_1, shoot_frame_1, fps):
    # Cálculo de las velocidades angulares
    vel_angle_A_1 = get_vel_angular(angle_A_1, fps)
    vel_angle_B_1 = get_vel_angular(angle_B_1, fps)
    vel_angle_C_1 =  get_vel_angular(angle_C_1, fps)

    # Separación en etapa 1 (antes del disparo) y etapa 2 (después del disparo)
    angle_A_1_first = angle_A_1[:shoot_frame_1]
    angle_A_1_second = angle_A_1[shoot_frame_1:]

    angle_B_1_first = angle_B_1[:shoot_frame_1]
    angle_B_1_second = angle_B_1[shoot_frame_1:]

    angle_C_1_first = angle_C_1[:shoot_frame_1]
    angle_C_1_second = angle_C_1[shoot_frame_1:]

    vel_angle_A_1_first = vel_angle_A_1[:shoot_frame_1]
    vel_angle_A_1_second = vel_angle_A_1[shoot_frame_1:]

    vel_angle_B_1_first = vel_angle_B_1[:shoot_frame_1]
    vel_angle_B_1_second = vel_angle_B_1[shoot_frame_1:]

    vel_angle_C_1_first = vel_angle_C_1[:shoot_frame_1]
    vel_angle_C_1_second = vel_angle_C_1[shoot_frame_1:]

    # Primera Etapa:
    #   Se analizan los valores de los ángulos y velocidades angulares máximas
    min_angle_A_1_first = np.min(angle_A_1_first)
    min_angle_B_1_first = np.min(angle_B_1_first)
    max_angle_C_1_first = np.max(angle_C_1_first)

    max_vel_angle_A_1_first = np.max(vel_angle_A_1_first)
    max_vel_angle_B_1_first = np.max(vel_angle_B_1_first)
    max_vel_angle_C_1_first = np.min(vel_angle_C_1_first)

    # Segunda etapa:
    #   Se analizan los valores de los ángulos y velocidades angulares máximas
    max_angle_A_1_second = np.max(angle_A_1_second)
    max_angle_B_1_second = np.max(angle_B_1_second)
    min_angle_C_1_second = np.min(angle_C_1_second)

    max_vel_angle_A_1_second = np.max(vel_angle_A_1_second)
    max_vel_angle_B_1_second = np.max(vel_angle_B_1_second)
    max_vel_angle_C_1_second = np.min(vel_angle_C_1_second)

    array = [min_angle_A_1_first,
            min_angle_B_1_first,
            max_angle_C_1_first,

            max_vel_angle_A_1_first,
            max_vel_angle_B_1_first,
            max_vel_angle_C_1_first,
            
            max_angle_A_1_second,
            max_angle_B_1_second,
            min_angle_C_1_second,

            max_vel_angle_A_1_second,
            max_vel_angle_B_1_second,
            max_vel_angle_C_1_second]
    
    return array


def comparar_user_con_prof(path_data_pro, data_user):
    data = np.load(path_data_pro)
    data_pro = data['datos_pro'] 
    diferencias_user_pro = [] 
    for i in range(0,len(data_user)):
        diferencias_user_pro.append(data_pro[i] - data_user[i])
    
    return diferencias_user_pro


def analizar_video(path, fps, foot):
    #Obtener datos del video ingresado
    user_angle_A, user_angle_B, user_angle_C, user_shoot_index = process_Video(path,foot)

    #Calcular maximos y minimos de angulos y velocidades angulares 
    data = process_angulos(user_angle_A, user_angle_B, user_angle_C, user_shoot_index, fps)

    #Comparar con jugador profesional 
    diferencias_user_pro = comparar_user_con_prof('Datos Jugadores Profesionales\Datos CR7.npz',data)

    return diferencias_user_pro
