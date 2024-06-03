"""
try_videos.py

En este archivo, se analizan los videos de los jugadores profesionales.

"""

import os 
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


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

import os 
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from dtaidistance import dtw, similarity



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


def process_Video(path, zurdo, fps):
    """
    Analiza el video del path ingresado y devueve los ángulos de la Cadera, Rodilla y Tobillo
    por frame. Además, devuelve el frame en que se realiza el disparo.

    Parámetros
    ----------
    path : string
        Path del video a analizar
    zurdo : bool
        False si es derecho, True si es zurdo

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
                        # Si es zurdo, invertir la imagen
            if zurdo:
                frame = cv2.flip(frame, 1)
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

                cv2.resizeWindow(window_name, 1280, 720)
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
                    if cv2.waitKey(10) == 27:
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
    right_shoulder_x = savgol_filter(right_shoulder_x, window_length=20 * fps // 240, polyorder=1)
    right_shoulder_y = savgol_filter(right_shoulder_y, window_length=20 * fps // 240, polyorder=1)

    right_hip_x = savgol_filter(right_hip_x, window_length=20 * fps // 240, polyorder=1)
    right_hip_y = savgol_filter(right_hip_y, window_length=20 * fps // 240, polyorder=1)

    right_ankle_x = savgol_filter(right_ankle_x, window_length=20 * fps // 240, polyorder=1) 
    right_ankle_y = savgol_filter(right_ankle_y, window_length=20 * fps // 240, polyorder=1)

    right_knee_x = savgol_filter(right_knee_x, window_length=20 * fps // 240, polyorder=1) 
    right_knee_y = savgol_filter(right_knee_y, window_length=20 * fps // 240, polyorder=1)

    right_foot_index_x = savgol_filter(right_foot_index_x, window_length=20 * fps // 240, polyorder=1) 
    right_foot_index_y = savgol_filter(right_foot_index_y, window_length=20 * fps // 240, polyorder=1)

    left_ankle_x = savgol_filter(left_ankle_x, window_length=20 * fps // 240, polyorder=1) 
    left_ankle_y = savgol_filter(left_ankle_y, window_length=20 * fps // 240, polyorder=1) 

    # Se calculan todos los ángulos por frame
    for i in range(0,len(right_shoulder_x)):
        angle_A.append(angles_calc([right_shoulder_x[i], right_shoulder_y[i]], [right_hip_x[i],right_hip_y[i]],[right_knee_x[i], right_knee_y[i]]))

        angle_B.append(angles_calc([right_hip_x[i],right_hip_y[i]],[right_knee_x[i], right_knee_y[i]], [right_ankle_x[i], right_ankle_y[i]]))

        angle_C.append(angles_calc([right_foot_index_x[i], right_foot_index_y[i] ],[right_ankle_x[i], right_ankle_y[i]], [right_knee_x[i], right_knee_y[i]]))
                    
    # Se suavizan estos datos                
    angle_A = savgol_filter(angle_A, window_length=20 * fps // 240, polyorder=1)
    angle_B = savgol_filter(angle_B, window_length=20 * fps // 240, polyorder=1)
    angle_C = savgol_filter(angle_C, window_length=20 * fps // 240, polyorder=1)

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
    vel_angle_A = savgol_filter(vel_angle_A, window_length=20 * fps // 240, polyorder=1)

    return vel_angle_A


def score_calculation(all_number, porcen):
    score = porcen * 100 / all_number
    if(score < 0 ):
        score = 0
    return score

###################################################################################
#   Main Test
#
###################################################################################
if __name__ == "__main__":

    # Path del video de referencia
    path_1 = "Videos Jugadores Profesionales\KevinDeBruyne_cortado.mp4"

    # CR7 son 240 fps
    # De Bruyne son 120 FPS
    # Real Madrid son 60 fps
    fps = 120

    # Cálculo de los ángulos
    angle_A_1, angle_B_1, angle_C_1, shoot_frame_1 = process_Video(path_1, False, fps)

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
    
    print(array)


    # np.savez('Datos Jugadores Profesionales\Datos Cristiano Ronaldo tiempo.npz', angle_A_1 = angle_A_1, angle_B_1 = angle_B_1, angle_C_1 = angle_C_1, shoot_frame_1 = shoot_frame_1,
    #                                         vel_angle_A_1 = vel_angle_A_1, vel_angle_B_1 = vel_angle_B_1, vel_angle_C_1 = vel_angle_C_1)
    
    # np.savez('Datos Jugadores Profesionales\Datos Cristiano Ronaldo valores maximos y minimos.npz', 
    #         min_angle_A_1_first = min_angle_A_1_first, min_angle_B_1_first =  min_angle_B_1_first, max_angle_C_1_first = max_angle_C_1_first,
    #         max_vel_angle_A_1_first = max_vel_angle_A_1_first, max_vel_angle_B_1_first = max_vel_angle_B_1_first, max_vel_angle_C_1_first = max_vel_angle_C_1_first, 
    #         max_angle_A_1_second = max_angle_A_1_second, max_angle_B_1_second = max_angle_B_1_second, min_angle_C_1_second = min_angle_C_1_second,
    #         max_vel_angle_A_1_second = max_vel_angle_A_1_second, max_vel_angle_B_1_second = max_vel_angle_B_1_second, max_vel_angle_C_1_second = max_vel_angle_C_1_second)

    # De este archivo, en el programa se obtienen datos de profesional
    np.savez('Datos Jugadores Profesionales\Datos De Bruyne.npz', datos_pro = array)

    # Gráficos en el tiempo
    print(len(angle_A_1))
    print(len(vel_angle_A_1))

    # Gráfico de los ángulos video 1
    t = np.arange(1, len(angle_A_1) + 1, step = 1 )

    plt.plot(t, angle_A_1, color = 'black', label='Ángulo de Cadera 1')
    plt.plot(t, angle_B_1, color = 'red', label=' Ángulo de Rodilla 1')
    plt.plot(t, angle_C_1, color = 'blue', label=' Ángulo de Tobillo 1')
    plt.axvline(x=shoot_frame_1, color='g', linestyle='--')
    plt.xlabel("Frame")
    plt.ylabel("Ángulo [°]")
    plt.legend()

    plt.grid()

    plt.show()

    # Gráfico de las velocidades angulares video 1
    t = np.arange(1, len(vel_angle_A_1) + 1, step = 1 )

    plt.plot(t, vel_angle_A_1, color = 'black', label='Cadera 1')
    plt.plot(t, vel_angle_B_1, color = 'red', label='Rodilla 1')
    plt.plot(t, vel_angle_C_1, color = 'blue', label="Tobillo 1")
    plt.axvline(x=shoot_frame_1-1, color='g', linestyle='--')

    plt.xlabel("Frame")
    plt.ylabel("Velocidad angular [°/s]")
    plt.legend()

    plt.grid()

    plt.show()


    data = np.load('Datos Jugadores Profesionales\Datos CR7.npz')
    data_pro1 = data['datos_pro']
    
    data = np.load('Datos Jugadores Profesionales\Datos De Bruyne.npz')
    data_pro2 = data['datos_pro'] 

    array = []
    for i in range(0,len(data_pro1)):
        array.append((data_pro1[i] + data_pro2[i])/2)
        print(data_pro1[i], data_pro2[i], (data_pro1[i] + data_pro2[i])/2)
    
    np.savez('Datos Jugadores Profesionales\Datos Promediados.npz', datos_pro = array)
    