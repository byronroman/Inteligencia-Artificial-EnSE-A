import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from Dependencias.helpers import create_folder, draw_keypoints, mediapipe_detection, save_frames, there_hand
from constants import FONT, FONT_POS, FONT_SIZE, FRAME_ACTIONS_PATH, ROOT_PATH
from datetime import datetime

# Función para dibujar un rectángulo redondeado
def draw_rounded_rectangle(frame, top_left, bottom_right, color, radius=10, thickness=-1):
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Dibujar el rectángulo sin esquinas
    cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, thickness)

    # Dibujar círculos en las esquinas para redondearlas
    cv2.circle(frame, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(frame, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(frame, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(frame, (x2 - radius, y2 - radius), radius, color, thickness)

# Función para dibujar texto con un fondo de rectángulo redondeado
def draw_text_with_background(image, text, position, font, font_size, font_color, bg_color, padding=10):
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_size, 1)
    
    # Calculamos las coordenadas del rectángulo
    top_left = (position[0] - padding, position[1] - text_height - padding)
    bottom_right = (position[0] + text_width + padding, position[1] + padding)

    # Dibujamos el rectángulo redondeado
    draw_rounded_rectangle(image, top_left, bottom_right, bg_color, radius=15)

    # Dibujamos el texto sobre el rectángulo
    cv2.putText(image, text, position, font, font_size, font_color, 1)

def capture_samples(path, margin_frame=1, min_cant_frames=5, delay_frames=3):
    '''
    ### CAPTURA DE MUESTRAS PARA UNA PALABRA
    Recibe como parámetro la ubicación de guardado y guarda los frames
    
    `path` ruta de la carpeta de la palabra \n
    `margin_frame` cantidad de frames que se ignoran al comienzo y al final \n
    `min_cant_frames` cantidad de frames minimos para cada muestra \n
    `delay_frames` cantidad de frames que espera antes de detener la captura después de no detectar manos
    '''
    create_folder(path)
    
    count_frame = 0
    frames = []
    fix_frames = 0
    recording = False
    
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            image = frame.copy()
            results = mediapipe_detection(frame, holistic_model)
            
            if there_hand(results) or recording:
                recording = False
                count_frame += 1
                if count_frame > margin_frame:
                    draw_text_with_background(image, 'Tomando muestras...', (10, 50), FONT, FONT_SIZE * 0.8, (255, 255, 255), (255, 50, 0))
                    frames.append(np.asarray(frame))
            else:
                if len(frames) >= min_cant_frames + margin_frame:
                    fix_frames += 1
                    if fix_frames < delay_frames:
                        recording = True
                        continue
                    frames = frames[: - (margin_frame + delay_frames)]
                    today = datetime.now().strftime('%y%m%d%H%M%S%f')
                    output_folder = os.path.join(path, f"muestras_{today}")
                    create_folder(output_folder)
                    save_frames(frames, output_folder)
                
                recording, fix_frames = False, 0
                frames, count_frame = [], 0
                draw_text_with_background(image, 'Esperando para tomar muestras...', (10, 50), FONT, FONT_SIZE * 0.8, (255, 255, 255), (0, 220, 100))
            
            draw_keypoints(image, results)
            cv2.imshow(f'Toma de muestras para "{os.path.basename(path)}"', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    word_name = "como_estas" # Aquí se debe poner la palabra para la que deseamos realizar la toma de frames
    word_path = os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH, word_name)
    capture_samples(word_path)
