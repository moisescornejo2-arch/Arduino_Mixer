import cv2
import time
import serial
import csv
import numpy as np
import pickle
import threading
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf

print("=====================================================")
print("  GRABADOR DE CORRIDA - PD FORMAL + LSTM (FAST)      ")
print("=====================================================")

# --- CARGAR LA INTELIGENCIA ARTIFICIAL ---
print("[CARGANDO IA...] Despertando a las redes neuronales...")
try:
    modelo_cnn = load_model('modelo_vision_cnn.h5', compile=False)
    print("[OK] IA de Visión (CNN) Cargada.")
    
    modelo_lstm = load_model('modelo_saponificacion_completo.h5', compile=False)
    print("[OK] IA de Tiempo (LSTM) Cargada.")
    
    with open('scaler_X.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
    with open('scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)
    print("[OK] Escaladores Cargados.")
    
except Exception as e:
    print(f"[ERROR FATAL] Faltan modelos o archivos .pkl: {e}")
    exit()

PUERTO_COM = 'COM6'
CAM_INDEX = 2
ARCHIVO_VIDEO = 'corrida_ideal.mp4'
ARCHIVO_DATOS = 'datos_corrida_ideal.csv'

# --- 1. INICIALIZAR HARDWARE Y HILOS ---
arduino = None
esfuerzo_actual_global = 0.0
lectura_activa = True

def leer_arduino_en_segundo_plano():
    global esfuerzo_actual_global, lectura_activa
    while lectura_activa:
        if arduino and arduino.in_waiting > 0:
            try:
                lineas = arduino.readlines() # Leemos todo el buffer
                if lineas:
                    ultima_linea = lineas[-1].decode('utf-8').strip()
                    partes = ultima_linea.split(',')
                    if len(partes) >= 3:
                        esfuerzo_actual_global = float(partes[2])
            except:
                pass
        time.sleep(0.01) # Descanso microscópico para no saturar el procesador

try:
    arduino = serial.Serial(PUERTO_COM, 115200, timeout=0.05) # <-- BAUD RATE SUBIDO A 115200
    time.sleep(2)
    print(f"[OK] Arduino conectado en {PUERTO_COM}")
    
    # Iniciar el hilo de lectura
    hilo_arduino = threading.Thread(target=leer_arduino_en_segundo_plano)
    hilo_arduino.daemon = True
    hilo_arduino.start()
    print("[OK] Hilo de lectura de sensores activado.")
except:
    print(f"[ALERTA] No se pudo conectar a {PUERTO_COM}.")

try:
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened(): raise Exception("Cámara secundaria no disponible")
    print(f"[OK] Cámara inicializada.")
except:
    cap = cv2.VideoCapture(0)

# --- 2. CONFIGURAR GRABADOR Y SINCRONIZADOR FPS ---
TARGET_FPS = 10.0  # Frecuencia estricta para sincronizar video y datos reales
TIEMPO_POR_FRAME = 1.0 / TARGET_FPS
frame_width, frame_height = 640, 480
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out_video = cv2.VideoWriter(ARCHIVO_VIDEO, fourcc, TARGET_FPS, (frame_width, frame_height))

f_csv = open(ARCHIVO_DATOS, mode='w', newline='')
writer = csv.writer(f_csv)
writer.writerow(['Tiempo_s', 'Estado', 'Confianza_Fase1', 'PWM_Aplicado', 'Prediccion_CNN_Suave', 'Esfuerzo_V', 'Color_H', 'Color_S', 'Movimiento', 'Tiempo_Rest_LSTM', 'Esfuerzo_Limpio'])

# --- 3. VARIABLES Y FILTROS ANTI-RUIDO ---
estado_proceso = 1  
contador_fase_1 = 0
FRAMES_ANTIRREBOTE = 10 
pwm_actual = 100

buffer_cnn = []        
buffer_esfuerzo = []   
buffer_lstm = []       
tiempo_lstm_suavizado = None 

esfuerzo_limpio = 0.0 
prev_gray = None 
prediccion_cnn_suave = 0.0

print("\n>>> INICIANDO GRABACIÓN <<<")
tiempo_inicio = time.time()

try:
    while True:
        inicio_ciclo = time.time() # Cronómetro para el FPS
        
        ret, frame = cap.read()
        if not ret: break
            
        frame = cv2.resize(frame, (frame_width, frame_height))
        t_transcurrido = time.time() - tiempo_inicio

        # ==========================================
        # EXTRACCIÓN Y SUAVIZADO EN VIVO
        # ==========================================
        
        # Obtenemos el dato más fresco directo del hilo de Arduino
        esfuerzo_actual = esfuerzo_actual_global
        
        buffer_esfuerzo.append(esfuerzo_actual)
        if len(buffer_esfuerzo) > 11: buffer_esfuerzo.pop(0)
        esfuerzo_limpio = sum(buffer_esfuerzo) / len(buffer_esfuerzo)

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_h_actual = np.mean(hsv_frame[:, :, 0])
        color_s_actual = np.mean(hsv_frame[:, :, 1])

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray_frame
            movimiento_actual = 0.0
        else:
            movimiento_actual = np.mean(cv2.absdiff(prev_gray, gray_frame))
            prev_gray = gray_frame 

        # ==========================================
        # ESTADO 1: Homogeneización Inicial
        # ==========================================
        if estado_proceso == 1:
            pwm_actual = 100
            if arduino: arduino.write(f"{pwm_actual}\n".encode())
            cv2.putText(frame, f"ESTADO 1: Mezclando ({int(t_transcurrido)}s)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if t_transcurrido >= 10: 
                estado_proceso = 2

        # ==========================================
        # ESTADO 2: Lógica Inversa CNN
        # ==========================================
        elif estado_proceso == 2:
            pwm_actual = 100
            if arduino: arduino.write(f"{pwm_actual}\n".encode())
            
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_array = img_to_array(cv2.resize(img_rgb, (150, 150))) 
            img_array = np.expand_dims(img_array, axis=0) 
            
            try:
                # OPTIMIZACIÓN Keras: Usar el tensor directamente
                predicciones = modelo_cnn(img_array, training=False).numpy()[0]
                pred_raw = predicciones[1] if len(predicciones) > 1 else predicciones[0]
                
                buffer_cnn.append(pred_raw)
                if len(buffer_cnn) > 15: buffer_cnn.pop(0)
                prediccion_cnn_suave = sum(buffer_cnn) / len(buffer_cnn)
            except Exception as e: 
                pass
            
            # Usando tu umbral modificado de 0.35
            if prediccion_cnn_suave > 0.35: 
                contador_fase_1 += 1
            else:
                contador_fase_1 = 0

            # Tu Timeout de seguridad
            if t_transcurrido > 1200:
                print("\n[TIMEOUT] Forzando transición a Fase 3.")
                estado_proceso = 3

            cv2.putText(frame, f"ESTADO 2: IA ({contador_fase_1}/{FRAMES_ANTIRREBOTE})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Prob. Fase 1: {prediccion_cnn_suave*100:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if contador_fase_1 >= FRAMES_ANTIRREBOTE:
                estado_proceso = 3
                print("\n¡FASE 1 ASUMIDA! Activando Programación Dinámica y LSTM...")

        # ==========================================
        # ESTADO 3: Programación Dinámica basada en LSTM
        # ==========================================
        elif estado_proceso == 3:
            
            # --- PRIMER PASO: PREDECIR EL TIEMPO RESTANTE ---
            TIEMPO_MAXIMO_REACCION = 3600 
            
            buffer_lstm.append([t_transcurrido, esfuerzo_actual, color_h_actual, color_s_actual, movimiento_actual, esfuerzo_limpio])
            if len(buffer_lstm) > 20: buffer_lstm.pop(0) 
                
            if len(buffer_lstm) == 20:
                try:
                    # Preparar tensor para Keras
                    datos_escalados = np.expand_dims(scaler_X.transform(np.array(buffer_lstm)), axis=0) 
                    diccionario_entrada = {"input_layer": datos_escalados}
                    
                    # Predicción rápida
                    resultados_lstm = modelo_lstm(diccionario_entrada, training=False)
                    tiempo_raw = scaler_y.inverse_transform(np.array(resultados_lstm[1]).reshape(-1, 1))[-1][0]
                    
                    if tiempo_lstm_suavizado is None:
                        tiempo_lstm_suavizado = min(max(tiempo_raw, 0), TIEMPO_MAXIMO_REACCION)
                    else:
                        # Degradación lógica estricta (resta 0.1s por ciclo en un entorno a 10 FPS)
                        tiempo_degradado_logico = tiempo_lstm_suavizado - 0.10
                        tiempo_ia_filtrado = min(tiempo_raw, tiempo_degradado_logico)
                        tiempo_lstm_suavizado = (0.05 * tiempo_ia_filtrado) + (0.95 * tiempo_lstm_suavizado)
                        
                        if tiempo_lstm_suavizado <= 0:
                            tiempo_lstm_suavizado = 0.0
                            print("\n[¡PROCESO COMPLETADO!] Tiempo restante agotado.")
                            estado_proceso = 4 
                        
                except Exception as e: 
                    pass

            # --- SEGUNDO PASO: POLÍTICA DE PROGRAMACIÓN DINÁMICA (DP) ---
            # Enfoque: MDP (Markov Decision Process) con Horizonte Finito basado en LSTM
            
            # 1. Discretización del Estado (S): Dividimos el tiempo restante en 3 etapas críticas de la reacción
            if tiempo_lstm_suavizado is None or tiempo_lstm_suavizado > 100:
                estado_k = 'S_INICIAL'  # Alta necesidad de transferencia de masa
            elif tiempo_lstm_suavizado > 50:
                estado_k = 'S_MEDIO'    # Reacción en progreso
            else:
                estado_k = 'S_FINAL'    # Maduración y homogeneización final
                
            # 2. Vector de Acciones Permitidas (A): Niveles de PWM
            # A = {150, 200, 255}
            
            # 3. Política Óptima Pre-calculada (Pi*): 
            # Resultado de minimizar la función de costo J = Costo_Energia + Penalidad_Mezcla
            politica_optima_dp = {
                'S_INICIAL': 255, # Minimiza penalidad de mala mezcla al inicio (Torque Max)
                'S_MEDIO': 200,   # Balance óptimo entre energía y cinética química
                'S_FINAL': 150    # Minimiza costo energético, la penalidad química ya es baja
            }
            
            # 4. Extracción de la Acción Óptima (u_k)
            pwm_objetivo = politica_optima_dp[estado_k]
                
            # --- TERCER PASO: ACTUADOR CON AMORTIGUAMIENTO ---
            # Ecuación de recurrencia de primer orden para inercia mecánica
            pwm_actual = int((0.8 * pwm_actual) + (0.2 * pwm_objetivo))
            
            if arduino: arduino.write(f"{pwm_actual}\n".encode())
            # -----------------------------------------

            display_time = tiempo_lstm_suavizado if tiempo_lstm_suavizado is not None else 0.0
            cv2.putText(frame, f"ESTADO 3: DP POLICY (PWM:{pwm_actual})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Tiempo Restante: {display_time:.1f} seg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # ==========================================
        # ESTADO 4: Finalización Automática
        # ==========================================
        elif estado_proceso == 4:
            cv2.putText(frame, "PROCESO TERMINADO. APAGANDO MOTOR...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if arduino: arduino.write(b"0\n") 
            out_video.write(frame) # Guarda el último frame con el texto
            break # Sale del while True
            
        # --- GUARDAR DATOS ---
        out_video.write(frame)
        writer.writerow([
            round(t_transcurrido, 2), estado_proceso, contador_fase_1, pwm_actual, 
            round(prediccion_cnn_suave, 4), round(esfuerzo_actual, 4), 
            round(color_h_actual, 2), round(color_s_actual, 2), 
            round(movimiento_actual, 2), 
            round(tiempo_lstm_suavizado if tiempo_lstm_suavizado else 0, 2),
            round(esfuerzo_limpio, 4)
        ])
        cv2.imshow("MONITOR", frame)
        
        # ==========================================
        # SINCRONIZADOR DE FPS (ESTRICTO)
        # ==========================================
        tiempo_ejecucion = time.time() - inicio_ciclo
        if tiempo_ejecucion < TIEMPO_POR_FRAME:
            time.sleep(TIEMPO_POR_FRAME - tiempo_ejecucion)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'): break
        elif tecla == ord('f') and estado_proceso == 2: estado_proceso = 3

except KeyboardInterrupt: pass
finally:
    print("Cerrando y guardando archivos...")
    lectura_activa = False # Apagar el hilo del Arduino
    cap.release()
    out_video.release() 
    f_csv.close()
    cv2.destroyAllWindows()
    if arduino:
        arduino.write(b"0\n") 
        arduino.close()
    print("[EXITO TOTAL]")
