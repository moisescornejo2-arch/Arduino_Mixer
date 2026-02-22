import cv2
import serial
import time
import pandas as pd
import numpy as np
import os
import msvcrt  # <-- LIBRERÍA MAGICA PARA LEER EL TECLADO EN LA TERMINAL
from scipy.signal import savgol_filter

# ==========================================
# 1. CONFIGURACIÓN DEL EXPERIMENTO (DoE)
# ==========================================
print("=== CONFIGURACIÓN DE CORRIDA (MÉTODO TAGUCHI) ===")
nivel_agua = input("Nivel de Agua (Ej. 50g / 70g): ").strip().upper()
nivel_vel = input("Nivel de Velocidad PWM (100 / 255): ").strip().upper()

ruta_base = f"dataset_taguchi/Agua_{nivel_agua}_Vel_{nivel_vel}"
ruta_frames = os.path.join(ruta_base, "frames")
os.makedirs(ruta_frames, exist_ok=True)

# ==========================================
# 2. INICIALIZACIÓN DE HARDWARE
# ==========================================
print("\n[!] Conectando con hardware...")
PUERTO_SERIAL = 'COM6'

try:
    arduino = serial.Serial(PUERTO_SERIAL, 9600, timeout=1)
    time.sleep(2) 
    cap = cv2.VideoCapture(2) 
except Exception as e:
    print(f"Error de conexión: {e}")
    exit()

print(f"\n[OK] Iniciando motor a PWM {nivel_vel}...")
arduino.write(f"{nivel_vel}\n".encode())
arduino.flush()

# ==========================================
# 3. VARIABLES DE MONITOREO
# ==========================================
datos_log = []
tiempo_inicio = time.time()
fase_actual = 0

nombres_fases = {
    0: "Fase 1: Mezcla Inicial",
    1: "Fase 2: Reaccion Activa",
    2: "Fase 3: Emulsion",
    3: "Punto de Traza (FIN)"
}

ret, prev_frame = cap.read()
if ret: prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

print("\n" + "="*40)
print("  >>> CONTROLES ACTIVOS EN ESTA TERMINAL <<<")
print("  Mantén esta ventana negra seleccionada.")
print("  [ 1 ] -> Pasar a Fase 2: Reacción")
print("  [ 2 ] -> Pasar a Fase 3: Emulsión")
print("  [ 3 ] -> Marcar PUNTO DE TRAZA (Finaliza y guarda)")
print("  [ Q ] -> Abortar emergencia")
print("="*40 + "\n")

# ==========================================
# 4. BUCLE PRINCIPAL (NON-BLOCKING)
# ==========================================
try:
    while True:
        t_actual = time.time()
        t_transcurrido = t_actual - tiempo_inicio
        
        ret, frame = cap.read()
        if not ret: break

        # A. Lectura de Esfuerzo (Arduino)
        line = arduino.readline().decode('utf-8').strip()
        voltaje_raw = 0.0
        if line and ',' in line:
            try: voltaje_raw = float(line.split(',')[2])
            except: pass

        # B. Procesamiento de Visión
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        roi_color = cv2.mean(hsv[200:400, 200:400])[:2] 
        diff = cv2.absdiff(gray, prev_gray)
        movimiento = np.mean(diff)
        prev_gray = gray

        # C. Guardado de Fotograma
        nombre_img = f"frame_{t_transcurrido:.2f}.jpg"
        cv2.imwrite(os.path.join(ruta_frames, nombre_img), frame)

        # D. Registro en CSV
        datos_log.append({
            'tiempo_s': round(t_transcurrido, 2),
            'fase_proceso': fase_actual,
            'nombre_fase': nombres_fases[fase_actual],
            'esfuerzo_v': voltaje_raw,
            'color_h': round(roi_color[0], 2),
            'color_s': round(roi_color[1], 2),
            'movimiento': round(movimiento, 2),
            'archivo_img': nombre_img
        })

        # E. HUD Dinámico en el Video
        if fase_actual == 0: color_texto = (0, 255, 255)     
        elif fase_actual == 1: color_texto = (0, 165, 255)   
        else: color_texto = (0, 0, 255)                      

        cv2.putText(frame, f"ETAPA: {nombres_fases[fase_actual]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_texto, 2)
        cv2.putText(frame, f"Tiempo: {t_transcurrido:.1f}s | Volt: {voltaje_raw}V", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow('Monitoreo DoE - Saponificacion', frame)
        cv2.waitKey(1) # Solo para actualizar la imagen, ya no lee el teclado aquí

        # F. LÓGICA DE TECLADO EN LA TERMINAL (msvcrt)
        if msvcrt.kbhit():
            tecla = msvcrt.getch().decode('utf-8').lower()
            
            if tecla == '1' and fase_actual == 0:
                fase_actual = 1
                print(f"[{t_transcurrido:.1f}s] >>> Transición: Entrando a Fase 2 (Reacción).")
                
            elif tecla == '2' and fase_actual == 1:
                fase_actual = 2
                print(f"[{t_transcurrido:.1f}s] >>> Transición: Entrando a Fase 3 (Emulsión).")
                
            elif tecla == '3' and fase_actual == 2:
                fase_actual = 3
                print(f"[{t_transcurrido:.1f}s] >>> PUNTO DE TRAZA ALCANZADO. Guardando...")
                break 
                
            elif tecla == 'q':
                print(f"\n[{t_transcurrido:.1f}s] [!] Abortado por el usuario.")
                break

finally:
    # ==========================================
    # 5. CIERRE SEGURO Y EXPORTACIÓN
    # ==========================================
    print("Deteniendo motor IBT-2...")
    arduino.write(b"0\n")
    arduino.flush()
    time.sleep(0.5)
    arduino.close()
    cap.release()
    cv2.destroyAllWindows()
    
    if datos_log:
        df = pd.DataFrame(datos_log)
        if len(df) > 11:
            df['esfuerzo_limpio'] = savgol_filter(df['esfuerzo_v'], 11, 2)
        
        archivo_csv = os.path.join(ruta_base, f"datos_DoE_Agua{nivel_agua}_Vel{nivel_vel}.csv")
        df.to_csv(archivo_csv, index=False)
        print(f"\n=== RESUMEN DE LA CORRIDA EXITOSA ===")
        print(f"Ruta de guardado: {ruta_base}")
        print(f"Total de registros guardados en CSV: {len(df)}")