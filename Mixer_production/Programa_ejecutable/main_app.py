import tkinter as tk
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import serial
import csv
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Descomenta estas líneas en tu PC cuando uses los modelos reales
# from tensorflow.keras.models import load_model 
# from tensorflow.keras.preprocessing.image import img_to_array

# ======================================================================
# VARIABLES GLOBALES
# ======================================================================
proceso_activo = False
delay_simulacion = 30 # ms entre frames (30 = normal, 5 = cámara rápida)

# ======================================================================
# 1. VISUALIZACIÓN DE DATOS (MATPLOTLIB + PANDAS)
# ======================================================================
def generar_reporte_resultados(archivo_csv, gramos_agua=None):
    """Carga el CSV y despliega la ventana de Matplotlib. DEBE CORRER EN MAIN THREAD."""
    print(f"Cargando datos de: {archivo_csv}...")
    try:
        df = pd.read_csv(archivo_csv)
        print("¡Datos cargados correctamente!")
    except FileNotFoundError:
        print(f"[ERROR] No se encontró el archivo '{archivo_csv}'.")
        return

    # Crear figura
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Título dinámico
    if gramos_agua:
        titulo = f'Análisis de Corrida (Agua: {gramos_agua}g, Aceite: 200g, Sosa: 25.1g)'
    else:
        titulo = 'Análisis de la Corrida Ideal (Modo Simulación)'
        
    fig.suptitle(titulo, fontsize=14, fontweight='bold')

    # GRÁFICO 1: Estados y Visión (CNN)
    axs[0].plot(df['Tiempo_s'], df['Estado'], label='Estado (1=Mezcla, 2=IA, 3=DP)', color='black', linestyle='--', linewidth=2)
    axs[0].plot(df['Tiempo_s'], df['Prediccion_CNN_Suave'], label='Prob. Fase 1 (CNN)', color='blue', linewidth=2)
    axs[0].axhline(y=0.6, color='red', linestyle=':', label='Umbral CNN')
    axs[0].set_ylabel('Estado / Prob.')
    axs[0].set_title('Transición de Estados y Visión')
    axs[0].legend(loc='upper left')
    axs[0].grid(True, linestyle='--', alpha=0.7)

    # GRÁFICO 2: Esfuerzo del Motor
    axs[1].plot(df['Tiempo_s'], df['Esfuerzo_V'], label='Esfuerzo Motor (Crudo)', color='orange', alpha=0.6)
    axs[1].set_ylabel('Esfuerzo (V)')
    axs[1].set_title('Monitoreo del Esfuerzo del Motor')
    axs[1].legend(loc='upper left')
    axs[1].grid(True, linestyle='--', alpha=0.7)

    # GRÁFICO 3: Predicción de la LSTM
    axs[2].plot(df['Tiempo_s'], df['Tiempo_Rest_LSTM'], label='Tiempo Restante (LSTM)', color='purple', linewidth=2)
    axs[2].set_xlabel('Tiempo de Corrida (Segundos)')
    axs[2].set_ylabel('Segundos Restantes')
    axs[2].set_title('Predicción del Tiempo y DP')
    axs[2].legend(loc='upper right')
    axs[2].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()


# ======================================================================
# 2. LÓGICA DE CONTROL MAESTRO (DP + LSTM + CNN)
# ======================================================================
# CORREGIDO: Se añadió un callback_graficos para delegar el dibujo al hilo principal
def ejecutar_proceso_maestro(gramos_agua, puerto_com, cam_id, callback_graficos):
    global proceso_activo
    proceso_activo = True 
    print(f"--- INICIANDO PROCESO MAESTRO ---")
    
    ARCHIVO_VIDEO = 'corrida_actual.mp4'
    ARCHIVO_DATOS = 'datos_corrida_actual.csv'
    
    print("[CARGANDO IA...] Despertando a las redes neuronales...")
    try:
        # modelo_cnn = load_model('modelo_vision_cnn.h5', compile=False)
        # modelo_lstm = load_model('modelo_saponificacion_completo.h5', compile=False)
        # with open('scaler_X.pkl', 'rb') as f: scaler_X = pickle.load(f)
        # with open('scaler_y.pkl', 'rb') as f: scaler_y = pickle.load(f)
        print("[OK] Modelos IA Cargados (Simulado).")
    except Exception as e:
        print(f"[ERROR IA] {e}")
        proceso_activo = False
        return

    datos_sensor = {'esfuerzo': 0.0, 'activo': True}
    arduino = None

    def leer_arduino_en_segundo_plano():
        while datos_sensor['activo']:
            if arduino and arduino.in_waiting > 0:
                try:
                    lineas = arduino.readlines() 
                    if lineas:
                        ultima_linea = lineas[-1].decode('utf-8').strip()
                        partes = ultima_linea.split(',')
                        if len(partes) >= 3:
                            datos_sensor['esfuerzo'] = float(partes[2])
                except: pass
            time.sleep(0.01)

    if puerto_com:
        try:
            arduino = serial.Serial(puerto_com, 115200, timeout=0.05)
            time.sleep(2)
            hilo_arduino = threading.Thread(target=leer_arduino_en_segundo_plano)
            hilo_arduino.daemon = True
            hilo_arduino.start()
            print(f"[OK] Arduino en {puerto_com} con hilo activo.")
        except:
            print(f"[ALERTA] Arduino no detectado en {puerto_com}.")

    try: cap = cv2.VideoCapture(int(cam_id))
    except: cap = cv2.VideoCapture(0)
    
    frame_width, frame_height = 640, 480
    TARGET_FPS = 10.0 
    TIEMPO_POR_FRAME = 1.0 / TARGET_FPS
    out_video = cv2.VideoWriter(ARCHIVO_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), TARGET_FPS, (frame_width, frame_height))
    
    f_csv = open(ARCHIVO_DATOS, mode='w', newline='')
    writer = csv.writer(f_csv)
    writer.writerow(['Tiempo_s', 'Estado', 'Confianza_Fase1', 'PWM_Aplicado', 'Prediccion_CNN_Suave', 'Esfuerzo_V', 'Color_H', 'Color_S', 'Movimiento', 'Tiempo_Rest_LSTM', 'Esfuerzo_Limpio'])

    estado_proceso = 1  
    contador_fase_1 = 0
    FRAMES_ANTIRREBOTE = 10 
    pwm_actual = 100
    buffer_cnn, buffer_esfuerzo, buffer_lstm = [], [], []
    tiempo_lstm_suavizado = None 
    prev_gray = None 
    prediccion_cnn_suave = 0.0
    esfuerzo_limpio = 0.0

    tiempo_inicio = time.time()
    
    try:
        while proceso_activo:
            inicio_ciclo = time.time()
            ret, frame = cap.read()
            if not ret: break
                
            frame = cv2.resize(frame, (frame_width, frame_height))
            t_transcurrido = time.time() - tiempo_inicio

            esfuerzo_actual = datos_sensor['esfuerzo']
            buffer_esfuerzo.append(esfuerzo_actual)
            if len(buffer_esfuerzo) > 11: buffer_esfuerzo.pop(0)
            esfuerzo_limpio = sum(buffer_esfuerzo) / len(buffer_esfuerzo)

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            color_h_actual = np.mean(hsv_frame[:, :, 0])
            color_s_actual = np.mean(hsv_frame[:, :, 1])

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is None: prev_gray = gray_frame; movimiento_actual = 0.0
            else: movimiento_actual = np.mean(cv2.absdiff(prev_gray, gray_frame)); prev_gray = gray_frame 

            if estado_proceso == 1:
                pwm_actual = 100
                if arduino: arduino.write(f"{pwm_actual}\n".encode())
                cv2.putText(frame, f"ESTADO 1: Mezclando ({int(t_transcurrido)}s)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if t_transcurrido >= 10: estado_proceso = 2

            elif estado_proceso == 2:
                pwm_actual = 100
                if arduino: arduino.write(f"{pwm_actual}\n".encode())
                
                # img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # img_array = np.expand_dims(img_to_array(cv2.resize(img_rgb, (150, 150))), axis=0) 
                # preds = modelo_cnn(img_array, training=False).numpy()[0]
                # pred_raw = preds[1] if len(preds) > 1 else preds[0]
                pred_raw = 0.0 
                
                buffer_cnn.append(pred_raw)
                if len(buffer_cnn) > 15: buffer_cnn.pop(0)
                prediccion_cnn_suave = sum(buffer_cnn) / len(buffer_cnn)
                
                if prediccion_cnn_suave > 0.35: contador_fase_1 += 1
                else: contador_fase_1 = 0

                if t_transcurrido > 1200: estado_proceso = 3 

                cv2.putText(frame, f"ESTADO 2: IA ({contador_fase_1}/{FRAMES_ANTIRREBOTE})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Prob. Fase 1: {prediccion_cnn_suave*100:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if contador_fase_1 >= FRAMES_ANTIRREBOTE: estado_proceso = 3

            elif estado_proceso == 3:
                TIEMPO_MAXIMO_REACCION = 3600 
                buffer_lstm.append([t_transcurrido, esfuerzo_actual, color_h_actual, color_s_actual, movimiento_actual, esfuerzo_limpio])
                if len(buffer_lstm) > 20: buffer_lstm.pop(0) 
                    
                if len(buffer_lstm) == 20:
                    try:
                        # datos_escalados = np.expand_dims(scaler_X.transform(np.array(buffer_lstm)), axis=0) 
                        # resultados_lstm = modelo_lstm({"input_layer": datos_escalados}, training=False)
                        # tiempo_raw = scaler_y.inverse_transform(np.array(resultados_lstm[1]).reshape(-1, 1))[-1][0]
                        tiempo_raw = 100.0 

                        if tiempo_lstm_suavizado is None: tiempo_lstm_suavizado = min(max(tiempo_raw, 0), TIEMPO_MAXIMO_REACCION)
                        else:
                            tiempo_degradado = tiempo_lstm_suavizado - 0.10
                            tiempo_lstm_suavizado = (0.05 * min(tiempo_raw, tiempo_degradado)) + (0.95 * tiempo_lstm_suavizado)
                            if tiempo_lstm_suavizado <= 0:
                                tiempo_lstm_suavizado = 0.0; estado_proceso = 4 
                    except: pass

                if tiempo_lstm_suavizado is None or tiempo_lstm_suavizado > 100: estado_k = 'S_INICIAL'
                elif tiempo_lstm_suavizado > 50: estado_k = 'S_MEDIO'
                else: estado_k = 'S_FINAL'
                    
                politica_optima_dp = {'S_INICIAL': 255, 'S_MEDIO': 200, 'S_FINAL': 150}
                pwm_objetivo = politica_optima_dp[estado_k]
                pwm_actual = int((0.8 * pwm_actual) + (0.2 * pwm_objetivo))
                
                if arduino: arduino.write(f"{pwm_actual}\n".encode())
                cv2.putText(frame, f"ESTADO 3: DP POLICY (PWM:{pwm_actual})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Tiempo Restante: {tiempo_lstm_suavizado if tiempo_lstm_suavizado else 0:.1f} seg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            elif estado_proceso == 4:
                cv2.putText(frame, "PROCESO TERMINADO. APAGANDO...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if arduino: arduino.write(b"0\n") 
                out_video.write(frame)
                break 
                
            out_video.write(frame)
            writer.writerow([round(t_transcurrido, 2), estado_proceso, contador_fase_1, pwm_actual, round(prediccion_cnn_suave, 4), round(esfuerzo_actual, 4), round(color_h_actual, 2), round(color_s_actual, 2), round(movimiento_actual, 2), round(tiempo_lstm_suavizado if tiempo_lstm_suavizado else 0, 2), round(esfuerzo_limpio, 4)])
            cv2.imshow("Monitor en Vivo", frame)
            
            tiempo_ejecucion = time.time() - inicio_ciclo
            if tiempo_ejecucion < TIEMPO_POR_FRAME: time.sleep(TIEMPO_POR_FRAME - tiempo_ejecucion)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            elif cv2.waitKey(1) & 0xFF == ord('f') and estado_proceso == 2: estado_proceso = 3 

    finally:
        print("Cerrando hardware y guardando...")
        datos_sensor['activo'] = False 
        cap.release()
        out_video.release() 
        f_csv.close()
        cv2.destroyAllWindows()
        if arduino:
            arduino.write(b"0\n") 
            arduino.close()
        print("[EXITO] Proceso finalizado.")
        
        # CORREGIDO: Delegamos la generación de gráficos al hilo principal
        if callback_graficos:
            callback_graficos(ARCHIVO_DATOS, gramos_agua)


# ======================================================================
# 3. INTERFAZ GRÁFICA (LA "PIEL")
# ======================================================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class AppSaponificacion(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Sistema de Control SCADA - DP & IA")
        self.geometry("600x500") 

        self.label_titulo = ctk.CTkLabel(self, text="Panel de Control Maestro", font=("Arial", 24, "bold"))
        self.label_titulo.pack(pady=30)

        self.btn_experimental = ctk.CTkButton(self, text="Corrida Experimental (En Vivo)", command=self.abrir_ventana_experimental, width=250, height=50)
        self.btn_experimental.pack(pady=10)

        self.btn_simulacion = ctk.CTkButton(self, text="Modo Simulación (Gemelo Digital)", command=self.iniciar_simulacion, width=250, height=50, fg_color="#5a5a5a")
        self.btn_simulacion.pack(pady=10)
        
        self.frame_controles_sim = ctk.CTkFrame(self, fg_color="transparent")
        
        self.btn_acelerar_sim = ctk.CTkButton(self.frame_controles_sim, text="Velocidad: Normal", command=self.cambiar_velocidad_sim, width=120, fg_color="#E67E22", hover_color="#D35400")
        self.btn_acelerar_sim.pack(side="left", padx=10)

        self.btn_detener_sim = ctk.CTkButton(self.frame_controles_sim, text="PARAR SIMULACIÓN", command=self.detener_proceso, width=120, fg_color="red", hover_color="darkred")
        self.btn_detener_sim.pack(side="left", padx=10)

    def abrir_ventana_experimental(self):
        self.ventana_exp = ctk.CTkToplevel(self)
        self.ventana_exp.title("Configuración Experimental")
        self.ventana_exp.geometry("500x600")
        self.ventana_exp.attributes("-topmost", True)

        ctk.CTkLabel(self.ventana_exp, text="Configuración de Variables", font=("Arial", 16, "bold")).pack(pady=10)
        self.input_agua = ctk.CTkEntry(self.ventana_exp, placeholder_text="Gramos de Agua (ej: 70)")
        self.input_agua.pack(pady=5)
        self.input_serial = ctk.CTkEntry(self.ventana_exp, placeholder_text="Puerto Serial (ej: COM6)")
        self.input_serial.pack(pady=5)
        self.input_camara = ctk.CTkEntry(self.ventana_exp, placeholder_text="ID Cámara (ej: 0, 1, 2)")
        self.input_camara.pack(pady=5)

        self.btn_test_cam = ctk.CTkButton(self.ventana_exp, text="Prueba de Cámara (Overlay)", command=self.ventana_prueba_camara)
        self.btn_test_cam.pack(pady=20)

        self.btn_test_ard = ctk.CTkButton(self.ventana_exp, text="Prueba de Arduino", command=self.ventana_prueba_arduino)
        self.btn_test_ard.pack(pady=5)

        self.btn_iniciar = ctk.CTkButton(self.ventana_exp, text="INICIAR EJECUCIÓN EN VIVO", fg_color="green", hover_color="darkgreen", command=self.arrancar_proceso_real)
        self.btn_iniciar.pack(pady=40)
        
        self.btn_detener = ctk.CTkButton(self.ventana_exp, text="DETENER EMERGENCIA", fg_color="red", hover_color="darkred", command=self.detener_proceso)
        self.btn_detener.pack(pady=10)

    def ventana_prueba_camara(self):
        cam_id = int(self.input_camara.get() or 0)
        v_cam = ctk.CTkToplevel(self)
        v_cam.title("Alineación de Cámara")
        v_cam.geometry("640x520")
        lbl_video = tk.Label(v_cam)
        lbl_video.pack()

        try:
            img_ref = cv2.imread('referencia.jpg')
            img_ref = cv2.resize(img_ref, (640, 480))
        except: img_ref = None

        cap = cv2.VideoCapture(cam_id)
        def actualizar_video():
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (640, 480))
                if img_ref is not None:
                    frame = cv2.addWeighted(frame, 0.7, img_ref, 0.3, 0)
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                lbl_video.imgtk = imgtk
                lbl_video.configure(image=imgtk)
            if v_cam.winfo_exists(): v_cam.after(10, actualizar_video)
            else: cap.release()
        actualizar_video()

    def ventana_prueba_arduino(self):
        v_ard = ctk.CTkToplevel(self)
        v_ard.title("Calibración de Motor")
        v_ard.geometry("400x300")
        puerto = self.input_serial.get()
        try: self.ard_test = serial.Serial(puerto, 115200, timeout=0.1) if puerto else None
        except: self.ard_test = None

        ctk.CTkLabel(v_ard, text="Potencia del Motor (PWM)").pack(pady=20)
        slider = ctk.CTkSlider(v_ard, from_=0, to=255, number_of_steps=255)
        slider.pack(pady=10)
        lbl_val = ctk.CTkLabel(v_ard, text="0")
        lbl_val.pack()

        def update_val(val):
            valor = int(slider.get())
            lbl_val.configure(text=str(valor))
            if self.ard_test and self.ard_test.is_open:
                self.ard_test.write(f"{valor}\n".encode())

        slider.configure(command=update_val)

        def cerrar_ard():
            if self.ard_test: self.ard_test.close()
            v_ard.destroy()
            
        v_ard.protocol("WM_DELETE_WINDOW", cerrar_ard)

    def arrancar_proceso_real(self):
        agua = self.input_agua.get()
        com = self.input_serial.get()
        cam = self.input_camara.get() or "0" 
        
        self.btn_iniciar.configure(state="disabled", text="PROCESO EN CURSO...")
        
        # CORREGIDO: Callback que usa self.after para asegurar que plt.show() corra en el main thread
        def al_terminar(archivo, agua_val):
            self.after(0, generar_reporte_resultados, archivo, agua_val)

        hilo = threading.Thread(target=ejecutar_proceso_maestro, args=(agua, com, cam, al_terminar))
        hilo.daemon = True 
        hilo.start()

    def iniciar_simulacion(self):
        global delay_simulacion
        delay_simulacion = 30 
        
        self.btn_experimental.configure(state="disabled")
        self.btn_simulacion.configure(state="disabled", text="REPRODUCIENDO SIMULACIÓN...")
        self.frame_controles_sim.pack(pady=20) 
        
        hilo_sim = threading.Thread(target=self.hilo_reproducir_simulacion)
        hilo_sim.daemon = True
        hilo_sim.start()

    def cambiar_velocidad_sim(self):
        global delay_simulacion
        if delay_simulacion == 30:
            delay_simulacion = 5  
            self.btn_acelerar_sim.configure(text="Velocidad: RÁPIDA", fg_color="#8E44AD", hover_color="#732D91")
        else:
            delay_simulacion = 30 
            self.btn_acelerar_sim.configure(text="Velocidad: Normal", fg_color="#E67E22", hover_color="#D35400")

    def hilo_reproducir_simulacion(self):
        global proceso_activo, delay_simulacion
        proceso_activo = True
        
        cap_sim = cv2.VideoCapture('corrida_ideal.mp4')
        if not cap_sim.isOpened():
            print("[ERROR] No se encontró 'corrida_ideal.mp4'.")
            self.restaurar_botones()
            return

        while proceso_activo and cap_sim.isOpened():
            ret, frame = cap_sim.read()
            if not ret: break 
            
            frame = cv2.resize(frame, (640, 480))
            cv2.imshow("SIMULADOR - GEMELO DIGITAL", frame)
            
            if cv2.waitKey(delay_simulacion) & 0xFF == ord('q'):
                break
                
        cap_sim.release()
        cv2.destroyAllWindows()
        self.restaurar_botones()
        
        # CORREGIDO: Delegar al hilo principal usando self.after
        self.after(0, generar_reporte_resultados, 'datos_corrida_ideal.csv')
            
    def restaurar_botones(self):
        self.btn_experimental.configure(state="normal")
        self.btn_simulacion.configure(state="normal", text="Modo Simulación (Gemelo Digital)")
        self.frame_controles_sim.pack_forget() 
        try: self.btn_iniciar.configure(state="normal", text="INICIAR EJECUCIÓN EN VIVO")
        except: pass 

    def detener_proceso(self):
        global proceso_activo
        proceso_activo = False 
        print("¡ORDEN DE PARADA RECIBIDA!")
        self.restaurar_botones()

if __name__ == "__main__":
    app = AppSaponificacion()
    app.mainloop()