import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
archivo_csv = 'datos_corrida_ideal.csv'

# --- 1. CARGAR DATOS ---
print(f"Cargando datos de: {archivo_csv}...")
try:
    df = pd.read_csv(archivo_csv)
    print("¡Datos cargados correctamente!")
except FileNotFoundError:
    print(f"[ERROR] No se encontró el archivo '{archivo_csv}'.")
    print("Asegúrate de ejecutar el grabador primero para generar el archivo.")
    exit()

# --- 2. CREAR GRÁFICOS ---
# Creamos una figura con 3 subgráficos apilados verticalmente que comparten el eje X (Tiempo)
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig.suptitle('Análisis de la Corrida de Saponificación', fontsize=16, fontweight='bold')

# GRÁFICO 1: Estados y Visión (CNN)
axs[0].plot(df['Tiempo_s'], df['Estado'], label='Estado (1=Mezcla, 2=IA, 3=Reacción)', color='black', linestyle='--', linewidth=2)
axs[0].plot(df['Tiempo_s'], df['Prediccion_CNN_Suave'], label='Prob. Fase 1 (CNN)', color='blue', linewidth=2)
axs[0].axhline(y=0.6, color='red', linestyle=':', label='Umbral CNN (0.6)')
axs[0].set_ylabel('Estado / Probabilidad')
axs[0].set_title('Transición de Estados y Predicción de Visión')
axs[0].legend(loc='upper left')
axs[0].grid(True, linestyle='--', alpha=0.7)

# GRÁFICO 2: Esfuerzo del Motor
axs[1].plot(df['Tiempo_s'], df['Esfuerzo_V'], label='Esfuerzo Motor (Crudo)', color='orange', alpha=0.6)
# Opcional: Si quieres graficar también el movimiento de los píxeles, descomenta la siguiente línea
# axs[1].plot(df['Tiempo_s'], df['Movimiento'], label='Movimiento (Pixeles)', color='green', alpha=0.5)
axs[1].set_ylabel('Esfuerzo')
axs[1].set_title('Monitoreo del Esfuerzo del Motor')
axs[1].legend(loc='upper left')
axs[1].grid(True, linestyle='--', alpha=0.7)

# GRÁFICO 3: Predicción de la LSTM
axs[2].plot(df['Tiempo_s'], df['Tiempo_Rest_LSTM'], label='Tiempo Restante (LSTM)', color='purple', linewidth=2)
axs[2].set_xlabel('Tiempo de Corrida (Segundos)')
axs[2].set_ylabel('Segundos Restantes')
axs[2].set_title('Predicción del Tiempo de Reacción (LSTM)')
axs[2].legend(loc='upper right')
axs[2].grid(True, linestyle='--', alpha=0.7)

# --- 3. MOSTRAR INTERFAZ ---
plt.tight_layout() # Ajusta los márgenes para que no se superpongan los textos
plt.subplots_adjust(top=0.90) # Deja espacio para el título principal
plt.show()
