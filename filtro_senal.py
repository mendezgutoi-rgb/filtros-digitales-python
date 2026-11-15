import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# ==========================================
# 1. DEFINICIÓN DE LA SEÑAL DE ENTRADA
# ==========================================
fs = 1000.0  # Frecuencia de muestreo (Hz)
t = np.linspace(0, 1.0, int(fs), endpoint=False)  # Vector de tiempo de 1 segundo

# Señal compuesta: 
# - Componente de baja frecuencia (5 Hz)
# - Componente de alta frecuencia (120 Hz)
# - Ruido blanco gaussiano añadido
f_baja = 5.0
f_alta = 120.0
senal_pura = np.sin(2 * np.pi * f_baja * t) + 0.5 * np.sin(2 * np.pi * f_alta * t)
ruido = 0.5 * np.random.normal(size=t.shape)
senal_ruidosa = senal_pura + ruido

# ==========================================
# 2. DISEÑO DE LOS FILTROS DIGITALES
# ==========================================
orden = 4

# A. Filtro Pasa Bajos (Corte en 30 Hz -> deja pasar la señal de 5 Hz, elimina la de 120 Hz y ruido)
corte_pb = 30.0
b_pb, a_pb = signal.butter(orden, corte_pb / (0.5 * fs), btype='low')

# B. Filtro Pasa Altos (Corte en 80 Hz -> deja pasar la señal de 120 Hz, elimina la de 5 Hz)
corte_pa = 80.0
b_pa, a_pa = signal.butter(orden, corte_pa / (0.5 * fs), btype='high')

# C. Filtro Pasa Bandas (Entre 50 Hz y 150 Hz)
corte_bp = [50.0, 150.0]
b_bp, a_bp = signal.butter(orden, [c/ (0.5 * fs) for c in corte_bp], btype='band')

# ==========================================
# 3. APLICACIÓN DE LOS FILtROS A LA SEÑAL
# ==========================================
senal_filtrada_pb = signal.lfilter(b_pb, a_pb, senal_ruidosa)
senal_filtrada_pa = signal.lfilter(b_pa, a_pa, senal_ruidosa)
senal_filtrada_bp = signal.lfilter(b_bp, a_bp, senal_ruidosa)

# ==========================================
# 4. VISUALIZACIÓN DE LOS RESULTADOS
# ==========================================
plt.figure(figsize=(12, 10))

# Gráfica 1: Señal original vs Ruidosa
plt.subplot(4, 1, 1)
plt.plot(t, senal_ruidosa, label='Señal con Ruido', color='gray', alpha=0.7)
plt.plot(t, senal_pura, label='Señal Pura (Referencia)', color='blue', linewidth=1.5)
plt.title('1. Señal de Entrada Original y con Ruido')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.legend(loc='upper right')
plt.grid(True)

# Gráfica 2: Filtro Pasa Bajos
plt.subplot(4, 1, 2)
plt.plot(t, senal_ruidosa, label='Con Ruido', color='gray', alpha=0.4)
plt.plot(t, senal_filtrada_pb, label='Filtrada Pasa Bajos (<=30 Hz)', color='green', linewidth=2)
plt.title('2. Aplicación de Filtro Pasa Bajos')
plt.ylabel('Amplitud')
plt.legend(loc='upper right')
plt.grid(True)

# Gráfica 3: Filtro Pasa Altos
plt.subplot(4, 1, 3)
plt.plot(t, senal_ruidosa, label='Con Ruido', color='gray', alpha=0.4)
plt.plot(t, senal_filtrada_pa, label='Filtrada Pasa Altos (>=80 Hz)', color='red', linewidth=2)
plt.title('3. Aplicación de Filtro Pasa Altos')
plt.ylabel('Amplitud')
plt.legend(loc='upper right')
plt.grid(True)

# Gráfica 4: Filtro Pasa Bandas
plt.subplot(4, 1, 4)
plt.plot(t, senal_ruidosa, label='Con Ruido', color='gray', alpha=0.4)
plt.plot(t, senal_filtrada_bp, label='Filtrada Pasa Bandas (50-150 Hz)', color='purple', linewidth=2)
plt.title('4. Aplicación de Filtro Pasa Bandas')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.show()