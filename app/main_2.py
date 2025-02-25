from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from obspy import read
import json
import numpy as np
import uvicorn
import io
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, filtfilt

app = FastAPI()

# Función para aplicar filtro Butterworth pasa-banda
def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    nyquist = 0.5 * fs  # Frecuencia de Nyquist
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band', analog=False)
    return filtfilt(b, a, data)

# Parámetros del filtro Butterworth para sismología
low_cutoff = 5.0  # Frecuencia de corte baja en Hz
high_cutoff = 40.0  # Frecuencia de corte alta en Hz
filter_order = 4  # Orden del filtro

def procesar_evt(evt_data: bytes, factor_escala: float):
    """
    Procesa un archivo EVT y extrae aceleración, velocidad, desplazamiento y FFT.
    """
    try:
        # Leer el archivo EVT desde bytes
        stream = read(io.BytesIO(evt_data), format="KINEMETRICS_EVT")
        
        datos = []
        for traza in stream:

            time            = np.arange(0, traza.stats.npts / traza.stats.sampling_rate, traza.stats.delta)
            aceleracion     = (traza.data).tolist()
            
            # Calcular el promedio de aceleraciones
            aceleracion_promedio = np.mean(aceleracion)

            # calcular factor de conversión
            sensibilidad        = traza.stats.sensitivity if hasattr(traza.stats, 'sensitivity') else 1.25
            bits                = traza.stats.mseed.encoding if hasattr(traza.stats, 'mseed') else 24  
            voltaje             = 2.5  
            factor_conversion   = (voltaje / (2 ** (bits - 1))) / sensibilidad * 981
    

            # Normalizar aceleración
            aceleracion         = ((aceleracion - aceleracion_promedio) * factor_conversion * factor_escala).tolist()
            aceleracion_max     = np.max(aceleracion)
            aceleracion_min     = np.min(aceleracion)
            
     
            # Calcular FFT correctamente serializable
            freqs               = np.fft.rfftfreq(len(aceleracion), d=traza.stats.delta).tolist()
            fft                 = np.abs(np.fft.rfft(aceleracion)).tolist()

            velocidad           = cumulative_trapezoid(aceleracion, time, initial=0)  
            velocidad           = velocidad * 2
            velocidad_max       = np.max(velocidad)
            velocidad_min       = np.min(velocidad)

            desplazamiento      = cumulative_trapezoid(velocidad, time, initial=0)   
            desplazamiento      = desplazamiento * 2
            desplazamiento_max  = np.max(desplazamiento)
            desplazamiento_min  = np.min(desplazamiento)

            # calcular maximos de la fft
            fft_max         = np.max(fft)
            fft_array       = np.array(fft)
            # Calcular energía espectral
            # Obtener solo la mitad del espectro de Fourier (hasta N/2)
            N = len(aceleracion)
            fft_array = np.array(fft[:N//2])  # Solo tomamos hasta N/2

            # Aplicar el teorema de Parseval correctamente
            energia_espectral = (2 / N) * np.sum(fft_array ** 2)

            # Excluir la componente de Nyquist si N es par
            if N % 2 == 0:
                energia_espectral -= (1 / N) * fft_array[-1] ** 2
            
            energia_espectral   = energia_espectral / 10000 

              # Aplicar filtro pasa-banda
            sampling_rate = 1 / traza.stats.delta
            aceleracion_filtrada = butter_bandpass_filter(aceleracion, low_cutoff, high_cutoff, sampling_rate, filter_order)
            
            # Calcular velocidad y desplazamiento filtrados
            velocidad_filtrada = cumulative_trapezoid(aceleracion_filtrada, time, initial=0)
            desplazamiento_filtrado = cumulative_trapezoid(velocidad_filtrada, time, initial=0)
            
            # Calcular FFT de la aceleración filtrada
            freqs = np.fft.rfftfreq(len(aceleracion), d=traza.stats.delta).tolist()
            fft_filtrada = np.abs(np.fft.rfft(aceleracion_filtrada)).tolist()
            fft_max = np.max(fft_filtrada)
 
             # calcular maximos de la fft filtrada
            fft_filtrada_max         = np.max(fft_filtrada)
            fft_filtrada_array       = np.array(fft_filtrada)
            # Calcular energía espectral
            # Obtener solo la mitad del espectro de Fourier (hasta N/2)
            N = len(aceleracion_filtrada)
            fft_filtrada_array = np.array(fft_filtrada[:N//2])  # Solo tomamos hasta N/2

            # Aplicar el teorema de Parseval correctamente
            energia_espectral_filtrada = (2 / N) * np.sum(fft_filtrada_array ** 2)

            # Excluir la componente de Nyquist si N es par
            if N % 2 == 0:
                energia_espectral_filtrada -= (1 / N) * fft_filtrada_array[-1] ** 2
            
            energia_espectral_filtrada   = energia_espectral_filtrada / 10000 
            print(f" energia_espectra_filtradal: {energia_espectral_filtrada}")

            
            datos.append({
                "canal": traza.stats.channel,
                "tiempo": time.tolist(),
                "aceleracion": aceleracion,
                "aceleracion_promedio": aceleracion_promedio,
                "velocidad": velocidad.tolist(),
                "desplazamiento": desplazamiento.tolist(),
                "aceleracion_filtrada": aceleracion_filtrada.tolist(),
                "velocidad_filtrada": velocidad_filtrada.tolist(),
                "desplazamiento_filtrado": desplazamiento_filtrado.tolist(),
                "aceleracion_max": aceleracion_max,
                "aceleracion_min": aceleracion_min,
                "velocidad_max": velocidad_max,
                "velocidad_min": velocidad_min,
                "desplazamiento_max": desplazamiento_max,
                "desplazamiento_min": desplazamiento_min,
                "fft": fft,
                "fft_max": fft_max,
                "fft_filtrada": fft_filtrada,
                "frecuencias": freqs,
                "energia_espectral": energia_espectral    
            })
        
        return json.dumps(datos)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar el archivo: {str(e)}")

@app.post("/procesar_evt/")
async def procesar_archivo_evt(file: UploadFile = File(...), factor_escala: float = Form(1.0)):
    """
    Recibe un archivo EVT, lo procesa y devuelve los resultados en JSON.
    """
    try:
        contenido = await file.read()
        resultado = procesar_evt(contenido, factor_escala)
        return json.loads(resultado)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)