from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from obspy import read
import json
import numpy as np
import uvicorn
import io
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, filtfilt, find_peaks
from app.Database.db_operations import DBOperations

app = FastAPI()
db = DBOperations()


# 🔹 Función para aplicar filtro Butterworth pasa-banda
def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    nyquist = 0.5 * fs  
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band', analog=False)
    return filtfilt(b, a, data)


# 🔹 Función para estimar la distancia al epicentro
def estimar_distancia_epicentro(t_p, t_s, v_p, v_s):
    if t_p is None or t_s is None:
        return None
    delta_t = t_s - t_p
    distancia = delta_t * (v_p * v_s) / (v_p - v_s) * 1000
    return distancia


# 🔹 Función para procesar EVT con todas las variables originales
def procesar_evt(evt_data: bytes, config):
    """
    Procesa un archivo EVT utilizando la configuración obtenida de la base de datos.
    """
    try:
        stream = read(io.BytesIO(evt_data), format="KINEMETRICS_EVT")
        
        datos = []  
        tiempos_llegada = {"P": None, "S_Y": None, "S_X": None}  

        # 🔹 Extraer valores de configuración
        low_cutoff      = config["low_cutoff"]
        high_cutoff     = config["high_cutoff"]
        filter_order    = config["filter_order"]
        v_p             = config["v_p"]
        v_s             = config["v_s"]
        factor_escala   = config["factor_escala"]

        for traza in stream:
            canal = traza.stats.channel
            time = np.arange(0, traza.stats.npts / traza.stats.sampling_rate, traza.stats.delta)
            aceleracion = (traza.data).tolist()

            # 🔹 Calcular el promedio de aceleraciones
            aceleracion_promedio = np.mean(aceleracion)

            # 🔹 Factor de conversión basado en sensibilidad
            sensibilidad        = traza.stats.sensitivity if hasattr(traza.stats, 'sensitivity') else 1.25
            bits                = traza.stats.mseed.encoding if hasattr(traza.stats, 'mseed') else 24  
            voltaje             = 2.5  
            factor_conversion   = (voltaje / (2 ** (bits - 1))) / sensibilidad * 981

            # 🔹 Normalizar aceleración
            aceleracion         = ((aceleracion - aceleracion_promedio) * factor_conversion * factor_escala).tolist()
            aceleracion_max     = np.max(aceleracion)
            aceleracion_min     = np.min(aceleracion)

            # 🔹 Aplicar filtro Butterworth
            sampling_rate           = 1 / traza.stats.delta
            aceleracion_filtrada    = butter_bandpass_filter(aceleracion, low_cutoff, high_cutoff, sampling_rate, filter_order)

            # 🔹 Calcular velocidades y desplazamientos
            velocidad       = cumulative_trapezoid(aceleracion, time, initial=0) * 2
            velocidad_max   = np.max(velocidad)
            velocidad_min   = np.min(velocidad)

            desplazamiento      = cumulative_trapezoid(velocidad, time, initial=0) * 2
            desplazamiento_max  = np.max(desplazamiento)
            desplazamiento_min  = np.min(desplazamiento)

            # 🔹 Calcular FFT
            freqs   = np.fft.rfftfreq(len(aceleracion), d=traza.stats.delta).tolist()
            fft     = np.abs(np.fft.rfft(aceleracion)).tolist()
            fft_max = np.max(fft)

            # 🔹 Calcular FFT Filtrada
            fft_filtrada        = np.abs(np.fft.rfft(aceleracion_filtrada)).tolist()
            fft_filtrada_max    = np.max(fft_filtrada)

            # 🔹 Calcular energía espectral
            energia_espectral           = (2 / len(fft)) * np.sum(np.array(fft) ** 2)
            energia_espectral_filtrada  = (2 / len(fft_filtrada)) * np.sum(np.array(fft_filtrada) ** 2)

            # 🔹 Detectar la primera llegada de ondas
            peaks, _ = find_peaks(aceleracion_filtrada, height=np.max(aceleracion_filtrada) * 0.1)
            if len(peaks) > 0:
                tiempo_llegada_canal = time[peaks[0]]
                if canal == "0":
                    tiempos_llegada["P"] = tiempo_llegada_canal
                elif canal == "1":
                    tiempos_llegada["S_Y"] = tiempo_llegada_canal
                elif canal == "2":
                    tiempos_llegada["S_X"] = tiempo_llegada_canal

            # 🔹 Calcular distancia al epicentro
            t_p                 = tiempos_llegada["P"]
            t_s                 = tiempos_llegada["S_Y"] or tiempos_llegada["S_X"]
            distancia_epicentro = estimar_distancia_epicentro(t_p, t_s, v_p, v_s)

            datos.append({
                "canal": canal,
                "tiempo": time.tolist(),
                "aceleracion": aceleracion,
                "aceleracion_promedio": aceleracion_promedio,
                "velocidad": velocidad.tolist(),
                "aceleracion_max": aceleracion_max,
                "aceleracion_min": aceleracion_min,
                "velocidad_max": velocidad_max,
                "velocidad_min": velocidad_min,
                "desplazamiento_max": desplazamiento_max,
                "desplazamiento_min": desplazamiento_min,
                "fft": fft,
                "fft_max": fft_max,
                "frecuencias": freqs,
                "energia_espectral": energia_espectral,
                "distancia_epicentro": distancia_epicentro,
                "desplazamiento": desplazamiento.tolist(),
                "velocidad_filtrada": velocidad.tolist(),
                "aceleracion_filtrada": aceleracion_filtrada.tolist(),
                "desplazamiento_filtrado": desplazamiento.tolist(),
                "energia_espectral_filtrada": energia_espectral_filtrada,
                "fft_filtrada": fft_filtrada,
            })
        
        return json.dumps(datos)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar el archivo: {str(e)}")
@app.get("/ping")
def ping():
    return {"message": "pong"}

# 🔹 Endpoint para procesar archivos EVT
@app.post("/procesar_evt/")
async def procesar_archivo_evt(
    file: UploadFile = File(...),
    id_sensor: int = Form(None),
    document_id: int = Form(None),
    report_id: int = Form(None),

):
    """
    Recibe un archivo EVT, obtiene la configuración del sensor y procesa los datos.
    """
    try:
        contenido = await file.read()

        # 🔹 Obtener configuración desde la BD o usar valores por defecto
        config      = db.get_sensor_configuration(id_sensor , document_id, report_id)
        resultado   = procesar_evt(contenido, config)

        return json.loads(resultado)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
