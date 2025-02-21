from Config.db_connection import DBConnection
import pymysql

class DBOperations:
    def __init__(self):
        self.conn = DBConnection().connection


    def reconnect(self):
        """ Intenta reconectar si la conexión está cerrada. """
        if self.conn is None or not self.conn.open:
            self.conn = DBConnection().connection

    def get_sensor_configuration(self, sensor_id , document_id , report_id):
        """
        Obtiene la configuración del sensor desde la base de datos.
        Si no encuentra configuración, retorna valores por defecto.
        """
        self.reconnect()

        # 🔹 Valores por defecto en caso de que no haya configuración en la BD
        default_config = {
            "low_cutoff": 5.0,
            "high_cutoff": 40.0,
            "filter_order": 4,
            "v_p": 6.0,
            "v_s": 3.5,
            "factor_escala": 1.0
        }

        # print(f"sensor_id: {sensor_id}, document_id: {document_id}")

        if self.conn is None:
            print("No se pudo establecer conexión con la base de datos.")
            return default_config

        try:
            with self.conn.cursor() as cursor:
                sql = """
                    SELECT 
                        ac.sensor_id, 
                        ac.factor_escala, 
                        ac.velocidad_onda_p as v_p, 
                        ac.velocidad_onda_s as v_s, 
                        ace.low_cutoff, 
                        ace.high_cutoff, 
                        ace.filter_order 
                    FROM surveying_v2_.acelerometro_configuracion ac
                    INNER JOIN reporte_acelerometro ra ON ra.sensor_id = ac.sensor_id 
                    INNER JOIN acelerometro_configuracion_evento ace ON ace.reporte_id = ra.id_reporte
                    WHERE ac.sensor_id = %s
                    AND ra.document_id = %s
                    AND ra.id_reporte = %s
                """
                cursor.execute(sql, (sensor_id, document_id, report_id))
                config = cursor.fetchone()

                print(f"config: {config } sensor_id: {sensor_id}, document_id: {document_id}, report_id: {report_id}")

                return config if config else default_config

        except pymysql.OperationalError as e:
            if e.args[0] in [2006, 2013]:  
                self.conn = DBConnection().connection
            print(f"Error de conexión: {e}")
            return default_config
        except Exception as e:
            print(f"Error al obtener la configuración del sensor: {e}")
            return default_config
