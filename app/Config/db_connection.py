import pymysql.cursors
import os
from dotenv import load_dotenv

class DBConnection:
    def __init__(self):
        load_dotenv(override=True)  # Cargar las variables de entorno desde el archivo .env

        self.DB_HOST    = str(os.getenv('DB_HOST')) 
        self.DB_USERNAME = str(os.getenv('DB_USERNAME'))
        self.DB_PASSWORD = str(os.getenv('DB_PASSWORD'))
        self.DB_DATABASE = str(os.getenv('DB_DATABASE'))

        self.connection = self.connect()
    
    def connect(self):
        try:
            self.connection = pymysql.connect(host=self.DB_HOST,
                                             user=self.DB_USERNAME,
                                             password=self.DB_PASSWORD,
                                             database=self.DB_DATABASE,
                                             cursorclass=pymysql.cursors.DictCursor)
            
            return self.connection
        
        except pymysql.MySQLError as e:
            print(f"Error al conectar a la base de datos: {e}")
            return None