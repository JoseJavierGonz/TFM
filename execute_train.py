import sys
import os


sys.path.append('/home/joseja/Documentos/Master/TFM/CARLA_0.9.14/PythonAPI/carla')
os.environ['PYTHONPATH'] = '/home/joseja/Documentos/Master/TFM/CARLA_0.9.14/PythonAPI/carla'

# Ejecutar training
if __name__ == "__main__":
    exec(open('train/train_custom.py').read())