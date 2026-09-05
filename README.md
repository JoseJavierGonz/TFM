# Conduccion autonoma multiagente con MAPPO en CARLA

Dos vehiculos aprenden a circular de forma cooperativa en
Town10HD bajo lluvia mediante **MAPPO** (actores independientes y critico
centralizado), y se estudia la degradacion del rendimiento al sustituir la
percepcion ideal de CARLA por medidas reales de **radar**.

## Aportacion

- Arquitectura MARL robusta con CTDE.
- Sistema de recompensa modulado y aprendizaje por currículo.
- Infraestructura de simulación tolerante a fallos e inestabilidades.
- Evaluación estadística rigurosa de eventos infrecuentes.

## Estructura

```
env/          entorno Gym sobre CARLA (gymCARLA) y gestion del simulador (carlaControler)
models/       redes del actor y del critico
algorithms/   MAPPO: politica, ventajas GAE y bucle de actualizacion
train/        entrenamiento (train_custom), evaluacion (run) y escritura de metricas
analysis/     generacion de figuras y tablas del informe
agents/       modulo `agents` de la PythonAPI de CARLA (codigo de terceros, ver abajo)
results/      CSV, figuras y tablas generadas
checkpoints/  pesos guardados(no versionados)
start_carla.sh   supervisor que relanza el entrenamiento si el proceso muere
```

### Codigo de terceros

El directorio `agents/` contiene una copia del modulo `agents` perteneciente a la
API de CARLA (Copyright 2018-2020 CVC, licencia MIT). Este es necesario para el uso de
`GlobalRoutePlanner`, que construye la ruta entre origen y destino sobre el grafo
de topologia del mapa; es la ruta contra la que se calculan la desviacion lateral
del agente y la fraccion de trayecto completada. Se incluye en el repositorio para
que las rutas sean reproducibles con independencia de donde este instalada la
distribucion de CARLA. Ya que de ejecutar esto en servidor externo podrías no tener la
API de CARLA para importarla.


### 1. Instalar CARLA 0.9.14

Es un requisito imprescindible: sin un servidor CARLA accesible, ni el
entrenamiento ni la evaluacion pueden ejecutarse. Se descarga de la pagina de
releases del proyecto:

```bash
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.14.tar.gz
mkdir CARLA_0.9.14 && tar -xzf CARLA_0.9.14.tar.gz -C CARLA_0.9.14
```

La version **debe ser 0.9.14**: la API de sensores y el modulo `agents` cambian
entre versiones, y el codigo no es compatible hacia atras.

### 2. Instalar el cliente Python y las dependencias

El paquete `carla` no esta en PyPI; es el wheel que acompana al simulador:

```bash
pip install CARLA_0.9.14/PythonAPI/carla/dist/carla-0.9.14-cp38-cp38-linux_x86_64.whl
pip install -r requirements.txt
```

### 3. Lanzar el simulador y entrenar

El servidor puede correr en la misma maquina o en remoto; en este ultimo caso se
ajusta la direccion en `carlaControler`.

```bash
# terminal 1: el simulador
./CARLA_0.9.14/CarlaUE4.sh -RenderOffScreen -quality-level=Low

# terminal 2: el entrenamiento
python3 -m train.train_custom
```

`start_carla.sh` es un supervisor opcional que relanza el entrenamiento si el
proceso muere, util en sesiones largas donde el simulador puede caerse.

El entrenamiento reportado en la memoria se ejecuto en un contenedor del
laboratorio con **CARLA 0.9.14** y Python 3.8.

## Evaluacion

```bash
# escenarios base
python3 train/run.py --checkpoint checkpoints/<modelo>.pt \
  --episodes 30 --scenario no_traffic --npcs 0 --walkers 0

python3 train/run.py --checkpoint checkpoints/<modelo>.pt \
  --episodes 30 --scenario traffic --npcs 20 --walkers 20

# ablacion de percepcion (mismo escenario, misma politica)
python3 train/run.py --checkpoint checkpoints/<modelo>.pt \
  --episodes 30 --scenario traffic --npcs 20 --walkers 20 \
  --perception radar_validated

python3 train/run.py --checkpoint checkpoints/<modelo>.pt \
  --episodes 30 --scenario traffic --npcs 20 --walkers 20 \
  --perception radar_raw
```

La evaluacion usa **acciones deterministas** (la media de la gaussiana);
`--stochastic` muestrea de la politica. Los CSV se escriben en modo append con
`fsync` por fila, de modo que una caida del simulador no destruye lo ya medido.

## Metricas

Las definiciones de cada columna y **tres artefactos conocidos**
estan documentados en [`analysis/README_metricas.md`](analysis/README_metricas.md).
Conviene leerlo antes de interpretar los CSV.

