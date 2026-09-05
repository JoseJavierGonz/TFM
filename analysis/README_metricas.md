# Metricas: definiciones y limitaciones

Material de apoyo para la seccion de metodologia de la memoria. Describe que mide exactamente cada columna de los CSV y que artefactos conocidos tiene, para poder reportarlos.

## Uso

```bash
python -m analysis.report_train 
python -m analysis.report_eval  
```

Ambos scripts son independientes y toleran que falten ficheros: si el radar falla,
el informe de entrenamiento sigue saliendo. Escriben en `results/figures/` y `results/tables/`.

---

## Diferencia fundamental entre entrenamiento y evaluacion

| | entrenamiento | evaluacion |
|---|---|---|
| unidad de la fila | un rollout de 2048 steps | un episodio |
| terminaciones | multiples: tras cada muerte hay respawn | una: el episodio acaba |
| acciones | muestreadas de la gaussiana (con ruido de exploracion) | la media (deterministas) |
| que reporta | progreso del aprendizaje | rendimiento de la politica |

**Las cifras del capitulo de resultados salen de la evaluacion.** Las de entrenamiento
incluyen exploracion deliberada y sobreestiman la tasa de fallo: la desviacion tipica
de la politica tiene un límite inferior(`log_std_floor`) y un bonus de entropia constante,
asi que la exploracion sigue activa al final del entrenamiento.

---

## Entrenamiento (`results/train_episodes.csv`)

Una fila por agente y rollout.

| columna | significado |
|---|---|
| `episode` | indice del rollout, **no** de una vida |
| `reward` | recompensa acumulada del agente en los 2048 steps |
| `goals` / `collisions` / `offroad` / `timeouts` | numero de terminaciones de cada tipo **dentro** del rollout |
| `mean_velocity` | velocidad media del agente en el rollout (m/s) |
| `route_completion` | fraccion de ruta en la **ultima** terminacion del rollout |
| `actor_loss` | perdida PPO del actor de ese agente |
| `critic_loss` | perdida del critico, **compartida** por ambos agentes |

### Artefacto 1: `collisions` cuenta muertes, no choques

Tras una colision el agente reaparece y sigue en el mismo episodio.

### Artefacto 2: `route_completion` a 0 sin terminacion

Se escribe unicamente dentro de la rama `if cause:` del bucle de entrenamiento. Un
rollout donde el agente no muere, no llega y no se sale de la ruta deja el valor a
0.0 **por no haberse escrito nunca**, no porque no avanzara. `report_train.py`
detecta esas filas (`goals=collisions=offroad=timeouts=0`) y las excluye de las
estadisticas de ruta; la columna `n ruta` de la tabla T1 indica cuantas quedaron.

### Artefacto 3: `goals` tiene poquisima resolucion

Con 6-8 sucesos por ventana, el ruido de conteo domina. La tabla T1 acompaña la
tasa con su intervalo de confianza de Poisson precisamente para esto: dos ventanas
cuyos intervalos se solapan **no** son distinguibles, por mucho que las medias
difieran. Las metricas fiables para juzgar mejora son `collisions` y
`route_completion`.

---

## Evaluacion (`results/eval_episodes.csv`)

Una fila por agente y episodio, con acciones deterministas.

| columna | significado |
|---|---|
| `outcome` | desenlace unico: `goal`, `collision`, `offroad`, `timeout` |
| `route_completion` | fraccion de ruta recorrida, valida **tambien** en timeouts |
| `steps_alive` | steps hasta la terminacion |
| `perception` | origen de las 6 features: `ground_truth`, `radar_validated`, `radar_raw` |
| `deterministic` | `True` salvo que se pasara `--stochastic` |

`route_completion` se calcula como `1 - better_distance / initial_dist`, y
`better_distance` es monotona, asi que refleja el mejor progreso alcanzado.


Los CSV son append-only y acumulan ejecuciones. **Por defecto no se descarta nada:**
si una misma condicion tiene varios reinicios, se suman todas. La alternativa: con `--latest` se filtra.

---

## Radar (`results/radar_detections.csv`)

Dos tipos de fila, distinguidas por `row_type`:

- **`tick`** — una por agente y step: `n_detections` (reflexiones crudas),
  `n_clusters` (objetos tras agrupar), `n_matched` (clusters con un actor real a
  menos de 3 m) y `n_used` (los que acaban en las features).
- **`detection`** — una por cluster, 1 de cada `--radar-sample` ticks (20 por
  defecto): `depth`, `azimuth`, `altitude`, `velocity`, `matched` y `matched_type`
  (el `type_id` del actor, o `static`).

### La figura F9 y el argumento de la memoria

`carla.RadarDetection` entrega **solo** distancia, azimut, altitud y velocidad
radial. No hay identificador de actor ni clase. Con esa informacion no se puede distinguir
 un muro de un coche parado. Esto se mide asi: se toma `|velocity|` como unico discriminante y se calcula el **AUC**
de separar actores reales de entorno estatico, mas el **coeficiente de solapamiento** de ambos
histogramas.

La tabla T6 desglosa ademas vehiculo y peaton por separado frente a estatico.

### El embudo (tabla T4)

`n_detections -> n_clusters` mide cuanto comprime el agrupamiento (un vehiculo
devuelve 15-20 reflexiones, no una). `n_clusters -> n_matched` mide que fraccion de
lo que ve el radar corresponde a un actor real; el resto es geometria estatica, que
es justamente lo que `radar_raw` no puede descartar.

Limitacion del metodo de validacion: `_match_actor` usa un radio fijo de 3 m para
todos los actores, y un peaton es mucho mas pequeño que un coche. Una reflexion en
la acera a 2,5 m de un peaton cuenta como peaton detectado, lo que infla
ligeramente `n_matched`.