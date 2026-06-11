Proceso BC-MLP robusto - JAKA Lift
==================================
Fecha: 2026-06-07

Objetivo de esta actualizacion
==============================
Esta actualizacion incorpora al paquete final la policy BC-MLP robusta que
iguala y extiende el comportamiento de la policy secuencial anterior:

  models/bc_mlp/bc_mlp_abs_ref_robust_128eps.pt

La policy comanda el JAKA en robosuite para aproximarse al cubo, cerrar la
pinza y levantarlo. El paquete conserva la policy secuencial .npz anterior y
anade los modelos, datasets, scripts, configs, logs y videos del flujo BC-MLP.


Relacion con los .bag humanos
=============================
La base del proyecto siguen siendo los archivos .bag de demostraciones humanas.
La cadena completa es:

1. Bags humanos
   - Videos/datos grabados desde vista cenital.
   - Solo aparece la mano y el cubo, no el brazo completo.

2. Extraccion de senales
   - scripts/extraccion_dataset_prueba.py
   - detector/best_2.onnx
   - Se extraen posicion del cubo, mano, direccion aproximada de acercamiento,
     progreso temporal y senales suavizadas.

3. Retargeting a robosuite/JAKA
   - scripts/generate_bag_direct_ctrl_dataset.py
   - Se transforman las senales humanas en trayectorias realizables por el JAKA.
   - Se aplican correcciones necesarias de cinematica, soporte, pinza abierta,
     base_twist_deg=30 y suavizado.

4. Dataset bag-derived original
   - datasets/bag_lift_direct_ctrl_basetwist30_openfix_variants_allbags_3var.hdf5
   - 89 bags fuente.
   - 3 variantes por bag.
   - 267 episodios.
   - 719400 muestras.

5. Teacher bag-derived
   - models/phase_sequence_policy_bag_basetwist30_openfix_variants_allbags_3var_cond_smooth31.npz
   - Esta policy secuencial se entreno desde el dataset derivado de bags.
   - Funciona como teacher estable para generar demostraciones ampliadas.

6. Dataset robusto para BC-MLP
   - datasets/bc_mlp/robust_teacher_delta_128eps.hdf5
   - Se genero ejecutando el teacher bag-derived en robosuite con variaciones
     adicionales de posicion del cubo, tamano, color e initial_q del robot.
   - Contiene 128 episodios teacher.
   - El entrenamiento BC-MLP usa las observaciones de esos episodios y aprende
     a predecir la referencia articular del teacher.

Por tanto, la BC-MLP no sale de un dataset artificial independiente. Sale de
un dataset ampliado por simulacion cuyo teacher fue entrenado desde los bags.
La simulacion se usa para convertir la informacion humana limitada de los .bag
en estados/acciones JAKA fisicamente ejecutables y variados.


Dataset BC-MLP robusto
======================
Archivo:
  datasets/bc_mlp/robust_teacher_delta_128eps.hdf5

Origen:
  scripts/bc_mlp/generate_teacher_delta_dataset.py

Contenido:
  - Episodios de robosuite/JAKA generados por el teacher bag-derived.
  - Variaciones de posicion del cubo.
  - Variaciones de half-size del cubo.
  - Variaciones de color del cubo.
  - Variaciones de initial_q del brazo.
  - Observaciones necesarias para entrenar la BC-MLP.

Muestras usadas por el modelo final:
  - train_samples: 333802
  - valid_samples: 44310

Observaciones usadas por el modelo final:
  - cube_initial_pos
  - progress
  - cube_size

Targets:
  - q_target: referencia articular teacher absoluta de 6 joints.
  - close: estado de gripper abierto/cerrado.

El fichero .hdf5 robusto no reemplaza el dataset original de bags. Lo
complementa. El dataset original mantiene la trazabilidad de los .bag; el
dataset robusto aumenta cobertura de posiciones y arranques mediante teacher.


Modelo BC-MLP final
===================
Archivo:
  models/bc_mlp/bc_mlp_abs_ref_robust_128eps.pt

Metadatos:
  models/bc_mlp/bc_mlp_abs_ref_robust_128eps.json

Tipo:
  Behavioural Cloning supervisado.

No es reinforcement learning:
  - No aprende por recompensa ni prueba/error.
  - Aprende por imitacion directa de pares observacion -> accion teacher.

No es unsupervised learning:
  - Los targets q_target y close estan definidos.
  - El entrenamiento minimiza error supervisado contra esos targets.

Arquitectura:
  - MLP feed-forward.
  - hidden_sizes: 512, 512, 256.
  - activacion: SiLU.
  - salida 1: q_ref de 6 articulaciones.
  - salida 2: logit de cierre de gripper.

Normalizacion:
  - obs_mean / obs_std para entradas.
  - q_mean / q_std para salidas articulares.
  - Todo queda guardado dentro del .pt.

Metricas finales de validacion offline:
  - best validation loss: 0.002273246180266142
  - q_mae_mean: 0.0033362715039402246 rad
  - gripper_acc: 0.999954879283905
  - epoch: 119

Evaluacion en rollout:
  - configs/bc_mlp/robust_eval_scenarios_8_feasible.json
  - 8 / 8 rollouts exitosos.
  - Videos incluidos en videos/bc_mlp/robust_abs_ref_8_feasible_00.mp4 ...


Scripts BC-MLP incluidos
========================
scripts/bc_mlp/train_bc_mlp.py
  Primer entrenamiento BC-MLP sobre el dataset bag-derived original.
  Sirvio para comprobar que un MLP podia igualar la policy secuencial base.

scripts/bc_mlp/eval_bc_mlp.py
  Evaluador de los primeros modelos BC-MLP.

scripts/bc_mlp/generate_teacher_delta_dataset.py
  Genera el dataset robusto ejecutando el teacher bag-derived con variaciones
  de cubo e initial_q.

scripts/bc_mlp/train_bc_mlp_abs_ref.py
  Entrena el modelo final absoluto:
    bc_mlp_abs_ref_robust_128eps.pt

scripts/bc_mlp/eval_bc_mlp_abs_ref.py
  Evalua el modelo final en robosuite y graba videos.

scripts/bc_mlp/train_bc_mlp_delta.py
  Entrenamiento alternativo que predice deltas. Se conserva por trazabilidad,
  pero el modelo final seleccionado es el absoluto.

scripts/bc_mlp/eval_bc_mlp_delta.py
  Evaluador del modelo delta alternativo.

scripts/demo_continuous_policy.py
  Demo continua actualizada. Por defecto carga:
    models/bc_mlp/bc_mlp_abs_ref_robust_128eps.pt
  y usa:
    configs/bc_mlp/robust_eval_scenarios_8_feasible.json
  Tambien acepta la policy .npz anterior con --policy.


Comandos reproducibles principales
==================================
Entrenar el BC-MLP robusto absoluto desde el dataset robusto:

  robomimic_env/bin/python scripts/bc_mlp/train_bc_mlp_abs_ref.py \
    --dataset datasets/bc_mlp/robust_teacher_delta_128eps.hdf5 \
    --teacher-policy models/phase_sequence_policy_bag_basetwist30_openfix_variants_allbags_3var_cond_smooth31.npz \
    --output models/bc_mlp/bc_mlp_abs_ref_robust_128eps.pt \
    --epochs 120 \
    --batch-size 8192 \
    --lr 8e-4 \
    --hidden-sizes 512,512,256 \
    --activation silu

Evaluar 8 escenarios validados sin video:

  robomimic_env/bin/python scripts/bc_mlp/eval_bc_mlp_abs_ref.py \
    --checkpoint models/bc_mlp/bc_mlp_abs_ref_robust_128eps.pt \
    --scenario-file configs/bc_mlp/robust_eval_scenarios_8_feasible.json \
    --rollouts 8

Grabar videos de los 8 escenarios:

  robomimic_env/bin/python scripts/bc_mlp/eval_bc_mlp_abs_ref.py \
    --checkpoint models/bc_mlp/bc_mlp_abs_ref_robust_128eps.pt \
    --scenario-file configs/bc_mlp/robust_eval_scenarios_8_feasible.json \
    --rollouts 8 \
    --video-path videos/bc_mlp/robust_abs_ref_8_feasible.mp4

Ejecutar demo continua:

  robomimic_env/bin/python scripts/demo_continuous_policy.py \
    --display \
    --real-time \
    --num-runs 0

Grabar demo continua:

  robomimic_env/bin/python scripts/demo_continuous_policy.py \
    --num-runs 3 \
    --record-dir videos/bc_mlp_continuous_demo


Carpetas BC-MLP anadidas al paquete
===================================
models/bc_mlp/
  Modelos .pt y metadatos .json. El modelo final recomendado es:
    bc_mlp_abs_ref_robust_128eps.pt

datasets/bc_mlp/
  Dataset robusto:
    robust_teacher_delta_128eps.hdf5

videos/bc_mlp/
  Videos de validacion de la BC-MLP base y robusta.

scripts/bc_mlp/
  Scripts de generacion de dataset, entrenamiento y evaluacion BC-MLP.

configs/bc_mlp/
  Escenarios de evaluacion, incluyendo:
    robust_eval_scenarios_8_feasible.json

logs/bc_mlp/
  Logs completos de entrenamiento, generacion de dataset y evaluacion.


Limitaciones conocidas
======================
La policy robusta fue validada en un rango ampliado, no en todo el espacio
posible de la mesa. Hay una esquina positiva extrema que fallo tanto con el
teacher como con pruebas IK scriptadas:

  x aprox. 0.048
  y aprox. 0.025
  initial_q perturbado fuerte

Ese caso no se incluyo como escenario validado. Para demos publicas se
recomienda usar:

  configs/bc_mlp/robust_eval_scenarios_8_feasible.json

o muestrear en:

  x: -0.020 a 0.050
  y: -0.050 a 0.020

evitando combinaciones extremas simultaneas fuera del conjunto validado.


Resumen tecnico
===============
El modelo final es una BC-MLP supervisada que aprende una aproximacion compacta
del teacher bag-derived. La policy ejecutada en rollout es el MLP: en cada paso
recibe posicion inicial del cubo, progreso temporal y tamano del cubo; predice
referencias articulares y cierre de pinza; el controlador de rollout suaviza y
satura el seguimiento para conservar movimientos lentos y estables.
