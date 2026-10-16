# Ruta autodidacta de neurociencia computacional

Plan completo de **18 semanas**, con una dedicación orientativa de **5-8 horas semanales** (90-144 horas en total). La duración y las actividades son una propuesta para autoestudio, no un requisito publicado por Stanford o Neuromatch Academy. Si un tema es nuevo, repite los ejercicios antes de avanzar en vez de acelerar el calendario.

## Antes de empezar

No necesitas dominar toda la matemática antes de programar. Repasa cada herramienta cuando la necesites:

- Python: funciones, listas, arreglos, bucles y gráficos con NumPy.
- Álgebra lineal: vectores, matrices, producto punto y autovalores.
- Cálculo: derivadas, integrales y ecuaciones diferenciales ordinarias.
- Probabilidad y estadística: distribuciones, media, varianza, regresión y validación.

El [repaso de prerrequisitos de Neuromatch](https://compneuro.neuromatch.io/prereqs/ComputationalNeuroscience.html) y sus [tutoriales introductorios de Python](https://compneuro.neuromatch.io/tutorials/W0D1_PythonWorkshop1/student/W0D1_Tutorial1.html) son una base práctica. El primer taller introduce conceptos de Python usando una neurona LIF.

## Rutina semanal sugerida

- **1-2 h:** estudiar el tutorial, lectura o apuntes indicados.
- **3-5 h:** programar, variar parámetros, graficar y repetir experimentos.
- **1 h:** escribir resultados, supuestos, preguntas y conclusiones.

Cada entrega puede ser un notebook o un script con un README breve. Registra las unidades, los parámetros y la semilla aleatoria cuando aplique. No uses datos sintéticos como evidencia de un fenómeno biológico.

## Ejecutar los scripts semanales

Desde la raíz del repositorio, puedes listar y ejecutar cada ejercicio:

```powershell
python weeks/run_weekly.py --list
python weeks/run_weekly.py 1
python weeks/run_weekly.py 18
```

También puedes ejecutar un archivo con `python weeks/week_01_python_reproducibility/week_01_python_reproducibility.py`. Cada script guarda sus figuras (`.png`), datos (`.npz`) o informe en la carpeta `outputs/` de su semana; esa carpeta está excluida de Git. Las salidas son material para inspeccionar y ampliar, no sustituyen la entrega escrita de cada semana.

Cada semana tiene su carpeta en [`weeks/`](../weeks/) con el script, su notebook ejecutado (teoría, ecuaciones, figura y validación) y sus salidas.

| Semana | Script | Experimento/artefacto |
|---|---|---|
| 1 | `week_01_python_reproducibility.py` | Señal con ruido, semilla fija y figura reproducible |
| 2 | `week_02_math_and_euler.py` | Euler frente a solución analítica para varios pasos |
| 3 | `week_03_lif_currents.py` | Voltaje y frecuencia de LIF frente a corriente |
| 4 | `week_04_hh_channels.py` | Bloqueo de Na⁺/K⁺ desde el reposo real de cada condición |
| 5 | `week_05_hh_timestep.py` | Convergencia HH respecto a paso temporal fino |
| 6 | `week_06_synaptic_filter.py` | Suma temporal de spikes con dos constantes sinápticas |
| 7 | `week_07_fixed_points.py` | Foco estable y silla: autovalores y retrato de fase |
| 8 | `week_08_wilson_cowan.py` | Punto fijo frente a ciclo límite, con nullclines |
| 9 | `week_09_hopfield_noise.py` | Tolerancia al ruido y capacidad (P/N) de Hopfield |
| 10 | `week_10_stochastic_spikes.py` | Conteos, ISI, factor de Fano y CV de trenes Poisson |
| 11 | `week_11_experiment_design.py` | Barrido de corriente LIF y plan de hipótesis |
| 12 | `week_12_model_generalization.py` | Error de entrenamiento y de prueba según el grado del polinomio |
| 13 | `week_13_poisson_glm.py` | Ajuste de tasa de conteos con GLM Poisson |
| 14 | `week_14_encoding_decoding.py` | Decodificador de estímulo evaluado con 5-fold |
| 15 | `week_15_population_pca.py` | Proyección PCA de actividad poblacional |
| 16 | `week_16_signal_processing.py` | Suavizado y espectro de una señal ruidosa |
| 17 | `week_17_bayesian_decision.py` | Decisión sensorial Bayesiana con observaciones ruidosas |
| 18 | `week_18_capstone.py` | Extrapolación f-I: línea base, lineal, umbral-lineal y teoría |

## Calendario semana a semana

### Bloque I: Herramientas y excitabilidad neuronal

#### Semana 1: Python científico y reproducibilidad

- **Estudia:** el [taller de Python de Neuromatch](https://compneuro.neuromatch.io/tutorials/W0D1_PythonWorkshop1/student/W0D1_Tutorial1.html); practica funciones, arreglos NumPy, gráficos y lectura de errores.
- **Practica:** crea un entorno, instala el proyecto, corre `python examples/run_models.py` y `python -m pytest`. Modifica el ejemplo para guardar una curva y sus parámetros.
- **Entrega:** un script o notebook que genere una figura reproducible y explique cómo ejecutarla.

#### Semana 2: Matemáticas para modelos

- **Estudia:** vectores y matrices, derivadas, exponenciales, ecuaciones diferenciales ordinarias, media y desviación estándar. Usa los [repasos de prerrequisitos de NMA](https://compneuro.neuromatch.io/prereqs/ComputationalNeuroscience.html) para identificar vacíos.
- **Practica:** grafica una exponencial, una señal sinusoidal y la solución numérica de una ecuación diferencial de primer orden; compara Euler con una solución analítica simple.
- **Entrega:** una página de notas con ecuación, unidades, parámetros y diferencia entre solución analítica y numérica.

#### Semana 3: Neurona leaky integrate-and-fire

- **Estudia:** capacitancia, resistencia, potencial de reposo, umbral, reinicio y período refractario. Vuelve al ejemplo LIF del taller NMA y explora el [módulo interactivo LIF](../interactive/lif.html).
- **Practica:** inspecciona `src/computational_neuroscience/lif.py`; simula corrientes subumbral y supraumbral y grafica voltaje y spikes.
- **Entrega:** comparación de al menos tres intensidades, con predicción previa y explicación del cambio en frecuencia de disparo.

#### Semana 4: ecuaciones y canales de Hodgkin-Huxley

- **Estudia:** el [tutorial NMA de modelos neuronales biológicos](https://compneuro.neuromatch.io/tutorials/W2D4_BiologicalNeuronModels/student/W2D4_Intro.html) y el [módulo interactivo del proyecto](../interactive/hodgkin-huxley.html). Identifica las corrientes de Na⁺, K⁺ y fuga, y las compuertas m, h, n.
- **Practica:** relaciona cada término de la ecuación de voltaje con conductancia, potencial de reversión y corriente.
- **Entrega:** diagrama anotado de las ecuaciones HH con unidades y una predicción de cómo bloquear Na⁺ o K⁺ afectaría el potencial de acción.

#### Semana 5: Simulación HH y convergencia numérica

- **Estudia:** `src/computational_neuroscience/hodgkin_huxley.py`; repasa Euler explícito y tamaño de paso.
- **Practica:** reproduce un potencial de acción y compara al menos tres valores de `dt_ms`. Comprueba el rango de las compuertas y la estabilidad de voltaje.
- **Entrega:** figura de voltaje y compuertas con parámetros y una nota sobre el costo de usar pasos más pequeños o más grandes.

#### Semana 6: Sinapsis y sistemas lineales

- **Estudia:** el tutorial NMA de [sistemas lineales](https://compneuro.neuromatch.io/tutorials/W2D3_LinearSystems/student/W2D3_Intro.html); repasa respuesta al impulso, filtros y convolución. Explora el [módulo de sinapsis](../interactive/sinapsis.html) para ver la suma temporal.
- **Practica:** genera una señal de entrada y compárala después de pasarla por un filtro de primer orden. Relaciona la constante temporal de la respuesta con integración sináptica.
- **Entrega:** gráfico de entrada/salida para dos constantes temporales y explicación de cuál produce una integración más lenta.

**Checkpoint 1:** puedes simular LIF y HH, describir sus variables y unidades, y explicar por qué el paso temporal cambia el resultado.

### Bloque II: Dinámica, redes y azar

#### Semana 7: Puntos fijos, estabilidad y excitabilidad

- **Estudia:** el tutorial NMA de [sistemas dinámicos](https://compneuro.neuromatch.io/tutorials/W2D5_DynamicalSystems/student/W2D5_Intro.html): puntos fijos, estabilidad, nullclines y retratos de fase. Usa el [módulo de sistemas lineales 2D](../interactive/sistemas-lineales.html) para relacionar autovalores y trayectorias.
- **Practica:** calcula o estima un punto fijo en un sistema de dos variables y observa cómo cambia la trayectoria al modificar parámetros.
- **Entrega:** retrato de fase anotado con punto fijo, trayectorias y explicación de estabilidad local.

#### Semana 8: Dinámica de poblaciones neuronales

- **Estudia:** modelos de tasa y redes recurrentes; conecta excitación e inhibición con las oscilaciones o estados estables del sistema. El [módulo de Wilson-Cowan](../interactive/wilson-cowan.html) permite explorar los regímenes antes de programarlos. Consulta el curso NMA y, como mapa conceptual avanzado, la ficha de Stanford [NBIO225](https://bulletin.stanford.edu/courses/2279421).
- **Practica:** implementa o adapta un modelo de dos poblaciones (excitatoria e inhibitoria) y explora cómo cambiar conexiones afecta su actividad.
- **Entrega:** comparación de dos regímenes dinámicos con parámetros documentados.

#### Semana 9: Hopfield, memoria asociativa y energía

- **Estudia:** el [módulo interactivo de Hopfield](../interactive/hopfield.html) y `src/computational_neuroscience/hopfield.py`. Repasa regla de Hebb, actualización asíncrona, atractores y energía.
- **Practica:** guarda patrones, añade ruido y observa si se recuperan. Grafica la energía en cada barrido y compara aprendizaje Hebbiano con los límites de capacidad observados.
- **Entrega:** curva de recuperación (porcentaje de bits recuperados frente a ruido) para varios patrones; registra también fallos y estados espurios.

#### Semana 10: Probabilidad y procesos estocásticos

- **Estudia:** variables aleatorias, Poisson, ruido gaussiano, muestreo y semillas aleatorias. Usa el [módulo de trenes Poisson](../interactive/poisson.html) para comparar simulación y teoría.
- **Practica:** compara un tren de spikes determinista y uno Poisson con tasa media semejante; estima tasa y variabilidad a partir de varias repeticiones.
- **Entrega:** figura con varias realizaciones y una explicación de por qué una sola repetición no describe toda la variabilidad.

#### Semana 11: Diseño de modelos y formulación de hipótesis

- **Estudia:** el tutorial de [práctica de modelado de Neuromatch](https://compneuro.neuromatch.io/tutorials/W2D1_ModelingPractice/student/W2D1_Intro.html): pregunta, hipótesis, ingredientes y supuestos.
- **Practica:** formula una pregunta concreta sobre uno de los modelos ya estudiados; define variable de entrada, observables, predicción y una explicación alternativa.
- **Entrega:** plan de experimento de una página, escrito antes de correr el experimento, con al menos una predicción cuantitativa.

**Checkpoint 2:** puedes distinguir punto fijo y atractor, describir cómo una red almacena estados y proponer un experimento que pueda refutar una predicción.

### Bloque III: Ajuste, respuestas neuronales y poblaciones

#### Semana 12: Ajuste de modelos y generalización

- **Estudia:** el tutorial NMA de [ajuste de modelos](https://compneuro.neuromatch.io/tutorials/W1D2_ModelFitting/student/W1D2_Intro.html): parámetros, error, regularización y comparación de modelos.
- **Practica:** usa `fit_linear_model` en datos sintéticos, separa entrenamiento y prueba manualmente y compara una relación lineal con una alternativa más compleja.
- **Entrega:** predicciones sobre datos de prueba y explicación de por qué evaluar sobre los mismos datos usados para ajustar puede engañar.

#### Semana 13: GLM de Poisson para conteos de spikes

- **Estudia:** el tutorial NMA de [GLM](https://compneuro.neuromatch.io/tutorials/W1D3_GeneralizedLinearModels/student/W1D3_Intro.html), variables de conteo, enlace log y supuestos de Poisson.
- **Practica:** usa `fit_poisson_glm` con conteos sintéticos generados desde una tasa conocida; compara parámetros verdaderos y estimados.
- **Entrega:** gráfico de tasa esperada frente a estímulo y nota sobre límites del modelo (independencia, sobredispersión o intervalos temporales).

#### Semana 14: Codificación, decodificación y validación

- **Estudia:** codificación y decodificación en el material NMA de GLM; repasa partición entrenamiento/prueba y validación cruzada.
- **Practica:** predice una variable de estímulo desde respuestas simuladas (decodificación) o predice la respuesta desde el estímulo (codificación). Mantén datos de prueba separados del ajuste y del preprocesamiento.
- **Entrega:** métrica en datos no vistos, comparación con una línea base y descripción de cómo evitaste fuga de información.

#### Semana 15: PCA y actividad poblacional

- **Estudia:** el tutorial NMA de [reducción de dimensionalidad](https://compneuro.neuromatch.io/tutorials/W1D4_DimensionalityReduction/student/W1D4_Intro.html): PCA, varianza explicada y proyecciones.
- **Practica:** aplica `fit_pca` a actividad simulada de varias neuronas; colorea la proyección por condición y compara componentes.
- **Entrega:** gráfico de los dos primeros componentes (si están disponibles), varianza explicada y una cautela sobre interpretación de componentes.

#### Semana 16: Procesamiento de señales neuronales

- **Estudia:** el tutorial NMA de [procesamiento de señales](https://compneuro.neuromatch.io/tutorials/W2D2_SignalProcessing/student/W2D2_Intro.html): muestreo, filtrado y espectro.
- **Practica:** crea una señal con dos frecuencias y ruido; filtra o calcula su espectro con herramientas NumPy y compara antes/después.
- **Entrega:** figura temporal y espectral, frecuencias identificadas y explicación de cómo la frecuencia de muestreo limita el análisis.

**Checkpoint 3:** puedes ajustar un modelo de conteos, evaluar generalización con datos reservados, visualizar una población con PCA y explicar los límites del filtrado.

### Bloque IV: Síntesis y proyecto

#### Semana 17: Especialización elegida

- **Elige una opción:** dinámica y cognición (decisión bayesiana, con su [módulo interactivo](../interactive/decision-bayesiana.html), o aprendizaje por refuerzo); datos (señales y modelos de codificación; la [extensión de fNIRS con Cedalion](../extensions/fnirs-cedalion/README.md) analiza un registro real de *finger tapping* con GLM, y la [demo de fNIRS](../interactive/fnirs.html) permite explorar 5 sujetos de un dataset abierto); redes (plasticidad y redes recurrentes).
- **Estudia:** el tema correspondiente en el [temario y calendario de Neuromatch](https://compneuro.neuromatch.io/tutorials/Schedule/schedule_intro.html). Para redes y aprendizaje, el archivo Stanford [PSYCH 209](https://web.stanford.edu/class/psych209/) sirve como referencia conceptual; es de 2016-17 y su software está obsoleto.
- **Practica:** define el alcance mínimo del proyecto final y reproduce un resultado pequeño relacionado con la especialización.
- **Entrega:** pregunta final, datos/modelo, hipótesis, plan de análisis y criterio de éxito escritos antes del análisis.

#### Semana 18: Proyecto final y comunicación

- **Construye:** una respuesta reproducible a la pregunta de la semana 17; incluye comparación con una línea base o alternativa.
- **Verifica:** pruebas o comprobaciones, unidades, parámetros, semilla, datos usados y pasos para volver a ejecutar el análisis.
- **Comunica:** presenta pregunta, motivación, método, resultado, limitaciones y siguiente experimento en un informe corto o presentación.
- **Entrega final:** notebook o scripts limpios, figuras con leyendas, instrucciones de reproducción y resumen de 1-2 páginas.

**Checkpoint final:** otra persona debería poder ejecutar el proyecto y entender qué resultado apoya la hipótesis, qué no demuestra y cuáles son las limitaciones principales.

## Material Stanford como complemento

- [NBIO225: Computational Neuroscience of Cognition](https://bulletin.stanford.edu/courses/2279421): útil para orientar una futura especialización en dinámica de poblaciones y cognición. El catálogo 2026-27 consultado indicaba “No Sections Found” para otoño; revisa la disponibilidad de nuevo si te interesa cursarlo.
- [PSYCH 209: Neural Network Models of Cognition](https://web.stanford.edu/class/psych209/): archivo de invierno 2016-17 sobre redes recurrentes, aprendizaje y refuerzo. Su configuración de Python/TensorFlow es antigua; consúltalo solo por los conceptos.

## Recursos de Neuromatch Academy

- [Computational Neuroscience: temario](https://compneuro.neuromatch.io/tutorials/intro.html)
- [Prerrequisitos y repasos](https://compneuro.neuromatch.io/prereqs/ComputationalNeuroscience.html)
- [Calendario y tutoriales](https://compneuro.neuromatch.io/tutorials/Schedule/schedule_intro.html)
- [Preguntas frecuentes de los cursos](https://neuromatch.io/courses/): el material público se puede usar para autoestudio; la experiencia de cohorte en vivo añade estructura, apoyo docente, colaboración y proyecto con mentoría. La disponibilidad de cohortes y fechas puede cambiar.

## Qué cubre y qué no cubre el plan

Al terminar estas 18 semanas habrás practicado fundamentos de Python, LIF, Hodgkin-Huxley, métodos numéricos, sistemas lineales y no lineales, modelos de población, Hopfield, azar, ajuste, GLM de Poisson, codificación/decodificación, PCA, procesamiento de señales y comunicación científica.

Es un primer recorrido, no una formación exhaustiva ni equivalente a cursar un programa universitario. No incluye en profundidad neuroanatomía, biofísica avanzada, plasticidad sináptica/STDP, inferencia bayesiana (más allá de la decisión binaria de la semana 17), aprendizaje por refuerzo, deep learning, causalidad, datos experimentales de gran escala ni reproducibilidad de pipelines de laboratorio (la extensión de fNIRS es una primera aproximación con un solo sujeto). Elige alguno como siguiente ciclo después del proyecto final. El paquete de este repositorio implementa LIF, HH, Hopfield, ajuste lineal, GLM Poisson y PCA. Los demás temas (filtro sináptico, Wilson-Cowan, procesos de Poisson, validación cruzada, procesamiento de señales y decisión bayesiana) se resuelven dentro de los scripts semanales con NumPy, sin una API reutilizable en el paquete.

## Criterios de avance

- Puedes explicar el resultado de una simulación en términos de ecuaciones y supuestos.
- Puedes cambiar un parámetro, predecir el efecto y verificarlo.
- Puedes distinguir estabilidad numérica de comportamiento biológico.
- En análisis de datos, evalúas en datos no usados durante el ajuste.
- Guardas parámetros, unidades, código y limitaciones junto con cada resultado.
