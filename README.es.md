# Neurociencia computacional

[English](README.md) · **Español**

Un roadmap autodidacta y práctico para aprender neurociencia computacional. El objetivo es conectar la intuición biológica, los modelos matemáticos y los experimentos ejecutables: comprender un modelo, implementarlo o explorarlo y usar simulaciones para comprobar qué explica.

El proyecto está en desarrollo. Actualmente contiene dos módulos interactivos y autocontenidos; el roadmap siguiente es una ruta propuesta, no significa que todos los temas ya estén cubiertos.

## Ruta de aprendizaje

| Etapa | Tema | Enfoque |
|---|---|---|
| 1 | Fundamentos | Repasar las matemáticas, la física y Python necesarios para describir neuronas y sistemas neuronales |
| 2 | Modelos de neurona | Potencial de membrana, canales iónicos, potenciales de acción y modelo de Hodgkin–Huxley |
| 3 | Dinámica neuronal | Excitabilidad, planos de fase, patrones de disparo y modelos neuronales simplificados |
| 4 | Redes y memoria | Redes recurrentes, atractores, memoria asociativa y modelo de Hopfield |
| 5 | Aprendizaje y plasticidad | Reglas de aprendizaje sináptico, plasticidad neuronal y aprendizaje en redes |
| 6 | De los modelos a los datos | Analizar señales neuronales, comparar modelos con observaciones y documentar limitaciones |

Cada etapa irá creciendo con explicaciones, experimentos pequeños y código a medida que avance el proyecto.

## Módulos actuales

| Módulo | Qué puedes explorar |
|---|---|
| [Hodgkin–Huxley: la neurona como sistema dinámico](<Hodgkin–Huxley_ la neurona como sistema dinámico.html>) | Ajustar el estímulo y las conductancias; explorar potenciales de acción, compuertas de canales, corrientes iónicas, espacio de fase y frecuencia de disparo |
| [Red de Hopfield: memoria como paisaje de energía](<Red de Hopfield_ memoria como paisaje de energía.html>) | Guardar y recuperar patrones, añadir ruido e inspeccionar la energía, los solapamientos, los pesos sinápticos, las reglas de actualización y la capacidad de memoria |

## Cómo empezar

Los módulos actuales no requieren instalación ni compilación. Abre cualquiera de los archivos HTML en un navegador moderno y experimenta con sus controles. Las simulaciones se ejecutan en el navegador.

## Método de estudio sugerido

1. Lee la explicación del modelo e identifica sus supuestos.
2. Predice cómo cambiará el sistema al modificar un parámetro.
3. Ejecuta un experimento y compara el resultado con tu predicción.
4. Explica el resultado con tus propias palabras y anota qué aspectos no representa el modelo.
5. Cuando el tema lo requiera, reproduce el experimento con código y conserva un registro reproducible.

## Estructura del proyecto

```text
computational-neuroscience/
├── Hodgkin–Huxley_ la neurona como sistema dinámico.html
├── Red de Hopfield_ memoria como paisaje de energía.html
├── README.md
├── README.es.md
└── .gitignore
```

## Idioma

Los módulos interactivos actuales están en español. Este README está disponible en [inglés](README.md) y español.

## Alcance

Este es un proyecto de estudio autodidacta, no un curso formal ni un sustituto de la literatura primaria. Los modelos son representaciones simplificadas de sistemas biológicos; sus resultados deben interpretarse teniendo en cuenta los supuestos de cada modelo.
