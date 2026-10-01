# Validación local · 1 de octubre de 2026

Equipo: MacBook Pro M4 Max, 128 GB, macOS 27.0.1. Entorno aislado Python 3.12.14.

## Comprobaciones realizadas

- `python -m pytest -q`: **83 pruebas aprobadas** de persistencia, geometría, COCO, API y contratos de SAM 3.
- `npm run test`: **11 pruebas aprobadas** de máscaras, autoguardado, conflictos, respuestas obsoletas y montaje real de prompts en React Konva.
- `npm run build`: compilación TypeScript y producción Vite aprobadas.
- `node --experimental-strip-types --test tests/frontend-masks.mjs`: **6 pruebas aprobadas**. Fixtures codificados por pycocotools; comparación píxel a píxel contra el decodificador del navegador.
- `pip check`: ninguna dependencia incompatible.
- Resolución limpia de `requirements.txt` con `pip --dry-run --ignore-installed --only-binary=:all:` aprobada para macOS ARM/Python 3.12; informe en `output/qa/install-resolution.json`. No hay dependencias ligadas a rutas locales.
- PyTorch 2.14.1 y torchvision 0.29.1 importan correctamente; Transformers 5.18.0 expone Sam3Model y Sam3TrackerModel.
- Metal/MPS disponible al ejecutar fuera del sandbox; prueba real de `torchvision.ops.roi_align` sobre tensores MPS aprobada.
- Checkpoint fijado: `facebook/sam3`, revisión `3c879f39826c281e95690f02c7821c4de09afae7`.

Las pruebas incluyen máscaras rectangulares, huecos, islas, objetos de un píxel, bordes, edición y vértices manuales, revisión optimista, guardados concurrentes, IDs COCO, traslado del proyecto, raíces externas, revinculación y aislamiento entre pestañas.

La comprobación en Chrome incluye importación COCO, navegación entre imágenes, arrastre/inserción/eliminación de vértices, desplazamiento del lienzo sobre controles, deshacer/rehacer, recarga y exportación. Deshacer recuperó exactamente la máscara original; rehacer y recargar conservaron la edición. La reconstrucción de ambos COCO exportados mediante `pycocotools.COCO.annToMask` coincidió píxel a píxel con sus máscaras guardadas, incluido su hueco. Se utilizaron imágenes sintéticas de prueba, no resultados simulados de SAM.

También se verificó que un borrador con puntos positivos/negativos y una segunda parte con caja permanece guardado al navegar y recuperar el proyecto, incluso cuando la inferencia falla por permisos del modelo. Ese caso no valida la calidad ni el resultado de la segmentación.

## Inferencia real: aprobada en MPS y CPU

El acceso a los pesos quedó autorizado y se descargó el checkpoint oficial de 3.439.938.512 bytes. La primera prueba bloqueada por 403 queda registrada en `output/qa/sam3-benchmark.json`; las siguientes pruebas usan el modelo real y resuelven ese bloqueo.

Imagen: `test_image.jpg` de 1280 × 720, del [ejemplo oficial de Meta](https://github.com/facebookresearch/sam3/blob/main/examples/sam3_image_predictor_example.ipynb), con concepto `shoe`. El modelo detectó **12 zapatos**. Se completaron **13 casos en MPS y 13 en CPU**: texto, positivo, negativo, cajas independientes, unión de dos partes, texto → clics, refinamiento desde RLE sin cachés y repeticiones con características de imagen reutilizadas. MPS permaneció en Metal durante todos sus casos, sin fallback a CPU.

Las máscaras de CPU y MPS fueron idénticas píxel a píxel en todos los casos de esta imagen. La revisión visual confirmó que las propuestas corresponden a zapatos; no se dispone de máscaras de referencia para medir precisión de bordes.

| Medida del motor | MPS | CPU |
| --- | ---: | ---: |
| Carga desde pesos locales | 8,39 s | 3,91 s |
| Primer texto | 6,24 s | 5,23 s |
| Primer clic | 4,43 s | 4,20 s |
| Texto con imagen en caché, mediana | 152 ms | 659 ms |
| Clics con imagen en caché, mediana | 18 ms | 28 ms |

Medianas sobre tres repeticiones. Los tiempos del motor incluyen sincronización del dispositivo; no incluyen conversión a polígonos, red o dibujo de la interfaz. La mayor muestra de memoria del driver MPS fue **6,75 GB** (5,47 GB de tensores asignados). Son muestras después de cada caso, no una medición continua del pico ni una prueba prolongada de memoria.

El clic negativo cambió la máscara, pero no excluyó exactamente el píxel elegido junto al borde en este ejemplo. Se verificó que la etiqueta negativa y las coordenadas llegan correctamente al modelo: los prompts orientan su predicción, no imponen una restricción dura de píxel. Para bordes exigentes sigue siendo necesaria la revisión y, cuando corresponda, la edición manual. Refinar desde un RLE recuperado produjo una diferencia de 16 píxeles respecto a usar los logits originales; abrir el proyecto sin volver a inferir conserva la máscara exacta.

En Chrome se aceptaron 11 propuestas en conjunto, se refinó la restante con clics positivo/negativo y se confirmó con Enter. Los otros 11 objetos permanecieron intactos. Se guardaron 12 objetos y la exportación COCO reconstruyó exactamente sus máscaras con `pycocotools`. Tras reiniciar el servidor y reabrir el proyecto se recuperó el mismo estado completo.

Durante la revisión se corrigieron dos validaciones: descripciones que exceden los 32 tokens del encoder y valores no finitos de presencia. Siete regresiones adicionales pasan. El motor actualizado se comprobó con pesos reales: devuelve 422 con una indicación clara para acortar un prompt largo y sigue produciendo las 12 propuestas con `shoe`, sin cambiar las anotaciones guardadas.

Evidencia local: `output/qa/sam3-real/mps.json`, `cpu.json`, `cpu-mps-comparison.json`, `web-verification.json` y carpetas `mps-masks`/`cpu-masks` con PNG y RLE. Proyecto de demostración: `output/qa/sam3-web`; las imágenes están fuera de él, en `output/qa/sam3-real`.

Para repetir con datos propios:

```sh
.venv/bin/python scripts/benchmark_sam3.py --image /ruta/imagen.jpg --text tomato --device mps --output output/sam3-mps.json --artifacts-dir output/sam3-mps-masks
```

Queda evaluar calidad en imágenes representativas del usuario y realizar sesiones prolongadas de anotación.

Linux/CUDA y Windows/WSL son plataformas previstas por el instalador; no se ejecutaron pruebas físicas en esos sistemas durante esta sesión.
