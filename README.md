# Atelier · anotación de imágenes

Editor local de segmentación de imágenes: **SAM 3**, texto, clics positivos y negativos, cajas, objetos con varias partes y polígonos editables. React + FastAPI. Imágenes y anotaciones permanecen en tu equipo.

## Instalar

Requiere **Python 3.12**, **Node.js 22.x (desde 22.13), 24.x o 26 en adelante**, y npm. macOS Apple Silicon ejecuta PyTorch con MPS; Linux y Windows mediante WSL pueden usar CUDA o CPU.

```sh
python3.12 scripts/setup.py
```

El instalador crea `.venv`, instala `requirements.txt` y compila la interfaz. No instala Python ni modifica el entorno global. Alternativa manual:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cd frontend
npm ci
npm run build
cd ..
```

En Linux/WSL, instala primero la variante de PyTorch apropiada para tu GPU desde las [instrucciones oficiales](https://pytorch.org/get-started/locally/), manteniendo las versiones de `requirements.txt`. No es necesario Docker; la GPU de Apple se utiliza desde macOS directamente.

## Acceso a SAM 3

1. Solicita y acepta el acceso al repositorio oficial [facebook/sam3](https://huggingface.co/facebook/sam3).
2. Autentica tu cuenta localmente: `.venv/bin/hf auth login`. No guardes tokens en archivos del proyecto.
3. En la aplicación, carga SAM 3 o realiza una primera segmentación. Los pesos oficiales se descargan a `.cache/huggingface`; necesitas varios GB libres y conexión durante la primera descarga.

La integración usa `Sam3Model` para propuestas por texto y `Sam3TrackerModel` para clics/cajas/refinamiento. No usa SAM 2 ni genera resultados simulados si falta el modelo. Un error 403/GatedRepo significa que la cuenta o token actual aún no tiene acceso a los pesos. Después de la primera descarga se puede trabajar sin conexión.

**Estado de validación:** SAM 3 probado con pesos oficiales en MPS y CPU en la M4 Max: texto, clics, cajas, unión de partes y texto → refinamiento. También se comprobó el recorrido en navegador hasta guardar y exportar COCO. Consulta [VALIDATION.md](VALIDATION.md) para ver resultados, tiempos y límites de la muestra. Queda evaluar tus imágenes; Linux/CUDA y WSL todavía requieren validación en esos equipos.

## Ejecutar

```sh
python3 run.py
# Opciones:
python3 run.py --device mps
python3 run.py --device cpu --no-browser
```

Abre [127.0.0.1:8765](http://127.0.0.1:8765). El servicio escucha exclusivamente en localhost. `Ctrl+C` lo detiene. La primera carga del modelo puede tardar; se muestra su estado y cualquier error real.

## Proyecto e imágenes

Elige por separado dónde guardar el proyecto y dónde están sus imágenes:

```text
mi-proyecto/
├── proyecto.sqlite3
├── anotaciones.coco.json   # Al exportar
└── imágenes/
    ├── foto-001.jpg
    └── lote-2/foto-002.jpg
```

La carpeta de imágenes también puede estar fuera del proyecto. Se trabaja con los originales sin copiarlos y nunca se guardan anotaciones dentro de esa carpeta. Las rutas de cada imagen son relativas a la raíz de imágenes: `lote-2/foto-002.jpg`, también en COCO.

Al mover el proyecto completo, las rutas internas siguen funcionando. Si mueves solo las imágenes, utiliza **Revincular imágenes** y selecciona la nueva raíz. Se comprueban archivos y dimensiones; no se emparejan archivos por nombre base. Conserva la estructura de subcarpetas.

## Anotar

- Selecciona una clase. Clic positivo para iniciar y negativo para excluir; también puedes dibujar una caja.
- Usa **Añadir parte al objeto** para incorporar otra región con clics o una caja. Los prompts corrigen la parte activa; redibujar su caja la sustituye. Enter confirma la unión de todas las partes como una sola instancia.
- Escribe un concepto para obtener propuestas de SAM 3. Selecciona las válidas y acepta varias a la vez, o refina una antes de confirmar.
- Después de confirmar, selecciona el objeto para mover sus vértices, insertar puntos o borrarlos. La máscara original se conserva exactamente hasta una edición manual.
- Cambiar de imagen conserva objetos y borradores. **Guardar** fuerza el autoguardado; **Exportar COCO** escribe un JSON independiente.

Atajos disponibles en los tooltips: `N` nuevo objeto, `V` selección, `P` clic positivo, `E` clic negativo, `B` caja, `Enter` confirmar, `Esc` suspender/deseleccionar, `Cmd/Ctrl+Z` deshacer y `Cmd/Ctrl+Shift+Z` rehacer. Arriba/abajo navega imágenes sin objeto seleccionado. Los atajos de anotación no interceptan la escritura en campos de texto.

## COCO y precisión

Importa COCO de **segmentación de instancias**, seleccionando JSON, carpeta de imágenes y un proyecto nuevo. Admite polígonos y RLE comprimido/sin comprimir. No fusiona proyectos ni convierte automáticamente cajas sin máscara en segmentaciones. Los errores de entrada no generan proyectos parciales.

Exporta máscaras en RLE comprimido, preservando huecos y componentes separados; calcula área y bounding box desde la máscara final. Los objetos nuevos usan `iscrowd=0`; se preserva el valor importado. Compatible con `pycocotools.COCO.annToMask`. Algunos importadores externos solo aceptan polígonos: esta aplicación prioriza máscaras fieles en COCO RLE.

La imagen se muestra usando su matriz de píxeles almacenada, sin aplicar automáticamente la orientación EXIF. Así coinciden dimensiones y coordenadas con las anotaciones COCO originales. No se modifican los archivos de imagen.

## Desarrollo y comprobaciones

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
node --experimental-strip-types --test tests/frontend-masks.mjs
cd frontend
npm run test
npm run build
```

Para desarrollo: backend con `python3 run.py --dev --no-browser`; frontend con `npm run dev` desde `frontend/` (proxy local de Vite).

La prueba real de SAM 3 está separada de las pruebas de geometría, persistencia y API: requiere pesos autorizados. Consulta `scripts/benchmark_sam3.py --help` con el Python del entorno para medir inferencia en imágenes propias. No se considera validación del modelo una prueba con respuestas simuladas.

Las dependencias están fijadas en `requirements.txt`, `constraints.txt` y `frontend/package-lock.json`. El repositorio contiene el código de integración, no redistribuye pesos. SAM 3 está sujeto a la [licencia de Meta](https://github.com/facebookresearch/sam3/blob/main/LICENSE).
