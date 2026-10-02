# Contrato de implementación

API local `/api`, un proyecto abierto por proceso. Errores HTTP `{detail: string}`. Todo JSON usa snake_case. No autenticación remota; localhost y origen validado. Coordenadas píxel originales, lectura de raster sin aplicar EXIF para mantener COCO.

## Tipos
- Category `{id:number,name:string,color:string,supercategory?:string}`
- Image `{id:number,file_name:string,width:number,height:number,annotation_count:number}`
- Project `{name:string,directory:string,image_root:string,categories:Category[],images:Image[]}`
- Mask: RLE COCO comprimido `{size:[height,width],counts:string}`.
- Component `{outer:[number,number][],holes:[number,number][][]}`
- Annotation `{id:string,category_id:number,mask:Mask,components:Component[],controls?:Component[],preview?:string,iscrowd:number}`. IDs string UUID internos; almacenamiento asigna IDs enteros estables para COCO. Preview es data URL PNG coloreada. `components` describe la máscara exacta; `controls` simplifica puntos editables sin modificar la máscara hasta una edición manual.
- Point `{x:number,y:number,label:0|1}`
- Part `{id:string,points:Point[],box?:[x1,y1,x2,y2],polygon?:{vertices:[number,number][],closed:boolean},mask?:Mask,seed_mask?:Mask,preview?:string,components?:Component[],controls?:Component[]}`. `polygon` conserva el dibujo original, incluido un borrador abierto; sus coordenadas son píxeles originales. Refinar rasteriza el polígono cerrado mediante `/geometry`, guarda ese RLE como `seed_mask` y llama `/infer/points` sin requerir clics ni caja. Editar el polígono invalida su máscara inicial, resultado y prompts de corrección.
- Draft `{id:string,category_id:number,parts:Part[],active_part_id:string}`
- Proposal igual Annotation + `{score:number,selected:boolean}`.
- ImageState `{image_id:number,revision:number,annotations:Annotation[],draft:Draft|null,proposals:Proposal[]}`

## Rutas
- GET `/health` -> `{status,model:{state,device,message}}`
- GET `/browse?path=...` -> `{path,parent,directories:[{name,path}],files:[{name,path}],exists:true}` (carpetas e imágenes y JSON).
- POST `/projects/open` `{directory,image_root?:string,files?:string[],recursive?:boolean}` -> Project. Crear requiere `image_root` explícito. Reabrir sin `image_root` conserva la raíz guardada. El explorador comienza en `directory` y no añade una subcarpeta automáticamente. `files` relativos raíz; omitido incorpora todas imágenes. No guardar metadata dentro raíz de imágenes (directory igual o descendiente de image_root inválido).
- GET `/project` -> Project | null
- POST `/project/relink` `{image_root}` -> Project
- POST `/project/categories` `{name,color}` -> Category
- PATCH `/project/categories/{id}` `{name?,color?}` -> Category
- GET `/images/{id}/file?thumbnail=true|false` -> imagen PNG (raster original, sin giro EXIF)
- GET `/images/{id}/state` -> ImageState
- PUT `/images/{id}/state` ImageState -> ImageState (revision debe coincidir, incremento servidor; conflicto 409). Autosave serializado frontend. Servidor valida máscaras/geometría/referencias. Guardar preview no necesario.
- POST `/geometry` `{image_id,components}` -> `{mask,components,controls,preview}`
- POST `/masks/union` `{image_id,masks:Mask[]}` -> `{mask,components,controls,preview}`
- POST `/masks/fill-holes` `{image_id,mask:Mask,max_area?:number=16}` -> `{mask,components,controls,preview,filled_holes,filled_pixels}`. Límite entero de 1 a 150000000 píxeles originales. Rellena solo regiones de fondo encerradas con área menor o igual al límite (conectividad por lados, no diagonales); conserva fondo conectado al borde, huecos mayores y todos los píxeles originales del objeto. No escribe en la base de datos: el frontend aplica el resultado al objeto seleccionado como edición con deshacer/rehacer y autoguardado. Un resultado sin cambios no crea historial.
- POST `/infer/points` `{image_id,revision,part:Part}` -> `{image_id,revision,mask,components,controls,preview}`
- POST `/infer/text` `{image_id,revision,text,category_id,source_language?:"en"|"es"}` -> `{image_id,revision,proposals:Proposal[],prompt:{original,english,source_language}}`. Inglés predeterminado sin traductor; español se traduce localmente a inglés antes de SAM. Fallos de traducción: 503; el frontend conserva el texto original. El texto no crea ni renombra clases.
- POST `/infer/visual` `{image_id,revision,category_id,reference_image,reference_box?,text?,source_language?:"en"|"es"}` -> `{image_id,revision,proposals:Proposal[],reference:{width,height},method:"cross_image_exemplar",prompt?:{original,english,source_language}}`. `reference_image` contiene base64 puro de PNG/JPEG/WebP (sin prefijo data URL), máximo 10 MiB decodificados y 16 millones de píxeles; no admite animaciones. La caja opcional `[x1,y1,x2,y2]` usa píxeles de la referencia después de aplicar EXIF y debe estar dentro de sus dimensiones. Sin caja se usa toda la referencia. Texto opcional; solo el texto español no vacío se traduce. Archivos/cajas inválidos: 422; límites de carga: 413; fallo de traducción o modelo: 503. No escribe el archivo subido ni modifica anotaciones: las propuestas se guardan mediante el flujo normal de estado. La búsqueda entre imágenes es una adaptación experimental del codificador de ejemplos visuales de SAM 3.
- POST `/model/load` -> model status (lazy loading, await complete)
- POST `/coco/import` `{directory,image_root,json_path}` -> Project (nuevo proyecto; no fusionar)
- POST `/coco/export` `{}` -> `{path,file_name}` (archivo anotaciones.coco.json en raíz proyecto, escritura atómica)
- GET `/coco/download` -> JSON exportado

## Módulos de backend asignados
Las solicitudes de proyecto incluyen `X-Project-Directory` (ruta codificada con encodeURIComponent); imágenes y descargas usan `?project=...`. El servidor valida y captura el proyecto al entrar cada petición. Si otra pestaña abrió otro proyecto, devuelve 409 en lugar de mezclar cambios.

`app/storage.py`: clase ProjectStore(directory), classmethods open(directory,image_root=None,files=None,recursive=True), import_coco(directory,image_root,json_path); methods project(), get_state(image_id), save_state(image_id,state), image_path(image_id), relink(image_root), add_category(name,color), update_category(id,updates), export_coco() -> Path. Excepciones ValueError para entradas y conflicto RevisionConflict.
`app/geometry.py`: encode_mask(np bool)->RLE, decode_mask(RLE)->np bool, mask_payload(mask,color='#8B8DE3')->{mask,components,preview}, rasterize_components(components,width,height)->np bool, union_masks(masks,width,height)->np bool.
`app/inference.py`: clase Sam3Engine: status()->dict, load()->dict, predict_points(image PIL, image_key:str, part:dict)->np bool, predict_text(image PIL,image_key:str,text:str)->list[{mask:np bool,score:float}], lock thread RLock. Métodos bloqueantes llamados threadpool API. SAM3 real únicamente. Recuperar seed_mask para texto→tracker. Cachés separadas y limitadas.

`Sam3Engine.predict_visual(image PIL,image_key:str,reference PIL,reference_box=None,text=None)` retorna la misma lista de propuestas. Las características geométricas se extraen de la referencia, se combinan con los tokens del texto opcional y se usan con las características de la imagen destino. Sin texto se usa el marcador interno `visual`, igual que el processor oficial para cajas. No se construye un montaje de imágenes ni se convierten los ejemplos en descripciones. Las máscaras y los logits para refinamiento pertenecen exclusivamente al raster de destino. La caché de ejemplos incluye su contenido y caja; se limpia junto con las demás cachés al cambiar de dispositivo. El flujo depende de los componentes de Transformers 5.18 fijados en requisitos.

`app/translation.py`: PromptTranslator.translate(text, source_language="en") -> str. Modelo público `Helsinki-NLP/opus-mt-es-en`, revisión `c96e2c5399ebfae4fc43d9669556b9afa74bb69d`; CPU, carga diferida, pesos en `.cache/huggingface`, caché de hasta 128 traducciones. Primer uso español descarga los pesos; después permite uso sin conexión. No se envían imágenes ni prompts a un servicio de traducción.
