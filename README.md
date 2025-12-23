# 🧬 PinaNet: Deep Learning Transposable Element Annotator

**PinaNet** es una herramienta bioinformática de alto rendimiento basada en **Deep Learning** para la detección y anotación automática de Elementos Transponibles (TEs) en secuencias genómicas crudas (FASTA).

El sistema utiliza una arquitectura híbrida de última generación que combina la capacidad de representación de **DNABERT-2** (un modelo de lenguaje pre-entrenado en ADN) con redes neuronales recurrentes bidireccionales (**BiLSTM**) para capturar el contexto secuencial y estructural de los TEs.

---

## 🚀 Características Principales

* **Arquitectura Híbrida Avanzada:** Integra DNABERT-2 (para embeddings ricos de k-mers) + BiLSTM (para memoria secuencial) + Clasificador Lineal.
* **3 Niveles de Clasificación:**
    * **Binario:** Detección de presencia/ausencia (TE vs Background).
    * **Orden:** Clasificación taxonómica general (ej. LTR, LINE, SINE, DNA).
    * **Superfamilia:** Clasificación taxonómica detallada (ej. Gypsy, Copia, Mutator, etc.).
* **Procesamiento Paralelo Eficiente:** Implementa un patrón Productor-Consumidor donde múltiples núcleos de CPU tokenizan y preparan el genoma mientras la GPU realiza la inferencia masiva.
* **Sliding Window Inteligente:** Procesa genomas completos de cualquier tamaño fragmentándolos en "chunks" de 50kb con ventanas deslizantes y fusión automática de predicciones adyacentes.
* **Salida Estándar:** Genera archivos **GFF3** compatibles con IGV, JBrowse y otros visores genómicos.

---

## 🛠️ Instalación

### 1. Prerrequisitos
* **Python 3.9** o superior.
* (Recomendado) GPU NVIDIA con drivers CUDA instalados para inferencia rápida.
* Git.

### 2. Clonar el Repositorio
```bash
git clone https://github.com/TU_USUARIO/PinaNet.git
cd PinaNet
```

### 3. Crear Entorno Virtual
Se recomienda aislar las dependencias para evitar conflictos:

```bash
# Crear entorno
python -m venv venv

# Activar en Linux/Mac
source venv/bin/activate

# Activar en Windows
.\venv\Scripts\activate
```

### 4. Instalar Dependencias
```bash
pip install -r requirements.txt
```
*(Asegúrese de que `torch`, `transformers`, `biopython`, `typer`, `pandas`, `numpy` y `tqdm` estén instalados).*

---

## 📂 Configuración de Modelos

Debido al gran tamaño de los pesos neuronales, los modelos entrenados **no se incluyen** en el control de versiones de Git. Debes copiar tus carpetas directamente de los archivos del servidor o solicitarlos al owner del proyecto.

La estructura de carpetas debe verse **exactamente** así para que el software los reconozca:

```text
PinaNet/
├── Te_annotator.py
├── models/
│   ├── binary/            <-- Archivos del modelo Binario
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── ...
│   ├── order/             <-- Archivos del modelo de Orden
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── ...
│   └── superfamilies/     <-- Archivos del modelo de Superfamilia
│       ├── config.json
│       ├── pytorch_model.bin
│       └── ...
└── ...
```

**Nota Importante:** Asegúrate de que cada carpeta contenga, como mínimo, el archivo de configuración `config.json` y los pesos del modelo `pytorch_model.bin`.

---

## 💻 Uso

El programa se ejecuta desde la línea de comandos (CLI). La sintaxis básica es:

```bash
python Te_annotator.py [ARGUMENTOS] [OPCIONES]
```

### Argumentos Principales

| Argumento | Descripción | Requerido |
| :--- | :--- | :---: |
| `fasta_file` | Ruta al archivo de entrada (`.fasta`, `.fa`, `.fna`). | ✅ |
| `output_gff` | Ruta donde se guardará el archivo de anotación (`.gff3`). | ✅ |

### Opciones y Parámetros

| Opción | Comando | Descripción | Default |
| :--- | :--- | :--- | :--- |
| **Nivel** | `--level` | Nivel de clasificación. Valores aceptados: `binary`, `order`, `superfamilies`. | `binary` |
| **Workers** | `--num-workers` | Número de núcleos de CPU para la tokenización en paralelo. | `4` |
| **Device** | `--device` | Dispositivo de ejecución: `cuda` (GPU) o `cpu`. | `cuda` |

---

## 🧪 Ejemplos de Ejecución

### 1. Detección Binaria (TE vs No-TE)
Escanea el genoma y marca regiones que contienen elementos transponibles sin clasificarlos. Útil para enmascaramiento rápido o detección de densidad.

```bash
python Te_annotator.py \
    ./test/genoma_maiz.fasta \
    ./resultados/deteccion_binaria.gff3 \
    --level binary \
    --num-workers 8 \
    --device cuda
```

### 2. Clasificación por Órdenes
Clasifica los elementos encontrados en grandes grupos taxonómicos (LTR, LINE, TIR, etc.).

```bash
python Te_annotator.py \
    ./test/genoma_arroz.fasta \
    ./resultados/clasificacion_ordenes.gff3 \
    --level order \
    --device cuda
```

### 3. Clasificación Fina (Superfamilias)
El análisis más detallado. Clasifica en familias específicas (Gypsy, Copia, etc.).

```bash
python Te_annotator.py \
    ./test/genoma_desconocido.fasta \
    ./resultados/full_annotation.gff3 \
    --level superfamilies \
    --num-workers 4
```

---

## 📊 Formato de Salida (GFF3)

El archivo generado sigue el estándar **GFF3** (Generic Feature Format versión 3). Ejemplo de salida:

```gff
##gff-version 3
chr1	DNABERT2	LTR	10500	12400	.	+	.	ID=LTR_10500_12400;Name=LTR_prediction
chr1	DNABERT2	LINE	15000	15800	.	+	.	ID=LINE_15000_15800;Name=LINE_prediction
```

* **Columna 1 (SeqID):** ID de la secuencia (cromosoma/contig).
* **Columna 2 (Source):** Fuente (`DNABERT2`).
* **Columna 3 (Type):** Tipo de TE (Predicción del modelo, ej. `LTR`).
* **Columna 4-5 (Start-End):** Coordenadas 1-based.
* **Columna 9 (Attributes):** ID único y metadatos para visualización.

---

## ⚙️ Arquitectura del Sistema

PinaNet resuelve el problema de la longitud de entrada limitada de los modelos tipo BERT mediante una estrategia de **"Divide y Vencerás"**:

1.  **Chunking:** El genoma se divide en fragmentos manejables de 50kbp (lazy loading).
2.  **Sliding Window:** Cada fragmento se subdivide en ventanas de 512 tokens con un solapamiento (*stride*) de 128 tokens para evitar pérdida de información en los bordes.
3.  **Inferencia Híbrida:**
    * **DNABERT-2:** Extrae características profundas de la secuencia de ADN.
    * **BiLSTM:** Analiza la secuencia de características en ambas direcciones para entender el contexto estructural.
4.  **Fusión:** Las predicciones de las ventanas se proyectan a coordenadas globales y los fragmentos adyacentes de la misma clase se fusionan en una sola anotación continua.

---

## ⚠️ Solución de Problemas Comunes

* **Error `CUDA Out of memory`:** El modelo es grande. Intenta reducir los trabajadores (`--num-workers 0`) para liberar RAM del sistema o asegura que ninguna otra aplicación use la VRAM. El *batch size* interno está optimizado a 1 (lo que equivale a procesar ~150 ventanas de 512bp en paralelo por cada chunk de 50kb).
* **Error `Model not found`:** Verifica que hayas copiado las carpetas `binary`, `order` y `superfamilies` dentro de la carpeta `models/` y que los nombres coincidan exactamente.
* **Advertencias de `Triton / Flash Attention`:** Son normales si no tienes la arquitectura de GPU más reciente (Hopper/Ampere). El sistema está configurado para cambiar automáticamente a una implementación compatible (PyTorch nativo).

---

## 📝 Licencia


---
**Desarrollado por Johan S. Piña - 2025**