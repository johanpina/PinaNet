# 🧬 PinaNet: Deep Learning Transposable Element Annotator

**PinaNet** es una herramienta bioinformática de alto rendimiento basada en **Deep Learning** para la detección y anotación automática de Elementos Transponibles (TEs) en secuencias genómicas crudas (FASTA).

El sistema utiliza una arquitectura híbrida de última generación que combina la capacidad de representación de **DNABERT-2** (un modelo de lenguaje pre-entrenado en ADN) con redes neuronales recurrentes bidireccionales (**BiLSTM**) para capturar el contexto secuencial y estructural de los TEs.

---

## 🚀 Características Principales

* **Arquitectura Híbrida Avanzada:** Integra DNABERT-2 (para embeddings ricos de k-mers) + BiLSTM (para memoria secuencial) + Clasificador Lineal.
* **Soporte Multi-GPU Automático:** Detecta y utiliza automáticamente todas las GPUs disponibles (DataParallel) para dividir la carga de trabajo y acelerar la inferencia exponencialmente.
* **Inferencia Vectorizada:** Utiliza operaciones de matrices (NumPy/PyTorch) y precisión mixta (FP16) para el post-procesamiento, eliminando los cuellos de botella de la CPU.
* **3 Niveles de Clasificación:**
    * **Binario:** Detección de presencia/ausencia (TE vs Background).
    * **Orden:** Clasificación taxonómica general (ej. LTR, LINE, SINE, DNA).
    * **Superfamilia:** Clasificación taxonómica detallada (ej. Gypsy, Copia, Mutator, etc.).
* **Estrategia "Mega-Chunks":** Procesa el genoma en fragmentos masivos configurables (ej. 1MB - 5MB) para saturar la memoria VRAM y minimizar la sobrecarga de comunicación.
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
| **Nivel** | `--level` | Nivel de clasificación: `binary`, `order`, `superfamilies`. | `binary` |
| **Chunk Size** | `--chunk-size` | Tamaño del fragmento de genoma a procesar en memoria (pares de bases). **Aumentar para mayor velocidad, disminuir si hay error de memoria.** | `200000` |
| **Workers** | `--num-workers` | Hilos de CPU para cargar datos. Se recomienda mantener bajo (2) ya que la inferencia GPU es muy rápida. | `4` |
| **Device** | `--device` | Dispositivo de ejecución: `cuda` (GPU) o `cpu`. | `cuda` |

---

## 🧪 Ejemplos de Ejecución

### 1. Detección Binaria (Rápida)
Escanea el genoma usando un chunk grande (2MB) para máxima velocidad en GPUs con buena VRAM (ej. 24GB+).

```bash
python Te_annotator.py \
    ./test/genoma_maiz.fasta \
    ./resultados/deteccion_binaria.gff3 \
    --level binary \
    --chunk-size 2000000 \
    --num-workers 2
```

### 2. Clasificación por Órdenes (Equilibrada)
Configuración estándar para GPUs de rango medio (12GB - 16GB VRAM). Chunk de 1MB.

```bash
python Te_annotator.py \
    ./test/genoma_arroz.fasta \
    ./resultados/clasificacion_ordenes.gff3 \
    --level order \
    --chunk-size 1000000 \
    --device cuda
```

### 3. Clasificación Fina (Segura)
El análisis más detallado. Si tienes poca VRAM libre, usa el chunk por defecto (200kb).

```bash
python Te_annotator.py \
    ./test/genoma_desconocido.fasta \
    ./resultados/full_annotation.gff3 \
    --level superfamilies \
    --chunk-size 200000
```

---

## 📊 Formato de Salida (GFF3)

El archivo generado sigue el estándar **GFF3** (Generic Feature Format versión 3). Ejemplo de salida:

```gff
##gff-version 3
chr1    DNABERT2    LTR 10500   12400   .   +   .   ID=LTR_10500_12400;Name=LTR_prediction
chr1    DNABERT2    LINE    15000   15800   .   +   .   ID=LINE_15000_15800;Name=LINE_prediction
```

* **Columna 1 (SeqID):** ID de la secuencia (cromosoma/contig).
* **Columna 2 (Source):** Fuente (`DNABERT2`).
* **Columna 3 (Type):** Tipo de TE (Predicción del modelo, ej. `LTR`).
* **Columna 4-5 (Start-End):** Coordenadas 1-based.
* **Columna 9 (Attributes):** ID único y metadatos para visualización.

---

## ⚙️ Arquitectura del Sistema

PinaNet resuelve el problema de la longitud de entrada limitada de los modelos tipo BERT mediante una estrategia de **"Divide y Vencerás"** optimizada:

1.  **Mega-Chunking:** El genoma se divide en fragmentos grandes (ej. 1MB - 2MB) que se cargan en la VRAM de golpe.
2.  **Sliding Window Paralelo:** Cada Mega-Chunk contiene miles de ventanas de 512bp. Estas se distribuyen automáticamente entre todas las GPUs disponibles.
3.  **Inferencia Híbrida (FP16):**
    * **DNABERT-2:** Extrae características profundas de la secuencia de ADN.
    * **BiLSTM:** Analiza el contexto secuencial.
4.  **Reconstrucción Vectorizada:** Las predicciones se decodifican usando máscaras booleanas de NumPy, evitando bucles lentos de Python y permitiendo procesar millones de bases por segundo.

---

## ⚠️ Solución de Problemas Comunes

* **Error `CUDA Out of memory`:** Estás intentando procesar un fragmento demasiado grande para tu GPU. **Solución:** Reduce el parámetro `--chunk-size`. Prueba bajando de `1000000` a `200000`.
* **Error `Model not found`:** Verifica que hayas copiado las carpetas `binary`, `order` y `superfamilies` dentro de la carpeta `models/` y que los nombres coincidan exactamente.
* **Advertencias de `Triton / Flash Attention`:** Son normales si no tienes la arquitectura de GPU más reciente (Hopper/Ampere). El sistema está configurado para cambiar automáticamente a una implementación compatible.

---

## 📝 Licencia

Este proyecto está bajo la licencia [MIT](LICENSE).

---
**Desarrollado por Johan S. Piña - 2025**