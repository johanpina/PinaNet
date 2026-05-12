# 🧬 TE-GER: Transposable Elements Genomic Entity Recognition

**TE-GER** is a high-performance bioinformatics tool based on **Deep Learning** for the automatic detection and annotation of Transposable Elements (TEs) in raw genomic sequences (FASTA).

The system uses a state-of-the-art hybrid architecture that combines the representational power of **DNABERT-2** (a language model pre-trained on DNA) with bidirectional recurrent neural networks (**BiLSTM**) to capture the sequential and structural context of TEs.

> **Model weights are hosted on Hugging Face and download automatically on first run — no manual setup required.**
>
> | Model | HuggingFace | Task |
> |:------|:------------|:-----|
> | Binary | [Jspinad/te-ger-binary](https://huggingface.co/Jspinad/te-ger-binary) | TE vs Background |
> | Order | [Jspinad/te-ger-order](https://huggingface.co/Jspinad/te-ger-order) | LTR, LINE, SINE, TIR, … |
> | Superfamilies | [Jspinad/te-ger-superfamilies](https://huggingface.co/Jspinad/te-ger-superfamilies) | Gypsy, Copia, HAT, … (21 classes) |

---

## 🚀 Key Features

*   **Advanced Hybrid Architecture:** Integrates DNABERT-2 (for rich k-mer embeddings) + BiLSTM (for sequential memory) + Linear Classifier.
*   **Configurable Multi-GPU Support:** Automatically detects and uses all available GPUs by default (DataParallel). Optionally, you can select specific GPU IDs (`--gpu-ids`) or limit the number of GPUs (`--num-gpus`) to fine-tune resource usage.
*   **Vectorized Inference:** Uses matrix operations (NumPy/PyTorch) and mixed precision (FP16) for post-processing, eliminating CPU bottlenecks.
*   **3 Levels of Classification:**
    *   **Binary:** Presence/absence detection (TE vs. Background).
    *   **Order:** General taxonomic classification (e.g., LTR, LINE, SINE, DNA).
    *   **Superfamily:** Detailed taxonomic classification (e.g., Gypsy, Copia, Mutator, etc.).
*   **"Mega-Chunks" Strategy:** Processes the genome in massive configurable fragments (e.g., 1,000,000 - 5,000,000 bp) to saturate VRAM and minimize communication overhead.
*   **Standard Output:** Generates **GFF3** files compatible with IGV, JBrowse, and other genomic viewers.

---

## 🛠️ Installation

### Prerequisites
*   **Python 3.9** or higher.
*   (Recommended) NVIDIA GPU with CUDA drivers installed for fast inference.
*   Git.

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/johanpina/TE-GER.git
cd TE-GER

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # Linux / Mac
# .\venv\Scripts\activate      # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run — weights download automatically from HuggingFace on first execution
python Te_annotator.py genome.fasta output.gff3 --level binary
```

On first run, TE-GER detects that the model weights are missing and downloads them directly from HuggingFace (~460 MB per level). Subsequent runs use the cached local copy.

```
📥 Model weights not found locally. Downloading from HuggingFace: Jspinad/te-ger-binary
   Destination: ./models/binary/
✅ Weights downloaded successfully.
🧠 Loading Hybrid Model: binary...
```

---

## 📂 Model Weights

The trained weights are hosted publicly on Hugging Face and **do not need to be downloaded manually**. TE-GER handles this automatically.

| Level | Repo | Size | Labels |
|:------|:-----|:----:|:-------|
| `binary` | [Jspinad/te-ger-binary](https://huggingface.co/Jspinad/te-ger-binary) | ~460 MB | Background, TE |
| `order` | [Jspinad/te-ger-order](https://huggingface.co/Jspinad/te-ger-order) | ~460 MB | Background, DIRS, HELITRON, LINE, LTR, PLE, SINE, TIR |
| `superfamilies` | [Jspinad/te-ger-superfamilies](https://huggingface.co/Jspinad/te-ger-superfamilies) | ~460 MB | 21 superfamilies (Gypsy, Copia, HAT, Mutator, …) |

Once downloaded, the weights are stored in `./models/{level}/` and reused on every subsequent run.

---

## 💻 Usage

The program is run from the command line (CLI). The basic syntax is:

```bash
python Te_annotator.py [ARGUMENTS] [OPTIONS]
```

### Main Arguments

| Argument     | Description                                      | Required | 
| :----------- | :----------------------------------------------- | :------: |
| `fasta_file` | Path to the input file (`.fasta`, `.fa`, `.fna`). |    ✅    |
| `output_gff` | Path where the annotation file will be saved (`.gff3`). |    ✅    |

### Options and Parameters

| Option       | Command         | Description                                                                                             | Default     |
| :----------- | :-------------- | :------------------------------------------------------------------------------------------------------ | :---------- |
| **Level**    | `--level`       | Classification level: `binary`, `order`, `superfamilies`.                                               | `binary`    |
| **Create Library** | `--create-library` | Generate a FASTA library of candidate TE sequences. Use `--no-create-library` to disable.             | `True`      |
| **Chunk Size** | `--chunk-size`  | Size of the genome fragment to process in memory (base pairs). **Increase for higher speed, decrease if there is a memory error.** | `2,000,000`   |
| **Workers**  | `--num-workers` | CPU threads for data loading. It is recommended to keep it low (2) as GPU inference is very fast.      | `4`         |
| **Device**   | `--device`      | Execution device: `cuda` (GPU) or `cpu`.                                                                | `cuda`      |
| **GPU IDs**  | `--gpu-ids`     | Comma-separated list of specific GPU IDs to use (e.g., `0,2,3`). Useful when sharing a server or when certain GPUs are busy. | `None` (all) |
| **Num GPUs** | `--num-gpus`    | Maximum number of GPUs to use, selected sequentially starting from GPU 0. `0` means use all available.  | `0` (all)   |

---

## 🧪 Execution Examples

### 1. Binary Detection (Fast)
Scans the genome using a large chunk (2,000,000 bp) for maximum speed on GPUs with good VRAM (e.g., 24GB+).

```bash
python Te_annotator.py \
    ./test/corn_genome.fasta \
    ./results/binary_detection.gff3 \
    --level binary \
    --chunk-size 2000000 \
    --num-workers 2
```

### 2. Classification by Order (Balanced)
Standard configuration for mid-range GPUs (12GB - 16GB VRAM). 1,000,000 bp chunk.

```bash
python Te_annotator.py \
    ./test/rice_genome.fasta \
    ./results/order_classification.gff3 \
    --level order \
    --chunk-size 1000000 \
    --device cuda
```

### 3. Fine-grained Classification (Safe)
The most detailed analysis. If you have low free VRAM, use the default chunk size (200,000 bp).

```bash
python Te_annotator.py \
    ./test/unknown_genome.fasta \
    ./results/full_annotation.gff3 \
    --level superfamilies \
    --chunk-size 200000
```

### 4. Multi-GPU Configuration Examples

By default, TE-GER detects and uses **all available GPUs** automatically. The following options allow you to control which and how many GPUs are used.

**Use only specific GPUs by their IDs** (e.g., GPUs 0 and 2 on a 4-GPU server):

```bash
python Te_annotator.py \
    ./test/corn_genome.fasta \
    ./results/binary_detection.gff3 \
    --level binary \
    --gpu-ids "0,2"
```

**Limit to a fixed number of GPUs** (e.g., use only the first 2 GPUs on an 8-GPU node):

```bash
python Te_annotator.py \
    ./test/corn_genome.fasta \
    ./results/order_classification.gff3 \
    --level order \
    --num-gpus 2
```

**Run on a single GPU** (useful for debugging or when sharing resources):

```bash
python Te_annotator.py \
    ./test/corn_genome.fasta \
    ./results/binary_detection.gff3 \
    --level binary \
    --gpu-ids "0"
```

**Force CPU execution** (no GPU required):

```bash
python Te_annotator.py \
    ./test/corn_genome.fasta \
    ./results/binary_detection.gff3 \
    --level binary \
    --device cpu
```

> **Note:** If neither `--gpu-ids` nor `--num-gpus` is specified, TE-GER will automatically use all available GPUs. The `--gpu-ids` option takes precedence over `--num-gpus` if both are provided.

---

## 📊 Output Formats

### GFF3 Annotation File

The main output file follows the **GFF3** (Generic Feature Format version 3) standard. Example:

```gff
##gff-version 3
chr1    TE-GER    LTR 10500   12400   .   +   .   ID=LTR_10500_12400;Name=LTR_prediction
chr1    TE-GER    LINE    15000   15800   .   +   .   ID=LINE_15000_15800;Name=LINE_prediction
```

*   **Column 1 (SeqID):** Sequence ID (chromosome/contig).
*   **Column 2 (Source):** Source (`TE-GER`).
*   **Column 3 (Type):** TE type (Model prediction, e.g., `LTR`).
*   **Column 4-5 (Start-End):** 1-based coordinates.
*   **Column 9 (Attributes):** Unique ID and metadata for visualization.

### Candidate TE FASTA Library

By default (or with `--create-library`), the tool also generates a FASTA file containing the DNA sequences of all predicted TEs. The output path will be the same as the GFF file, but with a `.fasta` extension (e.g., `full_annotation.gff3.fasta`).

The FASTA headers are formatted similarly to RepeatModeler 2 to be easily parsable and informative:

```fasta
>TE_1#LTR
AGCT...
>TE_2#LINE
TTCA...
```

*   The ID is a unique sequential number for each candidate (`TE_1`, `TE_2`, etc.).
*   The classification is appended after a `#` symbol, taken directly from the model's prediction.

This library is useful for downstream analyses like building consensus sequences, BLASTing against other databases, or manual inspection.

---

## 📚 Library Builder (Clustering + Consensus Pipeline)

TE-GER includes a second script, `library_builder.py`, that takes the candidate FASTA library and generates a **consensus library** through clustering and multiple sequence alignment.

### Pipeline

```
Candidate FASTA → MMseqs2 (clustering) → MAFFT (MSA per cluster) → CIAlign (consensus per MSA)
```

### Additional Dependencies

These tools must be installed in your conda environment before using `library_builder.py`:

```bash
conda install -c bioconda -c conda-forge mmseqs2 mafft
pip install cialign
```

### Usage

```bash
python library_builder.py [FASTA_INPUT] [OUTPUT_DIR] [OPTIONS]
```

### Arguments

| Argument      | Description                                              | Required |
| :------------ | :------------------------------------------------------- | :------: |
| `fasta_input` | FASTA file of candidate TEs (output from `Te_annotator.py`). |    ✅    |
| `output_dir`  | Directory where all results will be saved.               |    ✅    |

### Options

| Option              | Command              | Description                                                              | Default |
| :------------------ | :------------------- | :----------------------------------------------------------------------- | :------ |
| **Min Seq ID**      | `--min-seq-id`       | Minimum sequence identity for MMseqs2 clustering (0-1).                  | `0.8`   |
| **Coverage**        | `--coverage`         | Minimum alignment coverage for MMseqs2 clustering (0-1).                 | `0.8`   |
| **Threads**         | `--threads`          | CPU threads for MMseqs2 and MAFFT.                                       | `4`     |
| **Workers**         | `--workers`          | Parallel processes for MAFFT + CIAlign (multiprocessing).                | `4`     |
| **Min Cluster Size**| `--min-cluster-size` | Minimum sequences in a cluster to generate MSA. Smaller clusters are skipped. | `2`     |

### Examples

**Basic run using default parameters:**

```bash
python library_builder.py \
    ./results/binary_detection.gff3.fasta \
    ./results/library_output/
```

**Fine-tune clustering stringency and parallelism:**

```bash
python library_builder.py \
    ./results/order_classification.gff3.fasta \
    ./results/library_output/ \
    --min-seq-id 0.6 \
    --coverage 0.7 \
    --threads 8 \
    --workers 6 \
    --min-cluster-size 3
```

**Include singleton clusters (clusters with 1 sequence):**

```bash
python library_builder.py \
    ./results/binary_detection.gff3.fasta \
    ./results/library_output/ \
    --min-cluster-size 1
```

### Output Structure

```text
output_dir/
├── clusterRes_cluster.tsv          ← MMseqs2 cluster assignments
├── clusterRes_rep_seq.fasta        ← Representative sequences
├── clusterRes_all_seqs.fasta       ← All sequences with clusters
├── tmp/                            ← MMseqs2 temp files
├── clusters/
│   ├── cluster_0.fasta             ← Sequences per cluster
│   ├── cluster_0_msa.fasta         ← MAFFT alignment per cluster
│   └── ...
├── consensus/
│   ├── cluster_0_consensus.fasta   ← CIAlign consensus per cluster
│   └── ...
└── consensus_library.fasta         ← FINAL CONSENSUS LIBRARY
```

The final file `consensus_library.fasta` contains one consensus sequence per cluster and can be used directly with tools like RepeatMasker (`-lib consensus_library.fasta`).

---

## ⚙️ System Architecture

TE-GER solves the problem of the limited input length of BERT-like models through an optimized **"Divide and Conquer"** strategy:

1.  **Mega-Chunking:** The genome is divided into large fragments (e.g., 1,000,000 - 2,000,000 bp) that are loaded into VRAM at once.
2.  **Parallel Sliding Window:** Each Mega-Chunk contains thousands of 512bp windows. These are automatically distributed among the selected GPUs (all by default, or a user-defined subset via `--gpu-ids` / `--num-gpus`).
3.  **Hybrid Inference (FP16):**
    *   **DNABERT-2:** Extracts deep features from the DNA sequence.
    *   **BiLSTM:** Analyzes the sequential context.
4.  **Vectorized Reconstruction:** Predictions are decoded using NumPy boolean masks, avoiding slow Python loops and allowing millions of bases to be processed per second.

---

## ⚠️ Common Troubleshooting

*   **Error `CUDA Out of memory`:** The fragment is too large for your GPU VRAM. Reduce `--chunk-size` (e.g., from `1000000` to `200000`).
*   **Slow or failed download:** The first run downloads ~460 MB from HuggingFace. Make sure you have an active internet connection. If the download is interrupted, delete the incomplete `./models/{level}/` folder and run again.
*   **`No HuggingFace repo configured` error:** Check that `--level` is one of `binary`, `order`, or `superfamilies`.
*   **`Triton / Flash Attention` warnings:** Normal on GPUs older than Ampere/Hopper. TE-GER switches to a compatible attention implementation automatically.

---

## 📝 License

This project is licensed under the [MIT](LICENSE) License.

---
**Developed by Johan S. Piña - 2025**
