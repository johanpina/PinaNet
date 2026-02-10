import typer
import torch
import warnings
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel, AutoConfig
from Bio import SeqIO
from tqdm import tqdm
from typing import List, Dict
import torch.nn as nn
import json
import time

# Filtrar advertencias
warnings.filterwarnings("ignore")

app = typer.Typer(
    name="DNABERT-2 TE Annotator",
    help="CLI para detectar TEs a niveles Binario, Orden o Superfamilia."
)

# --- CONFIGURACIÓN DE RUTAS ---
BASE_MODELS_PATH = "./models"
os.environ["TOKENIZERS_PARALLELISM"] = 'True'

# --- 1. DEFINICIÓN DE LA ARQUITECTURA HÍBRIDA ---
class DNABERT_BiLSTM_NER(nn.Module):
    def __init__(self, checkpoint, num_labels, id2label, label2id):
        super().__init__()
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
        self.config = self.bert.config
        self.config.num_labels = num_labels
        self.config.id2label = id2label
        self.config.label2id = label2id
        
        input_dim = self.config.hidden_size 
        lstm_hidden_dim = 256 
        self.bilstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden_dim,
                              num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(lstm_hidden_dim * 2, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        
        # Flatten para optimización multi-GPU
        self.bilstm.flatten_parameters()
        
        lstm_output, _ = self.bilstm(sequence_output)
        logits = self.classifier(self.dropout(lstm_output))
        return logits

# --- 2. DATASET (MEGA-CHUNKS) ---
class GenomeChunkDataset(Dataset):
    # AJUSTE: Bajamos de 1MB a 200kb. 
    # Es el equilibrio perfecto: satura las 8 GPUs sin riesgo de OOM.
    def __init__(self, fasta_path: str, tokenizer, chunk_size=200000, window_size=512, stride=128):
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.stride = stride
        self.chunk_size = chunk_size
        self.chunks_metadata = []
        
        print(f"📖 Indexando secuencia FASTA: {fasta_path}...")
        begin = time.time()
        for record in SeqIO.parse(fasta_path, "fasta"):
            seq_len = len(record.seq)
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                self.chunks_metadata.append({
                    "seq_id": record.id,
                    "seq_str": str(record.seq[start:end]).upper(),
                    "global_start": start
                })
        print(f"LOG: Secuencia indexada en {(time.time() - begin):.2f}s. Total Chunks: {len(self.chunks_metadata)}")

    def __len__(self):
        return len(self.chunks_metadata)
        
    # ... (__getitem__ igual) ...
    def __getitem__(self, idx):
        meta = self.chunks_metadata[idx]
        tokens = self.tokenizer(
            meta["seq_str"],
            truncation=True,
            max_length=self.window_size,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "offset_mapping": tokens["offset_mapping"],
            "global_start": meta["global_start"],
            "seq_id": meta["seq_id"]
        }

# --- 3. POST-PROCESAMIENTO ---
def merge_annotations(raw_preds: List[Dict], gap_tolerance=10) -> List[Dict]:
    if not raw_preds: return []
    # Ordenar es vital para el merge
    raw_preds.sort(key=lambda x: (x['seq_id'], x['start']))
    merged = []
    current = raw_preds[0]
    
    for next_pred in raw_preds[1:]:
        if (next_pred['seq_id'] == current['seq_id'] and 
            next_pred['label'] == current['label'] and 
            next_pred['start'] <= current['end'] + gap_tolerance):
            current['end'] = max(current['end'], next_pred['end'])
        else:
            merged.append(current)
            current = next_pred
    merged.append(current)
    return merged

def write_gff3(annotations: List[Dict], output_path: str, source="PinaNet"):
    print(f"💾 Guardando {len(annotations)} anotaciones en {output_path}...")
    with open(output_path, "w") as f:
        f.write("##gff-version 3\n")
        for ann in annotations:
            # GFF es 1-based
            start = ann['start'] + 1
            end = ann['end']
            line = f"{ann['seq_id']}\t{source}\t{ann['label']}\t{start}\t{end}\t.\t+\t.\tID={ann['label']}_{start}_{end}\n"
            f.write(line)

def write_fasta_library(annotations: List[Dict], genome_path: str, output_path: str):
    """
    Genera una librería FASTA a partir de las anotaciones GFF.
    """
    print(f"📚 Generando librería FASTA en {output_path}...")
    try:
        genome = SeqIO.to_dict(SeqIO.parse(genome_path, "fasta"))
        with open(output_path, "w") as f:
            for i, ann in enumerate(annotations):
                seq_id = ann['seq_id']
                start = ann['start'] # 0-based
                end = ann['end']
                label = ann['label']
                
                if seq_id in genome:
                    # Formato de cabecera: >TE_1#LTR/Gypsy
                    header = f">TE_{i+1}#{label}"
                    sequence = str(genome[seq_id].seq[start:end])
                    f.write(f"{header}\n{sequence}\n")
    except Exception as e:
        print(f"⚠️ Error al generar la librería FASTA: {e}")

# --- 4. COMANDO CLI PRINCIPAL ---
@app.command()
def predict(
    fasta_file: str = typer.Argument(..., help="Archivo FASTA."),
    output_gff: str = typer.Argument(..., help="Archivo GFF3 de salida."),
    level: str = typer.Option("binary", help="Classification Level: binary, order, superfamilies."),
    create_library: bool = typer.Option(True, help="Generar librería FASTA de secuencias candidatas."),
    num_workers: int = typer.Option(4, help="CPUs para pre-procesamiento."),
    chunk_size: int = typer.Option(1000000, help="Tamaño del chunk en pb. Ajustar según VRAM."),
    device: str = typer.Option("cuda", help="Dispositivo (cuda/cpu)."),
    gpu_ids: str = typer.Option(None, help="IDs de GPUs a usar, separados por coma (ej: '0,1,2'). Por defecto usa todas."),
    num_gpus: int = typer.Option(0, help="Número máximo de GPUs a usar. 0 = usar todas las disponibles.")
):
    # Limpieza preventiva de memoria
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    level = level.lower()
    model_dir = f"./models/{level}/"
    
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # --- RESOLUCIÓN DE GPUs ---
    selected_device_ids = None  # None = usar todas (comportamiento por defecto de DataParallel)

    if device == "cuda" and torch.cuda.is_available():
        total_gpus = torch.cuda.device_count()

        if gpu_ids is not None:
            # El usuario especificó IDs exactos: --gpu-ids "0,2,3"
            try:
                selected_device_ids = [int(x.strip()) for x in gpu_ids.split(",")]
                # Validar que los IDs existan
                invalid_ids = [gid for gid in selected_device_ids if gid >= total_gpus or gid < 0]
                if invalid_ids:
                    typer.echo(f"⚠️ GPU IDs inválidos: {invalid_ids}. Disponibles: 0-{total_gpus - 1}")
                    raise typer.Exit(1)
            except ValueError:
                typer.echo(f"❌ Formato inválido para --gpu-ids. Usa números separados por coma, ej: '0,1,2'")
                raise typer.Exit(1)
        elif num_gpus > 0:
            # El usuario especificó cuántas GPUs usar: --num-gpus 4
            effective_num = min(num_gpus, total_gpus)
            selected_device_ids = list(range(effective_num))

        # Información de GPUs
        if selected_device_ids is not None:
            print(f"🎯 GPUs seleccionadas: {selected_device_ids} (de {total_gpus} disponibles)")
            for gid in selected_device_ids:
                print(f"   GPU {gid}: {torch.cuda.get_device_name(gid)}")
            primary_device = f"cuda:{selected_device_ids[0]}"
        else:
            print(f"🎯 Usando todas las GPUs disponibles: {total_gpus}")
            for gid in range(total_gpus):
                print(f"   GPU {gid}: {torch.cuda.get_device_name(gid)}")
            primary_device = "cuda:0"

        device = primary_device

    print(f"Using device: {device} ⚙️")

    try:
        begin = time.time()
        tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            config_data = json.load(f)
        
        id2label = {int(k): v for k, v in config_data["id2label"].items()}
        label2id = config_data["label2id"]
        num_labels = len(id2label)
        bg_id = label2id.get("Background", label2id.get("0", 0))

        if level in ["binary", "superfamilies", "binario", "superfamilia", "order", "orden"]:
            typer.echo(f"🧠 Cargando Modelo Híbrido: {level}...")
            SAFE_CHECKPOINT = "quietflamingo/dnabert2-no-flashattention"
            model = DNABERT_BiLSTM_NER(SAFE_CHECKPOINT, num_labels, id2label, label2id)
            weights_path = os.path.join(model_dir, "pytorch_model.bin")
            model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        else:
            typer.echo(f"🧬 Cargando Modelo Estándar: {level}...")
            model = AutoModelForTokenClassification.from_pretrained(model_dir, trust_remote_code=True)

        use_multi_gpu = (
            "cuda" in device
            and torch.cuda.device_count() > 1
            and (selected_device_ids is None or len(selected_device_ids) > 1)
        )
        if use_multi_gpu:
            ngpus = len(selected_device_ids) if selected_device_ids else torch.cuda.device_count()
            print(f"⚡ ¡Multi-GPU Activado! Usando {ngpus} GPUs con device_ids={selected_device_ids or 'todas'}.")
            model = nn.DataParallel(model, device_ids=selected_device_ids)

        model.to(device).eval()    
    except Exception as e:
        typer.echo(f"❌ Error al cargar: {e}")
        raise typer.Exit(1)

    begin1 = time.time()
    
    # 200KB Chunk Size
    dataset = GenomeChunkDataset(
        fasta_path=fasta_file, 
        tokenizer=tokenizer,
        chunk_size=chunk_size  # <--- USAMOS LA VARIABLE
    )
    
    print(f"LOG: Dataset cargado. Chunk Size: {chunk_size/1000} kb.")

    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        num_workers=num_workers,    
        pin_memory=True if "cuda" in device else False
    )
    
    final_annotations = []
    
    print("🚀 Iniciando Inferencia Vectorizada (FP16)...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Procesando"):
            
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            if input_ids.dim() == 3:
                input_ids = input_ids.squeeze(0)
                attention_mask = attention_mask.squeeze(0)
            
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            
            # --- OPTIMIZACIÓN: FP16 (Mixed Precision) ---
            # Esto reduce el consumo de VRAM a la mitad y acelera en RTX
            autocast_device = "cuda" if "cuda" in device else "cpu"
            with torch.amp.autocast(device_type=autocast_device, dtype=torch.float16):
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                
                # Des-aplanar dimensiones antes de salir de la GPU
                B_windows, L, _ = logits.shape
                logits = logits.view(B_windows, L, -1)
                
                # Argmax en GPU es más rápido
                preds = torch.argmax(logits, dim=2).cpu().numpy()

            # --- RECONSTRUCCIÓN VECTORIZADA ---
            offset_mapping = batch['offset_mapping'][0].numpy() 
            global_start = batch['global_start'].item()
            seq_id = batch['seq_id'][0]

            # Filtro rápido
            valid_mask = (preds != bg_id) & (offset_mapping[:, :, 0] != offset_mapping[:, :, 1])
            
            if not np.any(valid_mask):
                continue 

            valid_labels_ids = preds[valid_mask]
            valid_starts_local = offset_mapping[:, :, 0][valid_mask]
            valid_ends_local = offset_mapping[:, :, 1][valid_mask]
            
            valid_starts_global = valid_starts_local + global_start
            valid_ends_global = valid_ends_local + global_start
            
            chunk_results = [
                {
                    "seq_id": seq_id,
                    "start": int(s), 
                    "end": int(e),
                    "label": id2label[l]
                }
                for s, e, l in zip(valid_starts_global, valid_ends_global, valid_labels_ids)
            ]
            
            chunk_merged = merge_annotations(chunk_results)
            final_annotations.extend(chunk_merged)
            
    print(f"LOG: Inferencia finalizada.")
    print("🧩 Realizando fusión final...")
    final_clean_annotations = merge_annotations(final_annotations)
    write_gff3(final_clean_annotations, output_gff)
    
    if create_library:
        fasta_library_path = f"{output_gff}.fasta"
        write_fasta_library(final_clean_annotations, fasta_file, fasta_library_path)

    end = time.time()
    print(f"⏱️ Tiempo TOTAL: {(end - begin1):.2f} s")
    typer.secho(f"✅ ¡Finalizado!", fg=typer.colors.GREEN, bold=True)

if __name__ == "__main__":
    app()