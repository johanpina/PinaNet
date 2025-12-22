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
# Se asume que los modelos están en una carpeta 'models' en el mismo directorio del script
BASE_MODELS_PATH = "./models"
MODEL_MAP = {
    "binario": "clasificacion_binaria",
    "orden": "clasificacion_orden",
    "superfamilia": "clasificacion_superfamilia"
}

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
        lstm_output, _ = self.bilstm(sequence_output)
        logits = self.classifier(self.dropout(lstm_output))
        return logits

# --- 1. DATASET PARA PARALELIZACIÓN ---
class GenomeChunkDataset(Dataset):
    def __init__(self, fasta_path: str, tokenizer, chunk_size=50000, window_size=512, stride=128):
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
        print(f"LOG: Secuencia indexada, tiempo: {(time.time() - begin):.2f} segundos.")
        print(f"✅ {len(self.chunks_metadata)} fragmentos generados.")

    def __len__(self):
        return len(self.chunks_metadata)

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

# --- 2. POST-PROCESAMIENTO ---
def merge_annotations(raw_preds: List[Dict], gap_tolerance=10) -> List[Dict]:
    if not raw_preds: return []
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

def write_gff3(annotations: List[Dict], output_path: str, source="DNABERT2"):
    print(f"💾 Guardando {len(annotations)} anotaciones en {output_path}...")
    with open(output_path, "w") as f:
        f.write("##gff-version 3\n")
        for ann in annotations:
            line = f"{ann['seq_id']}\t{source}\t{ann['label']}\t{ann['start'] + 1}\t{ann['end']}\t.\t+\t.\tID={ann['label']}_{ann['start']}_{ann['end']}\n"
            f.write(line)

# --- 3. COMANDO CLI PRINCIPAL ---
@app.command()
def predict(
    fasta_file: str = typer.Argument(..., help="Archivo FASTA."),
    output_gff: str = typer.Argument(..., help="Archivo GFF3 de salida."),
    level: str = typer.Option("binary", help="Classification Level: binary, order, superfamilies."),
    num_workers: int = typer.Option(4, help="CPUs para pre-procesamiento."),
    device: str = typer.Option("cuda", help="Dispositivo (cuda/cpu).")
):
    level = level.lower()
    model_dir = f"./models/{level}/"
    
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"Using device: {device} ⚙️")

    # --- LÓGICA DE CARGA ---
    try:
        begin = time.time()
        tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        print(f"LOG: Tokenizador cargado, tiempo: {(time.time() - begin):.2f} segundos.")
        
        begin = time.time()
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            config_data = json.load(f)
        
        id2label = {int(k): v for k, v in config_data["id2label"].items()}
        label2id = config_data["label2id"]
        num_labels = len(id2label)
        print(f"LOG: Etiquetas cargadas, tiempo: {(time.time() - begin):.2f} segundos.")

        if level in ["binary", "superfamilies", "binario", "superfamilia", "order", "orden"]:
            typer.echo(f"🧠 Cargando Modelo Híbrido: {level}...")
            SAFE_CHECKPOINT = "quietflamingo/dnabert2-no-flashattention"
            begin = time.time()
            model = DNABERT_BiLSTM_NER(SAFE_CHECKPOINT, num_labels, id2label, label2id)
            weights_path = os.path.join(model_dir, "pytorch_model.bin")
            model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
            print(f"LOG: Modelo cargado, tiempo: {(time.time() - begin):.2f} segundos.")
        else:
            typer.echo(f"🧬 Cargando Modelo Estándar: {level}...")
            model = AutoModelForTokenClassification.from_pretrained(model_dir, trust_remote_code=True)
        
        begin = time.time()

        if torch.cuda.device_count() > 1 and device == "cuda":
            print(f"⚡ ¡Multi-GPU Detectado! Usando {torch.cuda.device_count()} GPUs en paralelo.")
            model = nn.DataParallel(model)

        model.to(device).eval()    
        print(f"LOG: Modelo listo, tiempo: {(time.time() - begin):.2f} segundos.")

    except Exception as e:
        typer.echo(f"❌ Error al cargar el modelo: {e}")
        raise typer.Exit(1)

    begin1 = time.time()
    
    # --- PIPELINE DE INFERENCIA ---
    dataset = GenomeChunkDataset(fasta_path=fasta_file, tokenizer=tokenizer)
    print(f"LOG: Dataset cargado, tiempo: {(time.time() - begin1):.2f} segundos.")
    begin = time.time()

    # Batch Size = 1 es lo más seguro. DataParallel paraleliza las ventanas internas.
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        num_workers=num_workers,    
        pin_memory=True
    )
    
    print(f"LOG: Dataloader listo, tiempo: {(time.time() - begin):.2f} segundos.")
    
    raw_predictions = []
    
    # Usamos no_grad para evitar el error de LSTM inplace
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Anotando"):
            
            # input_ids shape: [1, Num_Ventanas, 512]
            input_ids = batch['input_ids'][0].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'][0].to(device, non_blocking=True)

            # Aplanamos para DataParallel: [Total_Ventanas, 512]
            B, N, L = input_ids.shape
            input_ids_flat = input_ids.view(-1, L)
            mask_flat = attention_mask.view(-1, L)
            
            # Inferencia Multi-GPU
            # Output shape: [Total_Ventanas, 512, Num_Labels]
            outputs = model(input_ids_flat, attention_mask=mask_flat)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            # --- CORRECCIÓN CRÍTICA DE DIMENSIONES ---
            # 1. Recuperamos la forma [B, N, 512, Num_Labels]
            #    Antes faltaba el 512, por eso colapsaba todo.
            logits = logits.view(B, N, L, -1) 
            
            # 2. Argmax sobre la dimensión de las etiquetas (dimensión 3)
            #    Resultado shape: [B, N, 512]
            preds = torch.argmax(logits, dim=3).cpu().numpy() 

            # --- RECONSTRUCCIÓN ---
            for b in range(B): 
                chunk_preds_ids = preds[b]                 # Shape: [N, 512]
                chunk_offsets = batch['offset_mapping'][b] # Shape: [N, 512, 2]
                global_start = batch['global_start'][b].item()
                seq_id = batch['seq_id'][b]

                # Iteramos sobre cada ventana (N)
                for win_idx, window_pred_ids in enumerate(chunk_preds_ids):
                    # window_pred_ids es ahora un array de 512 enteros. ¡Correcto!
                    window_offsets = chunk_offsets[win_idx] 
                    
                    # Iteramos sobre los 512 tokens
                    for token_idx, label_id in enumerate(window_pred_ids):
                        label = id2label[label_id]
                        if label in ["Background", "O"]: continue
                        
                        start_l, end_l = window_offsets[token_idx]
                        if start_l == end_l: continue 

                        raw_predictions.append({
                            "seq_id": seq_id,
                            "start": global_start + start_l.item(),
                            "end": global_start + end_l.item(),
                            "label": label
                        })
            
    print(f"LOG: Inferencia finalizada.")
    begin = time.time()
    
    # --- POST-PROCESAMIENTO ---
    clean_annotations = merge_annotations(raw_predictions)
    print(f"LOG: Fusión completada, tiempo: {(time.time() - begin):.2f} segundos.")
    
    write_gff3(clean_annotations, output_gff)
    
    end = time.time()
    print(f"⏱️ Tiempo TOTAL del pipeline: {(end - begin1):.2f} segundos.")
    typer.secho(f"✅ ¡Finalizado! Resultados en {output_gff}", fg=typer.colors.GREEN, bold=True)
    
if __name__ == "__main__":
    app()