import typer
import subprocess
import shutil
import os
import time
from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, List, Tuple

app = typer.Typer(
    name="PinaNet Library Builder",
    help="Pipeline para generar librerías consenso: MMseqs2 → MAFFT → CIAlign."
)

# --- 1. VERIFICACIÓN DE DEPENDENCIAS ---
def check_dependencies() -> bool:
    """
    Verifica que mmseqs y mafft estén instalados (obligatorios).
    CIAlign es opcional: si no está, se usa fallback en Python.
    Retorna True si CIAlign está disponible, False si no.
    """
    required = {
        "mmseqs": "MMseqs2 (conda install -c bioconda mmseqs2)",
        "mafft": "MAFFT (conda install -c bioconda mafft)",
    }
    missing = []
    for cmd, install_hint in required.items():
        if shutil.which(cmd) is None:
            missing.append(f"  - {cmd}: {install_hint}")

    if missing:
        typer.echo("❌ Dependencias obligatorias no encontradas en PATH:")
        for m in missing:
            typer.echo(m)
        typer.echo("\nInstala las dependencias faltantes y vuelve a ejecutar.")
        raise typer.Exit(1)

    cialign_available = shutil.which("CIAlign") is not None
    if cialign_available:
        typer.echo("✅ Todas las dependencias encontradas (mmseqs, mafft, CIAlign).")
    else:
        typer.echo("✅ mmseqs y mafft encontrados.")
        typer.echo("   ℹ️ CIAlign no encontrado. Se usará generador de consenso Python (fallback).")

    return cialign_available


# --- FALLBACK: CONSENSO EN PYTHON PURO ---
def python_consensus_from_msa(msa_path: str, consensus_path: str) -> bool:
    """
    Genera una secuencia consenso por voto mayoritario desde un MSA FASTA.
    No requiere dependencias externas. Retorna True si tuvo éxito.
    """
    from collections import Counter

    # Leer secuencias alineadas
    sequences = []
    current_seq = []
    with open(msa_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                current_seq = []
            else:
                current_seq.append(line)
        if current_seq:
            sequences.append("".join(current_seq))

    if not sequences:
        return False

    # Voto mayoritario por columna (ignorando gaps)
    aln_len = max(len(s) for s in sequences)
    consensus = []

    for col in range(aln_len):
        bases = []
        for seq in sequences:
            if col < len(seq):
                c = seq[col].upper()
                if c not in ("-", "."):
                    bases.append(c)

        if bases:
            most_common = Counter(bases).most_common(1)[0][0]
            consensus.append(most_common)

    if not consensus:
        return False

    consensus_seq = "".join(consensus)
    with open(consensus_path, "w") as f:
        f.write(f">consensus\n{consensus_seq}\n")

    return True


# --- 2. CLUSTERING CON MMSEQS2 ---
def run_mmseqs_clustering(
    fasta_input: str,
    output_dir: str,
    min_seq_id: float,
    coverage: float,
    threads: int
) -> str:
    """Ejecuta MMseqs2 easy-cluster y retorna la ruta al TSV de clusters."""
    cluster_prefix = os.path.join(output_dir, "clusterRes")
    tmp_dir = os.path.join(output_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    cmd = [
        "mmseqs", "easy-cluster",
        fasta_input,
        cluster_prefix,
        tmp_dir,
        "--min-seq-id", str(min_seq_id),
        "-c", str(coverage),
        "--cov-mode", "1",
        "--cluster-mode", "0",
        "--threads", str(threads)
    ]

    typer.echo(f"🔬 Ejecutando MMseqs2 clustering...")
    typer.echo(f"   Comando: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        typer.echo(f"❌ Error en MMseqs2:\n{result.stderr}")
        raise typer.Exit(1)

    tsv_path = f"{cluster_prefix}_cluster.tsv"
    if not os.path.exists(tsv_path):
        typer.echo(f"❌ Archivo de clusters no encontrado: {tsv_path}")
        raise typer.Exit(1)

    typer.echo(f"✅ Clustering completado: {tsv_path}")
    return tsv_path


# --- 3. PARSEO DE CLUSTERS ---
def parse_clusters_and_split(
    tsv_path: str,
    fasta_input: str,
    output_dir: str,
    min_cluster_size: int
) -> List[Dict]:
    """
    Lee el TSV de MMseqs2, agrupa secuencias por cluster,
    y escribe un FASTA por cluster.
    Retorna lista de dicts con metadata de cada cluster válido.
    """
    # Leer TSV: col1=representante, col2=miembro
    clusters = defaultdict(list)
    with open(tsv_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                rep, member = parts[0], parts[1]
                clusters[rep].append(member)

    # Leer secuencias del FASTA original
    sequences = {}
    current_header = None
    current_seq = []
    with open(fasta_input, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header is not None:
                    sequences[current_header] = "".join(current_seq)
                current_header = line[1:].split()[0]  # Solo el ID, sin descripción
                current_seq = []
            else:
                current_seq.append(line)
        if current_header is not None:
            sequences[current_header] = "".join(current_seq)

    # Crear directorio de clusters
    clusters_dir = os.path.join(output_dir, "clusters")
    os.makedirs(clusters_dir, exist_ok=True)

    cluster_info = []
    total_clusters = len(clusters)
    skipped = 0

    for idx, (rep, members) in enumerate(clusters.items()):
        size = len(members)

        if size < min_cluster_size:
            skipped += 1
            continue

        cluster_fasta = os.path.join(clusters_dir, f"cluster_{idx}.fasta")

        with open(cluster_fasta, "w") as f:
            for member_id in members:
                if member_id in sequences:
                    f.write(f">{member_id}\n{sequences[member_id]}\n")

        cluster_info.append({
            "idx": idx,
            "rep": rep,
            "size": size,
            "fasta_path": cluster_fasta,
        })

    typer.echo(f"📊 Clusters encontrados: {total_clusters}")
    typer.echo(f"   Clusters válidos (>= {min_cluster_size} seqs): {len(cluster_info)}")
    typer.echo(f"   Clusters omitidos (muy pequeños): {skipped}")

    return cluster_info


# --- 4. WORKER: MAFFT + CONSENSO POR CLUSTER ---
def process_single_cluster(args: Tuple) -> Dict:
    """
    Worker que ejecuta MAFFT → CIAlign (o fallback Python) para un solo cluster.
    Diseñado para ser llamado desde multiprocessing.Pool.
    """
    cluster, consensus_dir, mafft_threads, use_cialign = args
    idx = cluster["idx"]
    fasta_path = cluster["fasta_path"]
    size = cluster["size"]
    rep = cluster["rep"]

    result = {
        "idx": idx,
        "rep": rep,
        "size": size,
        "success": False,
        "consensus_path": None,
        "method": None,
        "error": None
    }

    clusters_dir = os.path.dirname(fasta_path)
    msa_path = os.path.join(clusters_dir, f"cluster_{idx}_msa.fasta")
    consensus_stem = os.path.join(consensus_dir, f"cluster_{idx}")
    consensus_path = f"{consensus_stem}_consensus.fasta"

    try:
        # --- Caso especial: cluster con 1 sola secuencia ---
        if size == 1:
            with open(fasta_path, "r") as fin, open(consensus_path, "w") as fout:
                for line in fin:
                    fout.write(line)
            result["success"] = True
            result["consensus_path"] = consensus_path
            result["method"] = "copy"
            return result

        # --- MAFFT: Alineamiento Múltiple ---
        mafft_cmd = [
            "mafft",
            "--auto",
            "--thread", str(mafft_threads),
            "--quiet",
            fasta_path
        ]
        with open(msa_path, "w") as msa_out:
            mafft_result = subprocess.run(
                mafft_cmd, stdout=msa_out, stderr=subprocess.PIPE, text=True
            )

        if mafft_result.returncode != 0:
            result["error"] = f"MAFFT falló: {mafft_result.stderr[:500]}"
            return result

        if os.path.getsize(msa_path) == 0:
            result["error"] = "MAFFT generó un archivo vacío."
            return result

        # --- GENERAR CONSENSO ---
        cialign_ok = False

        if use_cialign:
            # Intentar con CIAlign primero
            cialign_cmd = [
                "CIAlign",
                "--infile", msa_path,
                "--outfile_stem", consensus_stem,
                "--make_consensus",
                "--consensus_type", "majority_nongap",
            ]
            cialign_result = subprocess.run(
                cialign_cmd, capture_output=True, text=True
            )

            if (cialign_result.returncode == 0
                    and os.path.exists(consensus_path)
                    and os.path.getsize(consensus_path) > 0):
                cialign_ok = True
                result["success"] = True
                result["consensus_path"] = consensus_path
                result["method"] = "CIAlign"

        # Fallback: consenso Python si CIAlign falló o no está disponible
        if not cialign_ok:
            fallback_ok = python_consensus_from_msa(msa_path, consensus_path)
            if fallback_ok:
                result["success"] = True
                result["consensus_path"] = consensus_path
                result["method"] = "python_fallback"
            else:
                result["error"] = "Fallback Python también falló (MSA vacío o sin bases)."

    except Exception as e:
        result["error"] = str(e)

    return result


# --- 5. MERGE FINAL DE CONSENSOS ---
def merge_consensus_library(
    results: List[Dict],
    output_path: str
):
    """Concatena todos los consensos exitosos en una sola librería FASTA."""
    success_count = 0
    fail_count = 0

    with open(output_path, "w") as fout:
        for r in results:
            if r["success"] and r["consensus_path"]:
                with open(r["consensus_path"], "r") as fin:
                    for line in fin:
                        if line.startswith(">"):
                            # Renombrar header: incluir info del cluster
                            rep_label = r["rep"]
                            fout.write(f">{rep_label}_cluster{r['idx']}_n{r['size']}\n")
                        else:
                            fout.write(line)
                success_count += 1
            else:
                fail_count += 1

    return success_count, fail_count


# --- 6. COMANDO CLI PRINCIPAL ---
@app.command()
def build(
    fasta_input: str = typer.Argument(..., help="Archivo FASTA de candidatos (salida de PinaNet)."),
    output_dir: str = typer.Argument(..., help="Directorio de salida para todos los resultados."),
    min_seq_id: float = typer.Option(0.8, help="Identidad mínima de secuencia para clustering MMseqs2."),
    coverage: float = typer.Option(0.8, help="Cobertura mínima para clustering MMseqs2."),
    threads: int = typer.Option(4, help="Threads para MMseqs2 y MAFFT."),
    workers: int = typer.Option(4, help="Procesos paralelos (multiprocessing) para MAFFT + CIAlign."),
    min_cluster_size: int = typer.Option(2, help="Mínimo de secuencias en un cluster para generar MSA.")
):
    """
    Pipeline completo: MMseqs2 clustering → MAFFT MSA → CIAlign consenso.
    Genera una librería FASTA de secuencias consenso a partir de los candidatos de PinaNet.
    """
    begin = time.time()

    # Validaciones
    if not os.path.exists(fasta_input):
        typer.echo(f"❌ Archivo no encontrado: {fasta_input}")
        raise typer.Exit(1)

    os.makedirs(output_dir, exist_ok=True)
    consensus_dir = os.path.join(output_dir, "consensus")
    os.makedirs(consensus_dir, exist_ok=True)

    typer.echo("=" * 60)
    typer.echo("🧬 PinaNet Library Builder")
    typer.echo("=" * 60)
    typer.echo(f"   Input:           {fasta_input}")
    typer.echo(f"   Output:          {output_dir}")
    typer.echo(f"   Min Seq ID:      {min_seq_id}")
    typer.echo(f"   Coverage:        {coverage}")
    typer.echo(f"   Threads:         {threads}")
    typer.echo(f"   Workers:         {workers}")
    typer.echo(f"   Min Cluster Size: {min_cluster_size}")
    typer.echo("=" * 60)

    # Paso 0: Verificar dependencias
    cialign_available = check_dependencies()

    # Paso 1: Clustering
    typer.echo("\n" + "─" * 40)
    typer.echo("PASO 1/4: Clustering con MMseqs2")
    typer.echo("─" * 40)
    tsv_path = run_mmseqs_clustering(fasta_input, output_dir, min_seq_id, coverage, threads)

    # Paso 2: Parsear clusters y generar FASTAs individuales
    typer.echo("\n" + "─" * 40)
    typer.echo("PASO 2/4: Generando FASTAs por cluster")
    typer.echo("─" * 40)
    cluster_info = parse_clusters_and_split(tsv_path, fasta_input, output_dir, min_cluster_size)

    if not cluster_info:
        typer.echo("⚠️ No se encontraron clusters válidos. Revisa los parámetros de clustering.")
        raise typer.Exit(0)

    # Paso 3: MAFFT + CIAlign en paralelo
    typer.echo("\n" + "─" * 40)
    typer.echo(f"PASO 3/4: Alineamiento + Consenso ({workers} workers)")
    typer.echo("─" * 40)

    # Distribuir threads entre workers
    mafft_threads = max(1, threads // workers)
    typer.echo(f"   Threads por worker (MAFFT): {mafft_threads}")

    # Preparar argumentos para el Pool
    pool_args = [
        (cluster, consensus_dir, mafft_threads, cialign_available)
        for cluster in cluster_info
    ]

    with Pool(processes=workers) as pool:
        results = pool.map(process_single_cluster, pool_args)

    # Paso 4: Merge final
    typer.echo("\n" + "─" * 40)
    typer.echo("PASO 4/4: Generando librería final")
    typer.echo("─" * 40)

    library_path = os.path.join(output_dir, "consensus_library.fasta")
    success_count, fail_count = merge_consensus_library(results, library_path)

    # Estadísticas por método
    from collections import Counter
    method_counts = Counter(r.get("method") for r in results if r["success"])

    # Resumen
    elapsed = time.time() - begin
    typer.echo("\n" + "=" * 60)
    typer.echo("📋 RESUMEN")
    typer.echo("=" * 60)
    typer.echo(f"   Clusters procesados:  {success_count}")
    typer.echo(f"   Clusters con error:   {fail_count}")
    if method_counts:
        typer.echo(f"   Método CIAlign:       {method_counts.get('CIAlign', 0)}")
        typer.echo(f"   Método fallback Py:   {method_counts.get('python_fallback', 0)}")
        typer.echo(f"   Copiados (1 seq):     {method_counts.get('copy', 0)}")
    typer.echo(f"   Librería final:       {library_path}")
    typer.echo(f"   Tiempo total:         {elapsed:.1f}s")
    typer.echo("=" * 60)

    if fail_count > 0:
        typer.echo(f"\n⚠️ {fail_count} clusters con errores (primeros 20):")
        error_count = 0
        for r in results:
            if not r["success"]:
                typer.echo(f"   Cluster {r['idx']} ({r['rep']}): {r['error']}")
                error_count += 1
                if error_count >= 20:
                    typer.echo(f"   ... y {fail_count - 20} más.")
                    break

    typer.secho(f"\n✅ ¡Librería generada exitosamente!", fg=typer.colors.GREEN, bold=True)


if __name__ == "__main__":
    app()
