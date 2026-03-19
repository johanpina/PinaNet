import typer
import subprocess
import shutil
import os
import time
from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, List, Tuple

app = typer.Typer(
    name="TEGER Library Builder",
    help="Pipeline to generate consensus libraries: MMseqs2 → MAFFT → CIAlign."
)

# --- 1. DEPENDENCY CHECK ---
def check_dependencies() -> bool:
    """
    Verifies that mmseqs and mafft are installed (required).
    CIAlign is optional: if not present, a Python fallback is used.
    Returns True if CIAlign is available, False otherwise.
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
        typer.echo("❌ Required dependencies not found in PATH:")
        for m in missing:
            typer.echo(m)
        typer.echo("\nInstall the missing dependencies and re-run.")
        raise typer.Exit(1)

    cialign_available = shutil.which("CIAlign") is not None
    if cialign_available:
        typer.echo("✅ All dependencies found (mmseqs, mafft, CIAlign).")
    else:
        typer.echo("✅ mmseqs and mafft found.")
        typer.echo("   ℹ️ CIAlign not found. Python consensus generator (fallback) will be used.")

    return cialign_available


# --- FALLBACK: PURE PYTHON CONSENSUS ---
def python_consensus_from_msa(msa_path: str, consensus_path: str) -> bool:
    """
    Generates a consensus sequence by majority vote from an MSA FASTA.
    Requires no external dependencies. Returns True on success.
    """
    from collections import Counter

    # Read aligned sequences
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

    # Majority vote per column (ignoring gaps)
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


# --- 2. MMSEQS2 CLUSTERING ---
def run_mmseqs_clustering(
    fasta_input: str,
    output_dir: str,
    min_seq_id: float,
    coverage: float,
    threads: int
) -> str:
    """Runs MMseqs2 easy-cluster and returns the path to the cluster TSV file."""
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

    typer.echo(f"🔬 Running MMseqs2 clustering...")
    typer.echo(f"   Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        typer.echo(f"❌ MMseqs2 error:\n{result.stderr}")
        raise typer.Exit(1)

    tsv_path = f"{cluster_prefix}_cluster.tsv"
    if not os.path.exists(tsv_path):
        typer.echo(f"❌ Cluster file not found: {tsv_path}")
        raise typer.Exit(1)

    typer.echo(f"✅ Clustering complete: {tsv_path}")
    return tsv_path


# --- 3. CLUSTER PARSING ---
def parse_clusters_and_split(
    tsv_path: str,
    fasta_input: str,
    output_dir: str,
    min_cluster_size: int
) -> List[Dict]:
    """
    Reads the MMseqs2 TSV, groups sequences by cluster,
    and writes one FASTA per cluster.
    Returns a list of dicts with metadata for each valid cluster.
    """
    # Read TSV: col1=representative, col2=member
    clusters = defaultdict(list)
    with open(tsv_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                rep, member = parts[0], parts[1]
                clusters[rep].append(member)

    # Read sequences from original FASTA
    sequences = {}
    current_header = None
    current_seq = []
    with open(fasta_input, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header is not None:
                    sequences[current_header] = "".join(current_seq)
                current_header = line[1:].split()[0]  # ID only, no description
                current_seq = []
            else:
                current_seq.append(line)
        if current_header is not None:
            sequences[current_header] = "".join(current_seq)

    # Create clusters directory
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

    typer.echo(f"📊 Clusters found: {total_clusters}")
    typer.echo(f"   Valid clusters (>= {min_cluster_size} seqs): {len(cluster_info)}")
    typer.echo(f"   Skipped clusters (too small): {skipped}")

    return cluster_info


# --- 4. WORKER: MAFFT + CONSENSUS PER CLUSTER ---
def process_single_cluster(args: Tuple) -> Dict:
    """
    Worker that runs MAFFT → CIAlign (or Python fallback) for a single cluster.
    Designed to be called from multiprocessing.Pool.
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
        # --- Special case: cluster with a single sequence ---
        if size == 1:
            with open(fasta_path, "r") as fin, open(consensus_path, "w") as fout:
                for line in fin:
                    fout.write(line)
            result["success"] = True
            result["consensus_path"] = consensus_path
            result["method"] = "copy"
            return result

        # --- MAFFT: Multiple Sequence Alignment ---
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
            result["error"] = f"MAFFT failed: {mafft_result.stderr[:500]}"
            return result

        if os.path.getsize(msa_path) == 0:
            result["error"] = "MAFFT produced an empty file."
            return result

        # --- GENERATE CONSENSUS ---
        cialign_ok = False

        if use_cialign:
            # Try CIAlign first
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

        # Fallback: Python consensus if CIAlign failed or is not available
        if not cialign_ok:
            fallback_ok = python_consensus_from_msa(msa_path, consensus_path)
            if fallback_ok:
                result["success"] = True
                result["consensus_path"] = consensus_path
                result["method"] = "python_fallback"
            else:
                result["error"] = "Python fallback also failed (empty MSA or no valid bases)."

    except Exception as e:
        result["error"] = str(e)

    return result


# --- 5. FINAL CONSENSUS MERGE ---
def merge_consensus_library(
    results: List[Dict],
    output_path: str
):
    """Concatenates all successful consensus sequences into a single FASTA library."""
    success_count = 0
    fail_count = 0

    with open(output_path, "w") as fout:
        for r in results:
            if r["success"] and r["consensus_path"]:
                with open(r["consensus_path"], "r") as fin:
                    for line in fin:
                        if line.startswith(">"):
                            # Rename header: include cluster info
                            rep_label = r["rep"]
                            fout.write(f">{rep_label}_cluster{r['idx']}_n{r['size']}\n")
                        else:
                            fout.write(line)
                success_count += 1
            else:
                fail_count += 1

    return success_count, fail_count


# --- 6. MAIN CLI COMMAND ---
@app.command()
def build(
    fasta_input: str = typer.Argument(..., help="Input FASTA file of candidates (output from TEGER)."),
    output_dir: str = typer.Argument(..., help="Output directory for all results."),
    min_seq_id: float = typer.Option(0.8, help="Minimum sequence identity for MMseqs2 clustering (0-1)."),
    coverage: float = typer.Option(0.8, help="Minimum alignment coverage for MMseqs2 clustering (0-1)."),
    threads: int = typer.Option(4, help="CPU threads for MMseqs2 and MAFFT."),
    workers: int = typer.Option(4, help="Parallel processes (multiprocessing) for MAFFT + CIAlign."),
    min_cluster_size: int = typer.Option(2, help="Minimum sequences in a cluster to generate MSA.")
):
    """
    Full pipeline: MMseqs2 clustering → MAFFT MSA → CIAlign consensus.
    Generates a FASTA consensus library from TEGER candidate sequences.
    """
    begin = time.time()

    # Validate input
    if not os.path.exists(fasta_input):
        typer.echo(f"❌ File not found: {fasta_input}")
        raise typer.Exit(1)

    os.makedirs(output_dir, exist_ok=True)
    consensus_dir = os.path.join(output_dir, "consensus")
    os.makedirs(consensus_dir, exist_ok=True)

    typer.echo("=" * 60)
    typer.echo("🧬 TEGER Library Builder")
    typer.echo("=" * 60)
    typer.echo(f"   Input:            {fasta_input}")
    typer.echo(f"   Output:           {output_dir}")
    typer.echo(f"   Min Seq ID:       {min_seq_id}")
    typer.echo(f"   Coverage:         {coverage}")
    typer.echo(f"   Threads:          {threads}")
    typer.echo(f"   Workers:          {workers}")
    typer.echo(f"   Min Cluster Size: {min_cluster_size}")
    typer.echo("=" * 60)

    # Step 0: Check dependencies
    cialign_available = check_dependencies()

    # Step 1: Clustering
    typer.echo("\n" + "─" * 40)
    typer.echo("STEP 1/4: MMseqs2 Clustering")
    typer.echo("─" * 40)
    tsv_path = run_mmseqs_clustering(fasta_input, output_dir, min_seq_id, coverage, threads)

    # Step 2: Parse clusters and generate individual FASTAs
    typer.echo("\n" + "─" * 40)
    typer.echo("STEP 2/4: Generating per-cluster FASTAs")
    typer.echo("─" * 40)
    cluster_info = parse_clusters_and_split(tsv_path, fasta_input, output_dir, min_cluster_size)

    if not cluster_info:
        typer.echo("⚠️ No valid clusters found. Check clustering parameters.")
        raise typer.Exit(0)

    # Step 3: MAFFT + CIAlign in parallel
    typer.echo("\n" + "─" * 40)
    typer.echo(f"STEP 3/4: Alignment + Consensus ({workers} workers)")
    typer.echo("─" * 40)

    # Distribute threads across workers
    mafft_threads = max(1, threads // workers)
    typer.echo(f"   Threads per worker (MAFFT): {mafft_threads}")

    # Prepare arguments for the Pool
    pool_args = [
        (cluster, consensus_dir, mafft_threads, cialign_available)
        for cluster in cluster_info
    ]

    with Pool(processes=workers) as pool:
        results = pool.map(process_single_cluster, pool_args)

    # Step 4: Final merge
    typer.echo("\n" + "─" * 40)
    typer.echo("STEP 4/4: Generating final library")
    typer.echo("─" * 40)

    library_path = os.path.join(output_dir, "consensus_library.fasta")
    success_count, fail_count = merge_consensus_library(results, library_path)

    # Stats by method
    from collections import Counter
    method_counts = Counter(r.get("method") for r in results if r["success"])

    # Summary
    elapsed = time.time() - begin
    typer.echo("\n" + "=" * 60)
    typer.echo("📋 SUMMARY")
    typer.echo("=" * 60)
    typer.echo(f"   Clusters processed:   {success_count}")
    typer.echo(f"   Clusters with errors: {fail_count}")
    if method_counts:
        typer.echo(f"   CIAlign method:       {method_counts.get('CIAlign', 0)}")
        typer.echo(f"   Python fallback:      {method_counts.get('python_fallback', 0)}")
        typer.echo(f"   Copied (1 seq):       {method_counts.get('copy', 0)}")
    typer.echo(f"   Final library:        {library_path}")
    typer.echo(f"   Total time:           {elapsed:.1f}s")
    typer.echo("=" * 60)

    if fail_count > 0:
        typer.echo(f"\n⚠️ {fail_count} clusters with errors (first 20):")
        error_count = 0
        for r in results:
            if not r["success"]:
                typer.echo(f"   Cluster {r['idx']} ({r['rep']}): {r['error']}")
                error_count += 1
                if error_count >= 20:
                    typer.echo(f"   ... and {fail_count - 20} more.")
                    break

    typer.secho(f"\n✅ Library generated successfully!", fg=typer.colors.GREEN, bold=True)


if __name__ == "__main__":
    app()
