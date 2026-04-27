import numpy as np
import json
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Set


# Prefixes whose variables share template-like descriptions — require stricter threshold
TEMPLATE_PREFIXES = ("FAM.", "HLT.", "FNC.", "MHLT.", "CLIN.", "CSF.")


def build_synonym_clusters(
    variables: List[Dict],
    embeddings: np.ndarray,
    threshold: float = 0.75,
    template_threshold: float = 0.85,
    min_cluster_size: int = 2,
    output_path: str = None
) -> Dict[str, List[str]]:
    """
    Build synonym clusters with special handling for template variables.

    Template variables (FAM.*, HLT.*, FNC.*, MHLT.*) use a higher threshold
    because their descriptions share structure but represent different concepts.

    Args:
        variables: List of dicts with 'name' and 'description' keys
        embeddings: Precomputed embedding matrix (n_vars x embed_dim), normalized
        threshold: Cosine similarity cutoff for regular variables
        template_threshold: Stricter cutoff for template-prefix variable pairs
        min_cluster_size: Minimum variables per cluster
        output_path: Where to save the cluster JSON

    Returns:
        dict: {cluster_label: [var_name1, var_name2, ...]}
    """
    n = len(variables)

    is_template = np.array([
        any(v["name"].startswith(p) for p in TEMPLATE_PREFIXES)
        for v in variables
    ])
    n_template = int(is_template.sum())
    print(f"Building clusters for {n} variables  "
          f"({n - n_template} regular @ {threshold}, "
          f"{n_template} template @ {template_threshold})")

    # Embeddings already normalized → dot product = cosine similarity
    sim_matrix = embeddings @ embeddings.T

    # Build adjacency list with per-pair threshold
    adj: Dict[int, Set[int]] = defaultdict(set)
    edge_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            thresh = template_threshold if (is_template[i] and is_template[j]) else threshold
            if sim_matrix[i, j] >= thresh:
                adj[i].add(j)
                adj[j].add(i)
                edge_count += 1

    print(f"  Found {edge_count} similarity edges")

    # Add mega-cluster warning before BFS
    visited: Set[int] = set()
    clusters = []

    for start in range(n):
        if start in visited:
            continue
        component: Set[int] = set()
        queue = [start]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            queue.extend(adj[node] - visited)
        if len(component) >= min_cluster_size:
            clusters.append(component)

    print(f"  Found {len(clusters)} clusters (size >= {min_cluster_size})")

    # Label each cluster
    labeled_clusters: Dict[str, List[str]] = {}
    cluster_stats = []

    for cluster_indices in sorted(clusters, key=len, reverse=True):
        cluster_vars = [variables[idx] for idx in cluster_indices]
        var_names = [v["name"] for v in cluster_vars]

        label = _generate_cluster_label(cluster_vars)

        # Handle label collisions by appending a suffix
        base_label = label
        suffix = 2
        while label in labeled_clusters:
            label = f"{base_label}_{suffix}"
            suffix += 1

        labeled_clusters[label] = sorted(var_names)

        cluster_stats.append({
            "label": label,
            "size": len(var_names),
            "vars": var_names[:5],
        })

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"CLUSTER SUMMARY (top 20 by size):")
    print(f"{'=' * 70}")
    for stat in cluster_stats[:20]:
        preview = ", ".join(stat["vars"][:3])
        if stat["size"] > 3:
            preview += f", ... ({stat['size'] - 3} more)"
        print(f"  [{stat['size']:3d} vars] {stat['label']:25s} | {preview}")

    # Warn about mega-clusters
    mega = [s for s in cluster_stats if s["size"] > 50]
    if mega:
        print(f"\n  WARNING: {len(mega)} mega-cluster(s) detected (>50 vars) — consider raising template_threshold")
        for mc in mega:
            print(f"    [{mc['size']} vars] {mc['label']}")

    # Save to disk
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w") as f:
            json.dump(labeled_clusters, f, indent=2)
        print(f"\n✓ Clusters saved to: {out}")

        stats_path = out.parent / f"{out.stem}_stats.json"
        with open(stats_path, "w") as f:
            json.dump(cluster_stats, f, indent=2)
        print(f"✓ Stats saved to:    {stats_path}")

    return labeled_clusters


def _generate_cluster_label(cluster_vars: List[Dict]) -> str:
    """
    Generate a meaningful label for a cluster from variable name tokens.
    """
    GENERIC = {"dem", "cog", "med", "fam", "sub", "nacc", "uds", "lbd",
               "codem", "clin", "csf", "gen", "imag", "neurobat"}
    BLAND   = {"visit", "date", "score", "total", "test", "number", "time",
               "month", "year", "day"}

    all_tokens = []
    for v in cluster_vars:
        name = v["name"].lower().replace(".", " ").replace("_", " ")
        tokens = [t for t in name.split()
                  if len(t) > 2 and not t.isdigit() and t not in GENERIC]
        all_tokens.extend(tokens)

    if not all_tokens:
        return cluster_vars[0]["name"].lower().replace(".", "_")[:25]

    token_counts = Counter(all_tokens)
    label = token_counts.most_common(1)[0][0]

    # If label is bland, try a bigram
    if label in BLAND:
        bigrams = []
        for v in cluster_vars:
            tokens = v["name"].lower().replace(".", " ").replace("_", " ").split()
            for i in range(len(tokens) - 1):
                bigrams.append(f"{tokens[i]}_{tokens[i + 1]}")
        if bigrams:
            label = Counter(bigrams).most_common(1)[0][0]

    return label


if __name__ == "__main__":
    import sys

    emb_path = sys.argv[1] if len(sys.argv) > 1 else "debug_embeddings.npy"
    var_path = sys.argv[2] if len(sys.argv) > 2 else "debug_variables.json"
    out_path = sys.argv[3] if len(sys.argv) > 3 else "synonym_groups.json"
    threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.75
    template_threshold = float(sys.argv[5]) if len(sys.argv) > 5 else 0.85

    embeddings = np.load(emb_path)
    with open(var_path) as f:
        variables = json.load(f)

    build_synonym_clusters(
        variables=variables,
        embeddings=embeddings,
        threshold=threshold,
        template_threshold=template_threshold,
        min_cluster_size=2,
        output_path=out_path,
    )
