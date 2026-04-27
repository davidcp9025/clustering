"""
final_validation.py — Clustering quality validation for synonym groups.

Inputs:
    synonym_groups.json   — {cluster_label: [var_name, ...]}
    debug_variables.json  — [{name, description}, ...]
"""

import json
from collections import Counter

STOPWORDS = {"the", "and", "with", "that", "this", "from", "for", "are", "was",
             "each", "were", "have", "been", "which", "not", "but", "its"}

# ── Load data ────────────────────────────────────────────────────────────────
with open("synonym_groups.json") as f:
    clusters = json.load(f)

with open("debug_variables.json") as f:
    raw_vars = json.load(f)

all_vars = {v["name"]: v for v in raw_vars}
total_vars = len(raw_vars)

# Pre-compute per-cluster metadata
cluster_meta = {}
for label, members in clusters.items():
    prefixes = Counter(v.split(".")[0] for v in members if "." in v)
    no_prefix = sum(1 for v in members if "." not in v)
    if no_prefix:
        prefixes["(none)"] += no_prefix

    desc_words = []
    for vname in members:
        desc = all_vars.get(vname, {}).get("description", "")
        words = [w.lower().strip(".,;:()[]") for w in desc.split()
                 if len(w) > 4 and w.lower() not in STOPWORDS and w.isalpha()]
        desc_words.extend(words)

    cluster_meta[label] = {
        "size": len(members),
        "members": members,
        "prefixes": prefixes,
        "top_keywords": Counter(desc_words).most_common(5),
    }

sizes = sorted([m["size"] for m in cluster_meta.values()], reverse=True)
clustered_vars = {v for members in clusters.values() for v in members}

# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 1 — OVERALL STATISTICS")
print("=" * 70)
print(f"  Total variables in index : {total_vars}")
print(f"  Variables in a cluster   : {len(clustered_vars)}  "
      f"({100 * len(clustered_vars) / total_vars:.1f}% coverage)")
print(f"  Variables NOT clustered  : {total_vars - len(clustered_vars)}")
print(f"  Total clusters           : {len(clusters)}")
print(f"  Average cluster size     : {sum(sizes) / len(sizes):.1f}")
print(f"  Largest cluster          : {sizes[0]}")
print(f"  Median cluster size      : {sizes[len(sizes) // 2]}")
print(f"  Smallest cluster         : {sizes[-1]}")

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2 — TOP 3 LARGEST CLUSTERS")
print("=" * 70)

top3 = sorted(cluster_meta.items(), key=lambda x: x[1]["size"], reverse=True)[:3]

for label, meta in top3:
    print(f"\n  Cluster: '{label}'  ({meta['size']} vars)")
    print(f"  Prefix distribution: {dict(meta['prefixes'].most_common())}")
    print(f"  Top keywords: {meta['top_keywords']}")
    print(f"  Sample variables:")
    for vname in meta["members"][:8]:
        desc = all_vars.get(vname, {}).get("description", "???")[:65]
        print(f"    {vname:35} | {desc}")

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3 — QUALITY CHECKS")
print("=" * 70)

suspicious = [(label, meta) for label, meta in cluster_meta.items()
              if len(meta["prefixes"]) > 3]

if suspicious:
    print(f"\n  ⚠️  Suspicious clusters (>3 distinct prefixes): {len(suspicious)}")
    for label, meta in suspicious:
        print(f"    '{label}'  size={meta['size']}  prefixes={set(meta['prefixes'].keys())}")
else:
    print(f"\n  ✅ No suspicious clusters (all have ≤3 distinct prefixes)")

print(f"\n  Clustering coverage:")
print(f"    {len(clustered_vars)}/{total_vars} variables are in at least one cluster  "
      f"({100 * len(clustered_vars) / total_vars:.1f}%)")
unclustered = [v["name"] for v in raw_vars if v["name"] not in clustered_vars]
print(f"    {len(unclustered)} unclustered variables (singletons)")
if unclustered[:5]:
    print(f"    Examples: {unclustered[:5]}")

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4 — AUTOMATED VERDICT")
print("=" * 70)

max_size = sizes[0]
n_suspicious = len(suspicious)

if max_size <= 60 and n_suspicious <= 2:
    verdict = "PASS"
    icon = "✅"
    action = ("Clusters are production-ready.\n"
              "  Next step → Phase 3: integrate synonym_groups.json into\n"
              "  shared_resources.py to replace seed_synonyms.")
elif max_size <= 100:
    verdict = "BORDERLINE"
    icon = "⚠️ "
    action = (f"Largest cluster has {max_size} vars — inspect it with inspect_megaclusters.py.\n"
              "  Consider raising template_threshold for its prefix group,\n"
              "  then re-run cluster_builder.py.")
else:
    verdict = "FAIL"
    icon = "❌"
    action = (f"Mega-cluster of {max_size} vars detected — clustering is too loose.\n"
              "  Run: python cluster_builder.py ... 0.75 0.90\n"
              "  or add the offending prefix to TEMPLATE_PREFIXES in cluster_builder.py.")

print(f"\n  {icon} Verdict: {verdict}")
print(f"  Max cluster size  : {max_size}  (limit: 60)")
print(f"  Suspicious clusters: {n_suspicious}  (limit: 2)")
print(f"\n  Recommended action:\n  {action}")
print()
