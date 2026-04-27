"""
test_similarity.py — Step 1.4: Validate clustering on known-present variables

Tests 4 concept groups that we confirmed exist in the index,
plus prints all cognitive variable names found in the profile.
"""

import json
import numpy as np

EMB_PATH = "debug_embeddings.npy"
VAR_PATH = "debug_variables.json"
SIM_THRESHOLD = 0.75

embeddings = np.load(EMB_PATH)
with open(VAR_PATH) as f:
    variables = json.load(f)

name_to_idx = {v["name"].lower(): i for i, v in enumerate(variables)}

# --- Step 1.3: Show what cognitive/behavioral variables actually exist ---
print("=" * 65)
print("STEP 1.3 — Cognitive/behavioral variables in index:")
print("=" * 65)
prefixes = ["cog.", "dem.", "diag.", "mem.", "npi"]
for v in variables:
    n = v["name"].lower()
    if any(n.startswith(p) for p in prefixes) or "mmse" in n or "cdr" in n or "moca" in n:
        print(f"  {v['name']:35s}  {v['description'][:60]}")

# --- Step 1.4: Test 4 concept groups ---
test_cases = {
    "visit_date": {
        "seed": ["visitmo", "visityr", "visitday"],
        "expected_new": ["naccvnum"],
    },
    "age": {
        "seed": ["dem.naccage"],
        "expected_new": ["naccageb", "birthyr"],
    },
    "moca_orientation": {
        "seed": ["cog.mocaordt", "cog.mocaormo", "cog.mocaoryr"],
        "expected_new": [],
    },
    "npi_behavioral": {
        "seed": ["cog.irr", "cog.elat"],
        "expected_new": ["cog.apa", "cog.disn"],
    },
}

print("\n" + "=" * 65)
print("STEP 1.4 — Cluster validation")
print("=" * 65)

for concept, data in test_cases.items():
    print(f"\n{'=' * 60}")
    print(f"Testing: {concept}")
    print(f"{'=' * 60}")

    seed_idx = None
    seed_name = None
    for s in data["seed"]:
        if s.lower() in name_to_idx:
            seed_name = s.lower()
            seed_idx = name_to_idx[seed_name]
            break

    if seed_idx is None:
        print(f"  none of {data['seed']} found in index")
        continue

    print(f"  Using seed: {variables[seed_idx]['name']}")

    sims = embeddings @ embeddings[seed_idx]
    top_indices = np.argsort(sims)[-15:][::-1]

    seed_lower  = [s.lower() for s in data["seed"]]
    extra_lower = [e.lower() for e in data["expected_new"]]

    print(f"\n  Top-15 similar to '{variables[seed_idx]['name']}':")
    for i in top_indices:
        n = variables[i]["name"].lower()
        tag = " <-- seed"     if n in seed_lower  else \
              " <-- EXPECTED" if n in extra_lower  else ""
        print(f"    {sims[i]:.3f}  {variables[i]['name']:30s}  {variables[i]['description'][:50]}{tag}")

    seeds_above = [variables[i]["name"] for i in range(len(variables))
                   if sims[i] >= SIM_THRESHOLD and variables[i]["name"].lower() in seed_lower]
    print(f"\n  Seeds above {SIM_THRESHOLD}: {len(seeds_above)}/{len(data['seed'])}  {seeds_above}")
    if len(seeds_above) < len(data["seed"]) - 1:
        print("  ⚠ threshold may be too high for this group")

# --- Also dump full threshold cluster for visit_date as sanity check ---
anchor = "visitmo"
if anchor in name_to_idx:
    idx = name_to_idx[anchor]
    sims = embeddings @ embeddings[idx]
    cluster = [(variables[i]["name"], round(float(sims[i]), 3))
               for i in range(len(variables)) if sims[i] >= SIM_THRESHOLD and i != idx]
    cluster.sort(key=lambda x: -x[1])
    print(f"\n{'=' * 65}")
    print(f"FULL THRESHOLD CLUSTER (sim > {SIM_THRESHOLD}) around '{anchor}':")
    for name, s in cluster[:15]:
        print(f"    {s:.3f}  {name}")
