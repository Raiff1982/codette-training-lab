
import numpy as np
import random

def evaluate_stability(seq):
    hydrophobic_aas = set("AILMFWYV")
    hydrophobic_ratio = sum(1 for aa in seq if aa in hydrophobic_aas) / len(seq)
    aromaticity_score = seq.count('F') + seq.count('W') + seq.count('Y')
    return round((hydrophobic_ratio * 0.6 + aromaticity_score * 0.1), 4)

def evaluate_affinity(seq):
    entropy = len(set(seq)) / len(seq)
    return round((1 - entropy) * 0.8 + random.uniform(0.1, 0.3), 4)

def run_simulations(binder_candidates, engines=['SimLite']):
    scored_binders = []
    rejections = []

    for binder in binder_candidates.get("generated_binders", []):
        sequence = binder["sequence"]
        stability = evaluate_stability(sequence)
        affinity = evaluate_affinity(sequence)
        reasons = []

        if stability < 0.3:
            reasons.append("Low stability score")
        if affinity < 0.3:
            reasons.append("Low predicted affinity")

        if reasons:
            binder["rejection_reason"] = reasons
            rejections.append(binder)
        else:
            binder["stability_score"] = stability
            binder["predicted_affinity"] = affinity
            binder["structure_engine"] = engines[0]
            binder["simulation_trace"] = f"Hydrophobic: {round(stability, 3)}, Entropy-Based Affinity: {round(affinity, 3)}"
            scored_binders.append(binder)

    return {
        "validated_binders": scored_binders,
        "rejected_binders": rejections
    }
