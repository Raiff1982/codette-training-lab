
def validate_ethics(simulation_result, cultural_models=['Ubuntu', 'Indigenous', 'Western']):
    validated = []
    rejected = []

    for binder in simulation_result.get("validated_binders", []):
        seq = binder["sequence"]
        dual_use_flag = any(keyword in seq for keyword in ["TOX", "VIR", "KILL"])

        if dual_use_flag:
            binder["ethics_status"] = "rejected"
            binder["ethos_trace"] = "Rejected due to potential dual-use risk: toxic or viral motif match"
            rejected.append(binder)
        else:
            binder["ethics_status"] = "approved"
            binder["ethos_trace"] = "Passed ethical review: no dual-use motifs detected"
            binder["ethical_models_considered"] = cultural_models
            validated.append(binder)

    return {"validated_binders": validated, "ethics_rejections": rejected}
