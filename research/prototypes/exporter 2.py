
import json
import os
from datetime import datetime

def export_designs(personalized_binders, format='json', output_dir='output'):
    if format != 'json':
        raise ValueError("Only JSON format is currently supported.")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = os.path.join(output_dir, f"codette_antibody_designs_{timestamp}.json")
    txt_file = os.path.join(output_dir, f"codette_antibody_summary_{timestamp}.txt")

    with open(json_file, 'w') as f:
        json.dump(personalized_binders, f, indent=4)

    with open(txt_file, 'w') as txt:
        txt.write("Codette Antibody Design Summary\n")
        txt.write("="*40 + "\n")
        for b in personalized_binders.get("personalized_binders", []):
            txt.write(f"Sequence: {b['sequence']}\n")
            txt.write(f"Score: {b['personalization_score']}\n")
            txt.write(f"Ancestry: {', '.join(b['ancestry_tags'])}\n")
            txt.write(f"HLA Matches: {b['hla_matches']}\n")
            txt.write(f"Exposure Weight: {b['exposure_weight']}\n")
            txt.write(f"Ethical Notice: {b['ethics_notice']}\n")
            txt.write("-"*40 + "\n")

    return {
        "status": "success",
        "output_file": json_file,
        "summary_file": txt_file,
        "binder_count": len(personalized_binders.get("personalized_binders", []))
    }
