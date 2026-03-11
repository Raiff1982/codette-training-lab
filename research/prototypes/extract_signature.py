
import re
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def extract_signature(seq_input):
    """
    Extracts and analyzes a protein sequence using real bio-physical computations.
    Returns a dict with molecular properties.
    """
    # Clean sequence
    seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seq_input.upper())
    if len(seq) < 30:
        raise ValueError("Sequence too short for reliable analysis.")

    # Perform analysis
    analysis = ProteinAnalysis(seq)
    return {
        "cleaned_sequence": seq,
        "length": len(seq),
        "molecular_weight": analysis.molecular_weight(),
        "aromaticity": analysis.aromaticity(),
        "instability_index": analysis.instability_index(),
        "isoelectric_point": analysis.isoelectric_point(),
        "gravy": analysis.gravy()
    }
