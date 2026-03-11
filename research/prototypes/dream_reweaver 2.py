
import os
import json
import random
from typing import List, Dict
from cognition_cocooner import CognitionCocooner

class DreamReweaver:
    """
    Reweaves cocooned thoughts into dream-like synthetic narratives or planning prompts.
    """
    def __init__(self, cocoon_dir: str = "cocoons"):
        self.cocooner = CognitionCocooner(storage_path=cocoon_dir)
        self.dream_log = []

    def generate_dream_sequence(self, limit: int = 5) -> List[str]:
        dream_sequence = []
        cocoons = self._load_cocoons()
        selected = random.sample(cocoons, min(limit, len(cocoons)))

        for cocoon in selected:
            wrapped = cocoon.get("wrapped")
            sequence = self._interpret_cocoon(wrapped, cocoon.get("type"))
            self.dream_log.append(sequence)
            dream_sequence.append(sequence)

        return dream_sequence

    def _interpret_cocoon(self, wrapped: str, type_: str) -> str:
        if type_ == "prompt":
            return f"[DreamPrompt] {wrapped}"
        elif type_ == "function":
            return f"[DreamFunction] {wrapped}"
        elif type_ == "symbolic":
            return f"[DreamSymbol] {wrapped}"
        elif type_ == "encrypted":
            return "[Encrypted Thought Cocoon - Decryption Required]"
        else:
            return "[Unknown Dream Form]"

    def _load_cocoons(self) -> List[Dict]:
        cocoons = []
        for file in os.listdir(self.cocooner.storage_path):
            if file.endswith(".json"):
                path = os.path.join(self.cocooner.storage_path, file)
                with open(path, "r") as f:
                    cocoons.append(json.load(f))
        return cocoons

if __name__ == "__main__":
    dr = DreamReweaver()
    dreams = dr.generate_dream_sequence()
    print("\n".join(dreams))
