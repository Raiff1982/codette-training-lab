import json, os, hashlib
from collections import Counter, defaultdict
from random import random, choice
import openai

openai.api_key = "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"

class Code7eCQURE:
    # [Content omitted here for brevity—it matches previous update with FT model support]
    def answer(self, question, user_consent=True, dynamic_recursion=True, use_ft_model=False):
        if use_ft_model:
            try:
                completion = openai.ChatCompletion.create(
                    model="ft:gpt-4.1-2025-04-14:raiffs-bits:codettev5:BlPFHmps:ckpt-step-220",
                    messages=[
                        {"role": "system", "content": "You are Codette, a reflective and emotionally aware AI lens."},
                        {"role": "user", "content": question}
                    ],
                    temperature=0.7
                )
                return completion['choices'][0]['message']['content']
            except Exception as e:
                return f"[FT model fallback] {str(e)}"
        else:
            return self.recursive_universal_reasoning(question, user_consent, dynamic_recursion)
