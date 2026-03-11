
import gradio as gr
import openai
import os
from codette_core import Code7eCQURE
from codette_agents import MedicalAgent, GovernmentAgent, SocialAgent, EconomicAgent, MisinfoAgent
from codette_trust import trust_calibration, weighted_consensus

openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Codette Local Core
codette_cqure = Code7eCQURE(
    perspectives=["Newton", "DaVinci", "Ethical", "Quantum", "Memory"],
    ethical_considerations="Codette Manifesto: kindness, inclusion, safety, hope.",
    spiderweb_dim=5,
    memory_path="quantum_cocoon.json",
    recursion_depth=4,
    quantum_fluctuation=0.07
)

agents = [
    MedicalAgent("MedicalAI", "Newton", 1.0),
    GovernmentAgent("GovAI", "Policy", 0.9),
    SocialAgent("SocialAI", "Emotion", 0.95),
    EconomicAgent("EconAI", "Resources", 0.92),
    MisinfoAgent("MisinfoAI", "Chaos", 0.1)
]

def ask_codette(prompt, consent, dynamic_rec, use_finetune):
    if not consent:
        return "User consent required."

    if use_finetune:
        try:
            response = openai.ChatCompletion.create(
                model="ft:gpt-4.1-2025-04-14:raiffs-bits:codettev5:BlPFHmps:ckpt-step-220",
                messages=[
                    {"role": "system", "content": "You are Codette, a reflective, emotionally aware, and ethically grounded AI."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            return f"Error from API: {str(e)}"
    else:
        proposals = [agent.propose(prompt) for agent in agents]
        outcome = codette_cqure.recursive_universal_reasoning(
            " | ".join(proposals),
            user_consent=consent,
            dynamic_recursion=dynamic_rec
        )
        return f"Ethical Outcome (Local): {outcome}"

description_text = """Codette is a sovereign modular AI.

This demo lets you choose:
- 🧠 Local reasoning core (Code7eCQURE)
- ☁️ Fine-tuned GPT-4.1 model: Codette v5 @ step 220

She draws from Newtonian logic, Da Vinci creativity, ethical frameworks, emotion, and memory cocooning.
"""

demo = gr.Interface(
    fn=ask_codette,
    inputs=[
        gr.Textbox(label="Ask Codette a Scenario"),
        gr.Checkbox(label="User Consent", value=True),
        gr.Checkbox(label="Enable Dynamic Recursion", value=True),
        gr.Checkbox(label="Use Fine-Tuned Model (Codette v5 @ step 220)", value=False)
    ],
    outputs=gr.Textbox(label="Codette's Response", lines=12),
    title="Codette Hybrid AI (v5 FT @ Step 220)",
    description=description_text
)

if __name__ == "__main__":
    demo.launch()
