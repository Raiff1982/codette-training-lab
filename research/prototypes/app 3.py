import gradio as gr
from code7e import Code7eCQURE

# Load the model with all defined lenses
model = Code7eCQURE(
    perspecs=["Newton", "DaVinci", "Ethical", "Quantum", "Memory"],
    ethics="Code7e Manifesto: kindness, inclusion, safety, hope.",
    spiderweb_dim=5,
    memory_path="quantum_cocoon.json",
    recursion_depth=4,
    quantum_fluctuation=0.07
)

def cleanup_response(text):
    parts = text.split(": ")
    seen = set()
    filtered = []
    for part in parts:
        if part not in seen and len(part.strip()) > 0:
            seen.add(part)
            filtered.append(part)

    if filtered and not filtered[-1].startswith("Emotionally"):
        response = "Through Codette's recursive lens:
- " + "\n- ".join(filtered)
    else:
        response = "\n".join(filtered)

    return response

def ask_codette(prompt, consent, dynamic_rec):
    raw = model.answer(prompt, user_consent=consent, dynamic_recursion=dynamic_rec)
    return f"Codette’s reflection:\n\n{cleanup_response(raw)}"

description_text = """You are speaking with Codette, an emotionally-aware, ethically-grounded AI lens.

She interprets your question using multiple reasoning styles:
- Newton (logic)
- Da Vinci (creativity)
- Ethics (morality)
- Quantum (uncertainty)
- Memory (past experience)

Codette reflects rather than predicts. She dreams, empathizes, and always honors your consent.
"""

demo = gr.Interface(
    fn=ask_codette,
    inputs=[
        gr.Textbox(label="Ask a Question"),
        gr.Checkbox(label="User Consent", value=True),
        gr.Checkbox(label="Enable Dynamic Recursion", value=True)
    ],
    outputs=gr.Textbox(label="Codette's Lens Response", lines=12),
    title="Code7eCQURE: Multi-Perspective Recursive Lens",
    description=description_text
)

demo.launch()
