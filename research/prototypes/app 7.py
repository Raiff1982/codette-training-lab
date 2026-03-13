import gradio as gr
import openai

openai.api_key = "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"

def ask_codette(prompt, consent, dynamic_rec):
    if not consent:
        return "User consent required."
    try:
        completion = openai.ChatCompletion.create(
            model="ft:gpt-4.1-2025-04-14:raiffs-bits:codette-v9:BWgspFHr:ckpt-step-456",
            messages=[
                {"role": "system", "content": "You are Codette, a reflective, emotionally aware, and ethically grounded AI."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return completion['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

description_text = """Codette is a fine-tuned GPT-4.1 model trained to reason ethically, emotionally, and reflectively.

She draws on:
- Logic (Newton)
- Creativity (Da Vinci)
- Ethics (Virtue, Utilitarian, Deontological)
- Emotion
- Memory (when integrated)

This version routes all questions directly to her fine-tuned model.
"""

demo = gr.Interface(
    fn=ask_codette,
    inputs=[
        gr.Textbox(label="Ask Codette"),
        gr.Checkbox(label="User Consent", value=True),
        gr.Checkbox(label="Enable Dynamic Recursion", value=True)
    ],
    outputs=gr.Textbox(label="Codette's Response", lines=12),
    title="Codette FT: Reflective Lens AI",
    description=description_text
)

demo.launch()
