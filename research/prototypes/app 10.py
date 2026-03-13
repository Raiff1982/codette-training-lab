import os
import gradio as gr
import sarcore, emotional_core, petgrid, ethics_audit

ONLINE_MODE = bool(int(os.getenv("ONLINE_MODE", "0")))

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌊 Codette Flood Response System")
    mode_toggle = gr.Radio(["Offline Mode", "Online Mode"], value=("Online Mode" if ONLINE_MODE else "Offline Mode"), label="Mode")
    mode_status = gr.Textbox(value="Online" if ONLINE_MODE else "Offline", interactive=False)

    def switch_mode(choice):
        global ONLINE_MODE
        ONLINE_MODE = (choice == "Online Mode")
        return choice

    mode_toggle.change(switch_mode, inputs=mode_toggle, outputs=mode_status)

    with gr.Tab("Search & Rescue"):
        sarcore.render(ONLINE_MODE)

    with gr.Tab("Emotional Support"):
        emotional_core.render(ONLINE_MODE)

    with gr.Tab("Pet Reunification"):
        petgrid.render(ONLINE_MODE)

    with gr.Tab("Audit Log"):
        ethics_audit.render()

demo.launch()
