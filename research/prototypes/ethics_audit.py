import gradio as gr

log = []

def log_event(msg):
    log.append(msg)

def render():
    with gr.Column():
        gr.Markdown("### Ethics Audit Log")
        logbox = gr.Textbox(value="\n".join(log or ["No activity yet."]), lines=10, interactive=False)
        gr.Button("Refresh").click(lambda: "\n".join(log or ["No activity yet."]), outputs=logbox)