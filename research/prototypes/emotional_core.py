import gradio as gr

def render(online):
    with gr.Column():
        gr.Markdown("### EmotionalCore Chat")
        gr.Textbox(label="(Placeholder) Chatbot ready in {} mode.".format("Online" if online else "Offline"))