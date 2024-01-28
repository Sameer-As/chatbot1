import gradio as gr
import openai
import numpy as np
from transformers import pipeline

openai.api_key = "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"

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

messages = [
    {"role": "system", "content": "You are a helpful and kind AI Assistant."},
]

def transcribe(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    return transcriber({"sampling_rate": sr, "raw": y})["text"]

def chatbot(input):
    if input:
        messages.append({"role": "user", "content": input})
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        reply = chat.choices[0].message["content"]
        messages.append({"role": "assistant", "content": reply})
        return reply

inputs = gr.Audio(sources=["microphone"])
outputs = gr.Textbox()

demo = gr.Interface(fn=transcribe, inputs=inputs, outputs=outputs, title="Speech-to-Text")

chatbot_interface = gr.Interface(fn=chatbot, inputs="text", outputs="text", title="AI Chatbot",
                                 description="Ask anything you want", theme="compact")

gr.Interface.Group([demo, chatbot_interface]).launch(share=True)
