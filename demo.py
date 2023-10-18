import os
import gradio as gr
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM

import random
import logging
import json
import sys
import warnings
import torch
from torch import cuda
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

tokenizer_7b = AutoTokenizer.from_pretrained("./Taiwan-LLaMa-v1_0")
model_7b = LlamaForCausalLM.from_pretrained("./Taiwan-LLaMa-v1_0", device_map="auto")


def get_reply_from_llm(text, max_len, model_size):
    prompt = f"This is a chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {text} ASSISTANT:"    
    
    if model_size=='Taiwan-LLaMa-v1_0':        
        batch = tokenizer_7b(prompt + '\n', return_tensors="pt", add_special_tokens=False)
        
        batch = {k: v.to('cuda') for k, v in batch.items()}
        print ("Text tokenized.")
        generated = model_7b.generate(batch["input_ids"], max_length=max_len, temperature=0)
        answer = tokenizer_7b.decode(generated[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
    answer = answer.split('\n')[1:]
    answer = ''.join(answer)
    print(answer)
    return answer

def get_random_answer():
    bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
    return bot_message

def respond(message, history, max_len, model_size):
    history = history or []
    output = get_reply_from_llm(message, max_len, model_size)
    #output = get_random_answer()
    history.append((message, output))
    return history, history


block = gr.Blocks()

with block:
    gr.Markdown("""<h1><center>Llama2 showroom</center></h1>""")
    with gr.Row():
        with gr.Tab("Pre-trained model size"):
            model_size = gr.Dropdown(["Taiwan-LLaMa-v1_0"], value="Taiwan-LLaMa-v1_0", label="select model size")    
        with gr.Tab("Model generation parameters"):
            with gr.Column(): 
                max_len = gr.Slider(0, 1024, value=500, label="max length")
        with gr.Column(): 
                chatbot = gr.Chatbot()

    message = gr.Textbox(placeholder="Input your question", interactive=True)
    state = gr.State()
    submit = gr.Button("SEND")
    submit.click(respond, inputs=[message, state, max_len, model_size], outputs=[chatbot, state])


block.launch(server_name="0.0.0.0")

