import gradio as gr
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from openxlab.model import download

# from lmdeploy import pipeline, TurbomindEngineConfig
# backend_config = TurbomindEngineConfig(cache_max_entry_count=0.2)

# pipe = pipeline('/home/user/LQ/A_LLM/internLM2_demo/root/internlm2-chat-7b-4bit/',
#                 backend_config=backend_config)
# response = pipe(['Hi, pls intro yourself', '上海是'])
# print(response)


base_path = './internlm2-chat-7b'
os.system(f'git clone https://code.openxlab.org.cn/Liq22/internLM-chat-7B.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True, torch_dtype=torch.float16).cuda()

def chat(message,history):
    for response,history in model.stream_chat(tokenizer,message,history,max_length=2048,top_p=0.7,temperature=1):
        yield response

gr.ChatInterface(chat,
                 title="InternLM2-Chat-7B-int4",
                description="""
InternLM is mainly developed by Shanghai AI Laboratory.  
                 """,
                 ).queue(1).launch()
