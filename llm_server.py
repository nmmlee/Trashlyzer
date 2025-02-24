from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import base64
import io
from PIL import Image

app = FastAPI()

# 토큰 불러오기
chatgpt_token : str = ""
with open('api_token.json', 'r') as f:
    try:
        chatgpt_token = json.load(f)['chatgpt']['token']
        print("successfully loaded : " + chatgpt_token[:3] + "****...")
    except Exception as e:
        chatgpt_token = ""
        print("Error : ", e)


MODEL_ID = "Bllossom/llama-3.2-Korean-Bllossom-3B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

class QueryRequest(BaseModel):
    text: str
    image: str | None = None # base64로 인코딩된 이미지 파일

@app.post("/generate/")
async def generate_text(request: QueryRequest):
    instruction = request.text

    messages = [{"role": "user", "content": instruction}]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    terminators = [
        tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask, 
        max_new_tokens=1024,
        eos_token_id=terminators,
        pad_token_id=model.config.pad_token_id,  
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )

    generated_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    print(f"Generated Response: {generated_text}") 

    return {"llm_response": str(generated_text)} 
