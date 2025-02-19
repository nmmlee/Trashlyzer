from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    text: str

@app.post("/generate/")
async def generate_text(request: QueryRequest):
    
    # 여기서부터 LLM 로직으로 수정
    response_text = request.text + "!!"
    return {"llm_response": response_text}