from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    text: str

@app.post("/generate/")
async def generate_text(request: QueryRequest):
    
    response_text = request.text + "!!"
    return {"llm_response": response_text}