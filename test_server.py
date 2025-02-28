from fastapi import FastAPI
from pydantic import BaseModel
import base64
import io
from PIL import Image

app = FastAPI()

class QueryRequest(BaseModel):
    text: str
    image: str | None = None # base64로 인코딩됨

@app.post("/generate/")
async def generate_text(request: QueryRequest):
    
    response_text = request.text + "<- 전달된 질의\n"
    print(request.image[:10] + "......")
    print(request.text)
    
    return {"llm_response": response_text}