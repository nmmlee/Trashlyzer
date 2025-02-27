from pydantic import BaseModel
import pandas as pd
import re
from llama_cpp import Llama
from rapidfuzz import process, fuzz
from fastapi import FastAPI
import io
from PIL import Image
import cv2
import base64
from llm_server_utilities import *
from llm_server_data_processing import *

# fastapi 클래스 생성
app = FastAPI()

# fast api에서 데이터 받을 모델
class QueryRequest(BaseModel):
    text: str
    image: str | None = None # 





@app.post("/generate/")
async def generate_text(request: QueryRequest):
    instruction = request.text.strip() # 유저 질의
    user_image = ""
    if request.image != "":
        user_image = base64_to_img(request.image)

    # 키워드 추출 함수 (data_processing 모듈)
    extracted_keyword = extract_llm_keywords(instruction)

    cached_response = hit_cache_response(extracted_keyword)
    if cached_response != "해당 품목을 찾을 수 없습니다.":
        return {"llm_response": cached_response}
    # 유사도 검색 함수(5는 5개 반환) (data_processing 모듈)
    similar_items = find_closest_item(extracted_keyword, 5)

    # 응답생성  (data_processing 모듈)
    response_text = generate_llm_response(instruction, extracted_keyword, similar_items)

    return {"llm_response": response_text}
