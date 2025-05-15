from pydantic import BaseModel
import pandas as pd
import re
from llama_cpp import Llama
# from rapidfuzz import process, fuzz
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
    mode: str | None = "fast"





@app.post("/generate/")
async def generate_text(request: QueryRequest):
    llm_response = ""
    if request.mode.strip() == "fast":
        llm_response = fast_mode(request)
    elif request.mode.strip() == "premium":
        llm_response = premium_mode(request)
    else:
        return {"llm_response" : "답변 모드가 선택되지 않았습니다."}

    llm_response += """
    대형 폐가전제품 및 일부 소형 폐가전제품은 무상 수거 대상이므로, 한국전자제품자원순환공제조합(☎1599-0903)을 통해 무상 방문 수거 신청을 할 수 있습니다.
    """
    return {"llm_response" : llm_response}





def fast_mode(request: QueryRequest):
    instruction = request.text.strip() # 유저 질의
    user_image = ""
    #이미지가 삽입되었을 경우
    if request.image != "":
        #이미지 처리
        user_image = base64_to_img(request.image)
        #gemini에게 답변 는 함수
        dapjang : str = image_gumsaek(user_image)
        #에러발생시
        if dapjang == "에러발생":
            response_text = "죄송합니다. 이미지 분석이 원할하지 않습니다 :( 다음에 다시 시도해주세요!"
            return cached_response
        #gemini에게 답장을 받았다면
        else:
            #키워드 추출과 대조
            extracted_keyword = extract_llm_keywords(instruction)
            #Gemini 답장에 키워드가 있는경우
            print("gemini 이미지분석 결과 :\n" + dapjang)
            if extracted_keyword in dapjang:
                    #캐시 확인
                    cached_response: str = hit_cache_response(extracted_keyword)
                    #캐시에 있을경우
                    if cached_response != "해당 품목을 찾을 수 없습니다.":
                        print("이미지 분석 결과 동일 확인, 캐시를 리턴합니다.")
                        return cached_response
                    
                    #캐시에 없을경우
                    else:
                        # 유사도 검색 함수(5는 5개 반환) (data_processing 모듈)
                        similar_items = find_closest_item(extracted_keyword, 5)
                        #답변반환
                        response_text = generate_template_response(instruction, extracted_keyword, similar_items)
                        #cache insert 함수 일단 뺐습니다. TODO: 캐시 삽입 로직 추가
                        #답변반환
                        print("이미지 분석 결과 동일 확인, LLM 답변을 리턴하고 캐시에 저장합니다.")
                        return response_text
                    
            #Gemini 답장이 키워드에 없는경우
            else:
                #Gemini 답장 전처리
                lines = dapjang.split("\n")
                first_item = lines[0].split(". ")[1] #이미지분석 결과 1
                second_item = lines[1].split(". ")[1] #이미지분석 결과 2

                first_items = find_closest_item(first_item, 3)
                second_items = find_closest_item(second_item, 1)
                extracted_items = find_closest_item(extracted_keyword, 2) #텍스트분석 지칭대상

                #BIG llm 답변반환, 이건 구조상의 큰 변화가 있어서 따로 함수 호출
                #난 저거 버리고싶어요 라고 말하는 경우 캐시가 이상하게 잡혀서, 캐시 함수는 넣지 않았습니다.
                response_text = generate_special_template_response(
                instruction, extracted_keyword, first_items, second_items, extracted_items)
                print("이미지 분석 결과 불일치, LLM 답변을 리턴합니다.")
                print("node로 전달되는 최종답변 :\n" + response_text)
                return response_text

    #이미지가 삽입되지 않았을 경우
    else:
         # 키워드 추출 함수 (data_processing 모듈)
        extracted_keyword = instruction

    # 캐시히트시 캐시된 답변 전송
        cached_response: str = hit_cache_response(extracted_keyword)
        if cached_response != "해당 품목을 찾을 수 없습니다.":
            print(extracted_keyword)
            print("[이미지삽입x fast] cache hit, return 합니다.\n" + cached_response)
            return cached_response
    
    # 유사도 검색 함수(5는 5개 반환) (data_processing 모듈)
        similar_items = find_closest_item(extracted_keyword, 5)

    # 응답생성  (data_processing 모듈)
        response_text = generate_template_response(instruction, extracted_keyword, similar_items)
    
    # 생성된 응답 캐시에 저장
        #insert_cache(extracted_keyword, response_text)
        print("cache miss, llm답변을 리턴합니다.\n" + response_text)
        return response_text

def premium_mode(request: QueryRequest):
    instruction = request.text.strip() # 유저 질의
    user_image = ""
    #이미지가 삽입되었을 경우
    if request.image != "":
        #이미지 처리
        user_image = base64_to_img(request.image)
        #gemini에게 답변 는 함수
        dapjang : str = image_gumsaek(user_image)
        #에러발생시
        if dapjang == "에러발생":
            response_text = "죄송합니다. 이미지 분석이 원할하지 않습니다 :( 다음에 다시 시도해주세요!"
            return cached_response
        #gemini에게 답장을 받았다면
        else:
            #키워드 추출과 대조
            extracted_keyword = extract_llm_keywords(instruction)
            #Gemini 답장에 키워드가 있는경우
            print("gemini 이미지분석 결과 :\n" + dapjang)
            if extracted_keyword in dapjang:
                    #캐시 확인
                    cached_response: str = hit_cache_response(extracted_keyword)
                    #캐시에 있을경우
                    if cached_response != "해당 품목을 찾을 수 없습니다.":
                        print("이미지 분석 결과 동일 확인, 캐시를 리턴합니다.")
                        return cached_response
                    
                    #캐시에 없을경우
                    else:
                        # 유사도 검색 함수(5는 5개 반환) (data_processing 모듈)
                        similar_items = find_closest_item(extracted_keyword, 5)
                        #답변반환
                        response_text = generate_llm_response(instruction, extracted_keyword, similar_items)
                        #cache insert 함수 일단 뺐습니다. TODO: 캐시 삽입 로직 추가
                        #답변반환
                        print("이미지 분석 결과 동일 확인, LLM 답변을 리턴하고 캐시에 저장합니다.")
                        return response_text
                    
            #Gemini 답장이 키워드에 없는경우
            else:
                #Gemini 답장 전처리
                lines = dapjang.split("\n")
                first_item = lines[0].split(". ")[1] #이미지분석 결과 1
                second_item = lines[1].split(". ")[1] #이미지분석 결과 2

                first_items = find_closest_item(first_item, 3)
                second_items = find_closest_item(second_item, 1)
                extracted_items = find_closest_item(extracted_keyword, 2) #텍스트분석 지칭대상

                #BIG llm 답변반환, 이건 구조상의 큰 변화가 있어서 따로 함수 호출
                #난 저거 버리고싶어요 라고 말하는 경우 캐시가 이상하게 잡혀서, 캐시 함수는 넣지 않았습니다.
                response_text = generate_special_llm_response(
                instruction, extracted_keyword, first_items, second_items, extracted_items)
                print("이미지 분석 결과 불일치, LLM 답변을 리턴합니다.")
                print("node로 전달되는 최종답변 :\n" + response_text)
                return response_text

    #이미지가 삽입되지 않았을 경우
    else:
         # 키워드 추출 함수 (data_processing 모듈)
        extracted_keyword = extract_llm_keywords(instruction)

    # 캐시히트시 캐시된 답변 전송
        cached_response: str = hit_cache_response(extracted_keyword)
        if cached_response != "해당 품목을 찾을 수 없습니다.":
            print("추출키워드 : " + extracted_keyword)
            print("[이미지삽입x premium]cache hit, return 합니다.\n" + cached_response)
            return cached_response
    
    # 유사도 검색 함수(5는 5개 반환) (data_processing 모듈)
        similar_items = find_closest_item(extracted_keyword, 5)

    # 응답생성  (data_processing 모듈)
        response_text = generate_llm_response(instruction, extracted_keyword, similar_items)
    
    # 생성된 응답 캐시에 저장
        #insert_cache(extracted_keyword, response_text)
        print("cache miss, llm답변을 리턴합니다.\n" + response_text)
        return response_text