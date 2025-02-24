from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import re
from llama_cpp import Llama
from rapidfuzz import process, fuzz

#몰?루 내가 한거 아님
app = FastAPI()

# GGUF 모델 Path선언
MODEL_PATH_MAIN = "./models/Big.gguf"  #파라미터가 많은 대답용 LLM
MODEL_PATH_KEYWORD = "./models/Little.gguf"  # 문장에서 CSV에 검색할 키워드를 반환하는 소형 LLM
"""
이 프로젝트는 Meta의 Llama 3.1 모델과 Alibaba의 Qwen 2.5 모델을 사용합니다.

라이선스 정보:
- Llama 3.1: Meta의 Llama 3.1 License에 따라 제공됩니다
  https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE
- Qwen 2.5: Apache 2.0 License에 따라 제공됩니다.
  https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/LICENSE

Copyright © Meta Platforms, Inc. / Alibaba Group Holding Limited.
"""


#모델 불러오기
llm_main = Llama(model_path=MODEL_PATH_MAIN, n_gpu_layers=0, n_ctx=4800, verbose=False)
llm_keyword = Llama(model_path=MODEL_PATH_KEYWORD, n_gpu_layers=0, n_ctx=4800, verbose=False)

# csv 불러오기, 품목에서 유사도 검색에 방해되는 (), / 제외한 csv 사용.
file_path = "./대형폐기물분류표_정제.csv"
df = pd.read_csv(file_path)
items = df["품목"].astype(str).unique().tolist()

#???
class QueryRequest(BaseModel):
    text: str
    image: str | None = None # 
    
#LLM에서 키워드 추출
def extract_llm_keywords(user_input: str):
    prompt = f"""
- **사용자의 입력**에서 **버리려는** **핵심 품목**을 한 단어로만 반환하세요.
- 예제: '나 2층 침대 사다리 버리고 싶어' → '2층 침대 사다리'
- 문장의 형태가 되면 안 됩니다.
- 다른 설명 없이 **핵심 품목 키워드만** 반환하세요.
- 한 개의 단어는 꼭 포함되어야 합니다.

사용자 입력: {user_input}
출
출력:
"""
    #End of text 선언, 키워드 추출
    response = llm_keyword(prompt, max_tokens=15, temperature=0, stop=["\n", "<|endoftext|>", "<|im_end|>"])
    extracted_keyword = response["choices"][0]["text"].strip()

    #필터링
    extracted_keyword = re.sub(r"<\|.*?\|>", "", extracted_keyword).strip()
    extracted_keyword = extracted_keyword.split("\n")[0]  
    extracted_keyword = re.sub(r"[^\w\s]", "", extracted_keyword) 
    #반환된 글자에 한글이 없을시 예) ```, 공백 시 문장 반환
    if not re.search(r"[가-힣]", extracted_keyword):
        extracted_keyword = user_input.strip()
    return extracted_keyword

#유사도 검색, Fuzz 유사도
def find_closest_item(query: str, top_k):
    matches = process.extract(query, items, scorer=fuzz.partial_ratio, limit=top_k)
    
    # 유사도 점수 컷
    best_matches = [match[0] for match in matches if match[1] >= 55]

    return best_matches if best_matches else ["해당 품목을 찾을 수 없습니다."]

#응답생성 코드(파리미터 많은 LLM)
def generate_llm_response(user_input: str, extracted_keyword: str, matched_items: list):
    prompt = f"""
[역할]
당신은 대형 폐기물 배출 수수료를 안내하는 챗봇입니다.
사용자의 질문을 읽고, 정확한 폐기물 정보를 제공합니다.

[답변 규칙]
1. 반드시 배출 규격과 가격 정보를 포함하세요.
2. 불필요한 설명을 추가하지 말고, 핵심 정보만 전달하세요.
3. 리스트 형식이 아니라, 자연스럽게 문장으로 설명하세요.
4. 유리 별도는 유리는 따로 돈을 받는다는 것이고, 유리는 아래와 같이 수수료를 내야합니다.
"가구류","유리","긴 면이 50cm 마다(두께 8mm 미만)",1000
"가구류","유리","긴 면이 50cm 마다(두께 8mm 이상)",1500
5. '장롱 옷장'처럼 두 개의 품목이 함께 표시된 경우, 두 품목 모두 해당됩니다. 수수료를 안내해야 합니다.


[사용자 질문]
{user_input}

[검색된 품목 및 수수료 정보]
"""
    if not matched_items or matched_items == ["해당 품목을 찾을 수 없습니다."]:
        prompt += "해당 품목을 찾을 수 없습니다.\n"
    else:
        for item in matched_items:
            relevant_data = df[df["품목"] == item][["규격", "가격"]].drop_duplicates()
            prompt += f"{item} 중 "
            details = []
            for _, row in relevant_data.iterrows():
                details.append(f"{row['규격']}은 {row['가격']}원")

            prompt += ", ".join(details) + "입니다.\n"
    prompt += "\n\n[최종 답변]:"
    print(prompt)
    response = llm_main(
        prompt,
        max_tokens=300,  
        stop=["\n\n", "<|endoftext|>", "<|im_end|>"],  
        temperature=0.1,  
        top_p=0.8,
        repeat_penalty=1.2,
    )
    
    final_response = response["choices"][0]["text"].strip()
    final_response = re.sub(r"<\|.*?\|>", "", final_response).strip()  
    
    return final_response

@app.post("/generate/")
async def generate_text(request: QueryRequest):
    instruction = request.text.strip()

    # 키워드 추출 함수
    extracted_keyword = extract_llm_keywords(instruction)
    # 유사도 검색 함수(5는 5개 반환)
    similar_items = find_closest_item(extracted_keyword, 5)

    # 응답생성
    response_text = generate_llm_response(instruction, extracted_keyword, similar_items)

    return {"llm_response": response_text}
