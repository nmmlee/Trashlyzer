from pydantic import BaseModel
import pandas as pd
import re
from llama_cpp import Llama
from rapidfuzz import process, fuzz
import pymysql # Mariadb 커넥트
import json
from PIL import Image
import google.generativeai as genai
import cv2

QUERY_FIND = '''SELECT response FROM cache WHERE keyword = %s'''
QUERY_INSERT = '''
            INSERT INTO cache (keyword, response)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE response = VALUES(response);
            '''

# GGUF 모델 Path선언
MODEL_PATH_MAIN = "./models/Big.gguf"  #파라미터가 많은 대답용 LLM
MODEL_PATH_KEYWORD = "./models/Big.gguf"  # 문장에서 CSV에 검색할 키워드를 반환하는 소형 LLM
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
items = df["품목"].astype(str).unique().tolist() # TODO: items변수, 및 관련로직 MariaDB에서 가져오는 것으로 수정하기

#LLM에서 키워드 추출
def extract_llm_keywords(user_input: str):
    prompt = f"""
- **사용자의 입력**에서 **버리려는** **핵심 품목**을 한 단어로만 반환하세요.
- 예제: '나 2층 침대 사다리 버리고 싶어' → '2층침대사다리'
- 예제: '나 차량용 시트를 버리고싶어' -> '차량용시트'
- 예제: '나 2층침대의 사다리를 버리고싶어' -> '2층침대사다리'
- 문장의 형태가 되면 안 됩니다.
- 다른 설명 없이 **핵심 품목 키워드만** 반환하세요.
- 한 개의 단어는 꼭 포함되어야 합니다.
- 반드시 한국어로 답변하세요
예) 캐리어를 버리고 싶어요 -> "캐리어"

사용자 입력: {user_input}

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

# 쿼리 사용시에만 db 연결. 테이블 cache 컬럼 keyword, response
def execute_query(query: str, *args):
    # 커넥션 정보 불러오기
    f = open("secrets.json", "r")
    config = json.load(f)
    host = config["MariaDB"]["host"]
    port = config["MariaDB"]["port"]
    user = config["MariaDB"]["user"]
    password = config["MariaDB"]["password"]
    db = config["MariaDB"]["database"]
    connection = pymysql.connect(host='localhost',
                        port=port,
                        user=user,
                        password=password,
                        db = db,
                        charset='utf8mb4')
    cursor = connection.cursor()
    if len(args) == 0: # 빈 튜플(args없이 실행)
        result = cursor.execute(query)
    else:
        result = cursor.execute(query, args) # 기본값 = 영향받은 row 수
    try:
        # SELECT 문이면 SELECT 결과(튜플) 리턴
        if (query.strip().upper().startswith("SELECT")):
            result = cursor.fetchall()
        # 이외에는 실행확정하고 row 수 반납(int)
        else:
            cursor.commit()
        cursor.close()
        connection.close()
        return result
    
    except Exception as e:
        cursor.close()
        connection.close()
        print(e) # 오류 발생
        return result # 오류시 주로 0 리턴됨(=영향받은 row 0개)

    

# 캐싱 함수
def hit_cache_response(keyword: str):
    # 캐시 테이블 키워드 존재 검색
    response = execute_query(QUERY_FIND, keyword.strip())
# 캐시 미스
    if response == None or response == ():
        return "해당 품목을 찾을 수 없습니다."
    # 캐시 히트
    else:
        return response

# 키워드-응답 데이터 캐시에 저장
def insert_cache(keyword: str, response: str):
    return execute_query(QUERY_INSERT, keyword.strip(), response.strip())
    

#유사도 검색, Fuzz 유사도
def find_closest_item(query: str, top_k):
    matches = process.extract(query, items, scorer=fuzz.partial_ratio, limit=top_k)
    
    # 유사도 점수 컷
    best_matches = [match[0] for match in matches if match[1] >= 55]
    print(best_matches)
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
6. 구체적인 정보가 필요한경우 검색된 품목들을 규격과 가격을 최대한 설명하시고, 그 다음에 추가 정보를 요구하세요.

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

#응답생성 코드(파리미터 많은 LLM), 이미지 가 있고, 질문이 이에 맞지 않을 경우
#너무 바꿀 내용이 커서 이렇게 따로 뺄수밖에 없던 점 양해 부탁
def generate_special_llm_response(user_input: str, extracted_keyword: str, 
                                  first_items: list, second_items: list, extracted_items: list):
    prompt = f"""
[역할]
당신은 대형 폐기물 배출 수수료를 안내하는 챗봇입니다.
사용자가 이미지를 업로드 했고, 그걸 2개의 품목으로 분석했습니다. 
[이미지 분석 첫 번째 품목] [이미지 분석 두 번째 품목] [텍스트 분석 품목] 3가지를 읽고, 사용자의 질문에 답하세요.

[답변 규칙]
1. 반드시 배출 규격과 가격 정보를 포함하세요.
2. 불필요한 설명을 추가하지 말고, 핵심 정보만 전달하세요.
3. 리스트 형식이 아니라, 자연스럽게 문장으로 설명하세요.
4. 유리 별도는 유리는 따로 돈을 받는다는 것이고, 유리는 아래와 같이 수수료를 내야합니다.
   - 가구류 유리(두께 8mm 미만): 긴 면이 50cm 마다 1000원
   - 가구류 유리(두께 8mm 이상): 긴 면이 50cm 마다 1500원
5. '장롱 옷장'처럼 두 개의 품목이 함께 표시된 경우, 두 품목 모두 해당됩니다. 수수료를 안내해야 합니다.
6. 이미지 업로드를 했기에 질문이 폐기물 배출 수수료와 관련이 없을수도 있습니다. 그럴땐 꼭 이미지 분석 첫번째 품목을 설명하도록 하세요.
예) 저거 얼마인가요?, 저거 뭔지 알겠어? 
7. 어떠한 경우에도 수수료 정보를 포함하도록 하세요.
[사용자 질문]
{user_input}

[이미지 분석 첫 번째 품목]
"""
    if not first_items or first_items == ["해당 품목을 찾을 수 없습니다."]:
        prompt += "해당 품목을 찾을 수 없습니다.\n"
    else:
        for item in first_items:
            relevant_data = df[df["품목"] == item][["규격", "가격"]].drop_duplicates()
            prompt += f"{item}: "
            details = [f"{row['규격']}은 {row['가격']}원" for _, row in relevant_data.iterrows()]
            prompt += ", ".join(details) + "\n"

    prompt += "\n[이미지 분석 두 번째 품목]\n"
    if not second_items or second_items == ["해당 품목을 찾을 수 없습니다."]:
        prompt += "해당 품목을 찾을 수 없습니다.\n"
    else:
        for item in second_items:
            relevant_data = df[df["품목"] == item][["규격", "가격"]].drop_duplicates()
            prompt += f"{item}: "
            details = [f"{row['규격']}은 {row['가격']}원" for _, row in relevant_data.iterrows()]
            prompt += ", ".join(details) + "\n"

    prompt += "\n[텍스트 분석 품목]\n"
    if not extracted_items or extracted_items == ["해당 품목을 찾을 수 없습니다."]:
        prompt += "해당 품목을 찾을 수 없습니다.\n"
    else:
        for item in extracted_items:
            relevant_data = df[df["품목"] == item][["규격", "가격"]].drop_duplicates()
            prompt += f"{item}: "
            details = [f"{row['규격']}은 {row['가격']}원" for _, row in relevant_data.iterrows()]
            prompt += ", ".join(details) + "\n"

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

#이미지 검색 함수
def image_gumsaek(user_image):
    try:
        f1 = open("api_token.json", "r")  
        config1 = json.load(f1)
        token = config1["gemini_api"]["token"]

        genai.configure(api_key=token)

        user_image = Image.fromarray(cv2.cvtColor(user_image, cv2.COLOR_BGR2RGB))

        model = genai.GenerativeModel("gemini-2.0-flash")  # 최신 2.0 Flash 모델 사용

        image_prompt = """
        -이 이미지가 뭔지 한 단어로 짧게 설명하세요.
        -이미지 예상 단어를 3개 뽑으세요.

        답변 예시: 
        1.의자
        2.책상
        3.휴지

        이런 식으로 답변하세요. 부가적인 설명을 붙이지 마세요. 한국어로 설명해주세요.
        """

        gemini_dapjang = model.generate_content([user_image, image_prompt])

        return gemini_dapjang.text

    except Exception as e:
        return "에러발생"
