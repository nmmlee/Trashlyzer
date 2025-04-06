from fastapi import FastAPI
from pydantic import BaseModel
import base64
import io
from PIL import Image
import json
import pymysql

app = FastAPI()

class QueryRequest(BaseModel):
    text: str
    image: str | None = None # base64로 인코딩됨

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
    connection = pymysql.connect(host=host,
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
        # SELECT 문이면 SELECT 결과 리턴
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

@app.post("/generate/")
async def generate_text(request: QueryRequest):
    
    response_text = request.text + "<- 전달된 질의\n\n\n"
    print(request.image[:10] + "......")
    print(request.text)
    
    query_test = execute_query(r"select keyword from cache where keyword like '%침대%'")
    for i in query_test:
        response_text += i[0] + '\n'
    
    return {"llm_response": response_text}