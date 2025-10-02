const express = require("express");
const axios = require("axios");
const cors = require("cors");
const path = require("path");
const fs = require("fs");
const multer = require("multer");
var rateLimit = require("express-rate-limit");
const { createClient } = require('redis');
const { v4: uuidv4 } = require('uuid');

const app = express();
const storage = multer.memoryStorage();
const upload = multer({
    storage : storage,
    limits : { fileSize : 10 * 1024 * 1024}, // 업로드 총(질의 + 사진) 10MB 제한
});
const redis = createClient({ url: 'redis://trashlyzer_redis:6379' });

const PORT = 3000;
const PYTHON_LLM_URL = "http://trashlyzer_fastapi_server:8000/generate/";  // Python 서버 주소

app.use(cors());
app.use(express.json());
app.use(express.static('static'));
redis.connect()


// 동일 ip 잦은 llm 연산 요청 차단
app.use("/ask", rateLimit({
    windowMs: 30 * 1000, // 30초 간격
    max: 5, // windowMs동안 최대 호출 횟수
    handler(req, res) { // 제한 초과 시 콜백 함수
        res.status(this.statusCode).json({
          code: this.statusCode, // statusCode 기본값은 429
          message: 'TOO MANY REQUESTS',
          llm_response: '요청이 너무 많습니다. 잠시만 기다려주세요',
       });
    },
}));


// 메인 홈 get요청 처리
app.get('/', (request, response) => {
    fs.readFile('index.html', 'UTF-8', (err, data) => {
        if (err) { 
            response.send('No html');
        }
        response.send(data);
    })
  })

  app.get('/style.css', (request, response) => {
    fs.readFile('static/style.css', 'UTF-8', (err, data) => {
        if (err) { 
            response.send('No css');
        }
        response.send(data);
    })
  })


// 유저 요청 post -> 작업 큐 등록, LLM 서버로 전달
app.post("/ask", upload.fields([{ name: "image", maxCount: 1 }, { name: "text", maxCount: 1 }]), async (req, res) => {

    const task_id = uuidv4();; // 작업 ID
    await redis.set(task_id, JSON.stringify({ status: "waiting" })); // 작업 등록
    res.json({ 
        "task_id" : task_id,
        "llm_response" : "답변 생성 중입니다. AI답변 생성의 경우, 1개의 답변 생성에 5~6분의 시간이 소요됩니다."
    }); // 일단 응답 바로 보냄("/result/id에 get으로 작업완료 주기적으로 확인하게 됨)

    try {
        const userQuery = req.body.text;
        var userImage = "";
        
        // console.log("[nodejs] 유저 전송 데이터 body :", req.body);
        console.log(`[nodejs] request받은 유저 질의 : ${userQuery}`);
        
        // 이미지 1개 존재 확인
        if (req.files && req.files.image && req.files.image.length == 1) {
            console.log(`[nodejs] request받은 이미지 용량 : ${req.files.image[0].buffer.length / 1048576}MB`);
            // 이미지 용량 검증 (8MB 넘으면 python서버에 이미지 전송 X)
            if (req.files.image[0].buffer.length < 8388608) {
                userImage = req.files.image[0].buffer.toString("base64") // base64 인코딩
            }
            req.files.image.buffer = null; // 메모리 누수 방지
        }

        // Python LLM 서버로 요청 보내기.
        const response = await axios.post(PYTHON_LLM_URL, { 
            text : userQuery,
            image : userImage,
            mode : req.body.mode
        })
        
        await redis.set(task_id, JSON.stringify({
            status: "done",
            llm_response: response.data.llm_response
        }));
        

        // console.log(`[nodejs] 생성된 LLM 응답 : ${response.data.llm_response}`);
        // res.json({ llm_response: response.data.llm_response });
    } catch (error) {
        console.error("[nodejs] 오류 발생 :", error.message);
    }
});

// 작업큐 완료됐나 확인
app.get("/result/:task_id", async (req, res) => {
    const raw = await redis.get(req.params.task_id);
  
    if (!raw) {
        return res.status(404).json({ error: "존재하지 않는 task_id" });
    }
  
    const parsed = JSON.parse(raw);
    if (parsed.status === "waiting") {
        return res.json({ status: "waiting" });
    }
    else {
        console.log(parsed)
        return res.json(parsed);
    }
});

// 서버 실행
app.listen(PORT, () => {
    console.log(`node.js 서버 실행 포트 : ${PORT}`);
    console.log(`node.js 서버 실행 주소 http://localhost:${PORT}/.`);
});
