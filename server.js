const express = require("express");
const axios = require("axios");
const cors = require("cors");
const path = require("path");
const fs = require("fs");
const multer = require("multer");

const app = express();
const storage = multer.memoryStorage();
const upload = multer({ storage : storage});
const PORT = 3000;
const PYTHON_LLM_URL = "http://localhost:8000/generate/";  // Python 서버 주소

app.use(cors());
app.use(express.json());
app.use(express.static('static'));

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


// 유저 요청 post -> LLM 서버로 전달
app.post("/ask", upload.fields([{ name: "image", maxCount: 1 }, { name: "text", maxCount: 1 }]), async (req, res) => {
    try {
        
        const userQuery = req.body.text;
        var userImage = "";
        
        // console.log("[nodejs] 유저 전송 데이터 body :", req.body);
        console.log(`[nodejs] 유저 질의 : ${userQuery}`);
        
        // 이미지 존재여부 확인
        if (req.files && req.files.image && req.files.image.length > 0) {
            // console.log("there's image");
            userImage = req.files.image[0].buffer.toString("base64")
            req.files.image.buffer = null; // 메모리 누수 방지
        } // else console.log("there's no image");


        // Python LLM 서버로 요청 보내기.
        const response = await axios.post(PYTHON_LLM_URL, { 
            text : userQuery,
            image : userImage
        });

        console.log(`[nodejs] LLM 응답 : ${response.data.llm_response}`);
        res.json({ llm_response: response.data.llm_response });
    } catch (error) {
        console.error("[nodejs] 오류 발생 :", error.message);
    }
});

// 서버 실행
app.listen(PORT, () => {
    console.log(`node.js 서버 실행 포트 : ${PORT}`);
    console.log(`node.js 서버 실행 주소 http://localhost:${PORT}/.`);
});
