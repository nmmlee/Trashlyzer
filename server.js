const express = require("express");
const axios = require("axios");
const cors = require("cors");
const path = require("path");
const fs = require("fs");

const app = express();
const PORT = 3000;
const PYTHON_LLM_URL = "http://localhost:8000/generate/";  // Python 서버 주소

app.use(cors());
app.use(express.json());
app.use(express.static('static'));
//app.use(express.static(path.join(__dirname, "public")));  // public 폴더를 정적 파일 제공

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
app.post("/ask", async (req, res) => {
    try {
        const userQuery = req.body.text;
        console.log(`유저 질의 : ${userQuery}`);

        // Python LLM 서버로 요청 보내기
        const response = await axios.post(PYTHON_LLM_URL, { text: userQuery });

        console.log(`LLM 응답 : ${response.data.llm_response}`);
        res.json({ llm_response: response.data.llm_response });
    } catch (error) {
        console.error("오류 발생 : ", error.message);
    }
});

// 🔹 서버 실행
app.listen(PORT, () => {
    console.log(`node.js 서버 실행 포트 : ${PORT}`);
    console.log(`node.js 서버 실행 주소 http://localhost:${PORT}/.`);
});
