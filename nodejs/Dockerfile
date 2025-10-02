FROM node:lts-jod

WORKDIR /docker_app

COPY package*.json ./

RUN npm install

COPY . .

# 3000 port
EXPOSE 3000

CMD ["node", "server.js"]