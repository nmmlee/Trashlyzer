FROM python:3.12-slim
WORKDIR /docker_app
COPY ./requirements.txt /docker_app/requirements.txt

RUN apt-get update && apt-get install -y build-essential cmake
RUN apt-get update
RUN apt-get -y install libgl1
RUN apt-get -y install libglib2.0-0

RUN pip install --upgrade -r /docker_app/requirements.txt
COPY . /docker_app

EXPOSE 8000
CMD ["uvicorn", "llm_server:app", "--host", "0.0.0.0", "--port", "8000"]