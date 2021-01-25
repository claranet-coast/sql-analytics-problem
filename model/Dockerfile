FROM python:3.8

COPY artifacts artifacts
COPY requirements.txt requirements.txt
COPY main.py main.py

RUN pip install -r requirements.txt && rm -rf /root/.cache

ENTRYPOINT ["python", "main.py"]
