FROM python:3.14.2-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    fastapi uvicorn python-multipart
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

WORKDIR /src
COPY app.py .

CMD ["python3", "app.py"]
