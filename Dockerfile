FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install numpy opencv-python pillow matplotlib scikit-image scipy

CMD ["python", "watermark.py"]