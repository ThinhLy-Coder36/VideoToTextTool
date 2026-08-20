# Sử dụng Python image chính thức (slim để dung lượng nhẹ)
FROM python:3.10-slim

# Thiết lập các biến môi trường
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/hf_cache

# Cài đặt FFmpeg và các dependency hệ thống
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc và user không phải root (Hugging Face Spaces yêu cầu chạy với user ID 1000)
RUN useradd -m -u 1000 user
WORKDIR /app

# Sao chép file dependency và cài đặt thư viện
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt && \
    pip install fastapi uvicorn python-multipart jinja2

# Tạo thư mục cache cho HuggingFace và uploads, gán quyền cho user 1000
RUN mkdir -p /tmp/hf_cache && \
    mkdir -p uploads && \
    chown -R user:user /app /tmp/hf_cache

# Chuyển sang user không phải root
USER user

# Sao chép toàn bộ mã nguồn vào container
COPY --chown=user:user . .

# Mở cổng 7860 (Hugging Face Spaces yêu cầu ứng dụng chạy ở cổng 7860)
EXPOSE 7860

# Lệnh khởi chạy ứng dụng
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
