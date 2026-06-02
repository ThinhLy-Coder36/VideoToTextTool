---
title: Video To Text Tool
emoji: 🎥
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Video-to-Text Transcription Service


Chuyển đổi video (mp4, mkv, avi, mov…) thành text — **hoàn toàn offline** với Vosk, không cần AI API.

---

## Yêu cầu hệ thống

| Phần mềm | Cài đặt |
|----------|---------|
| **Python 3.8+** | https://python.org |
| **FFmpeg** | https://ffmpeg.org/download.html → thêm vào PATH |

---

## Cài đặt

```bash
# 1. Cài thư viện Python
pip install -r requirements.txt

# 2. Tải Vosk model (chọn một):

#   Tiếng Việt (nhỏ ~40 MB):
#   https://alphacephei.com/vosk/models/vosk-model-small-vn-0.4.zip

#   Tiếng Việt (lớn, chính xác hơn ~1.5 GB):
#   https://alphacephei.com/vosk/models/vosk-model-vn-0.4.zip

#   Tiếng Anh:
#   https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip

# 3. Giải nén model vào thư mục, ví dụ: ./models/vosk-model-vn-0.4
```

---

## Sử dụng

### Cách 1 — Command line (khuyên dùng)

```bash
# Tiếng Việt với Vosk (offline, không cần internet)
python transcriber.py video.mp4 --model ./models/vosk-model-vn-0.4

# Tiếng Anh
python transcriber.py video.mp4 --model ./models/vosk-model-en-us-0.22 --lang en-US

# Chỉ định file output
python transcriber.py video.mp4 --model ./models/vosk-model-vn-0.4 -o output.txt

# Không có model → dùng Google Web Speech (cần internet)
python transcriber.py video.mp4 --lang vi-VN
```

### Cách 2 — Dùng trong Python code

```python
from transcriber import VideoTranscriber

# Khởi tạo với Vosk model (offline)
transcriber = VideoTranscriber(
    model_path    = "./models/vosk-model-vn-0.4",
    chunk_duration= 60,   # mỗi chunk 60 giây
    max_workers   = 4,    # 4 luồng song song
)

# Chuyển đổi video
text = transcriber.transcribe(
    video_path  = "my_video.mp4",
    output_path = "transcript.txt",   # tuỳ chọn
)

print(text)
```

---

## Tất cả tham số CLI

| Tham số | Mô tả | Mặc định |
|---------|-------|---------|
| `video` | File video đầu vào | *(bắt buộc)* |
| `-o, --output` | File text đầu ra | `<tên_video>_transcript.txt` |
| `-m, --model` | Thư mục Vosk model | `None` (dùng Google) |
| `--lang` | Ngôn ngữ Google Speech | `vi-VN` |
| `--chunk` | Giây mỗi chunk | `60` |
| `--workers` | Số luồng song song | `4` |
| `--rate` | Sample rate (Hz) | `16000` |

---

## Kiến trúc xử lý

```
Video (mp4/mkv/…)
        │
        ▼  FFmpeg
  Audio (WAV 16kHz mono)
        │
        ▼  wave module
  [chunk_0000.wav] [chunk_0001.wav] … [chunk_N.wav]   ← mỗi chunk 60s
        │              │                   │
        ▼              ▼                   ▼
   Vosk / SR       Vosk / SR          Vosk / SR        ← song song N luồng
        │              │                   │
        └──────────────┴───────────────────┘
                        │
                        ▼  ghép theo thứ tự
                  Full Transcript (text)
                        │
                        ▼
                  transcript.txt
```

---

## Ghi chú

- **Video 40 phút** với 4 luồng: thường hoàn thành trong **5-15 phút** tuỳ CPU.
- **Vosk** chạy hoàn toàn offline — không gửi dữ liệu ra ngoài.
- **Google Web Speech** (fallback) cần internet và giới hạn dung lượng mỗi request.
- Độ chính xác phụ thuộc vào chất lượng âm thanh và model chọn.
