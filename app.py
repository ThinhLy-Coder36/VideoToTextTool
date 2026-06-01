import os
import uuid
import shutil
import logging
import threading
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Form, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Import transcriber
from transcriber import VideoTranscriber

# Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("web-service")

app = FastAPI(title="Video-to-Text Transcription Service")

# Global state to keep track of tasks
# task_id -> { "status": "processing/completed/failed", "filename": str, "progress": int, "total": int, "text": str, "error": str }
tasks_db = {}

# Ensure required directories exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Path to the model
MODEL_PATH = Path("models") / "vosk-model-small-vn-0.4"


def run_transcription_task(task_id: str, video_path: str, filename: str, engine: str, workers: int, is_temp_file: bool):
    """
    Chạy tác vụ trích xuất và nhận dạng giọng nói trong luồng nền.
    """
    try:
        tasks_db[task_id]["status"] = "processing"
        
        # Callback cập nhật tiến trình
        def progress_callback(done, total):
            tasks_db[task_id]["progress"] = done
            tasks_db[task_id]["total"] = total

        # Cấu hình transcriber
        model_dir = str(MODEL_PATH) if (engine == "vosk" and MODEL_PATH.exists()) else None
        transcriber = VideoTranscriber(
            model_path=model_dir,
            language="vi-VN",
            max_workers=workers,
            chunk_duration=60
        )

        log.info(f"Bắt đầu dịch task {task_id}: {filename}")
        text = transcriber.transcribe(video_path, progress_callback=progress_callback, engine=engine)
        
        # Hoàn thành
        tasks_db[task_id]["status"] = "completed"
        tasks_db[task_id]["text"] = text
        tasks_db[task_id]["progress"] = tasks_db[task_id]["total"] # Đảm bảo 100%
        log.info(f"Hoàn thành task {task_id} thành công ✓")

    except Exception as e:
        log.error(f"Lỗi khi dịch task {task_id}: {e}")
        tasks_db[task_id]["status"] = "failed"
        tasks_db[task_id]["error"] = str(e)
    
    finally:
        # Xóa file video tạm nếu được upload lên
        if is_temp_file and os.path.exists(video_path):
            try:
                os.remove(video_path)
                log.info(f"Đã xóa file tạm: {video_path}")
            except Exception as ex:
                log.warning(f"Không thể xóa file tạm {video_path}: {ex}")


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Trả về giao diện HTML chính."""
    index_path = Path("templates") / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy index.html")
    
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/transcribe")
async def start_transcription(
    background_tasks: BackgroundTasks,
    video_path: Optional[str] = Form(None),
    engine: str = Form("vosk"),
    workers: int = Form(4),
    file: Optional[UploadFile] = File(None)
):
    """Bắt đầu tác vụ chuyển đổi video."""
    task_id = str(uuid.uuid4())
    is_temp_file = False
    target_path = ""
    filename = ""

    if video_path:
        # Sử dụng đường dẫn cục bộ trực tiếp
        resolved_path = Path(video_path).resolve()
        if not resolved_path.exists():
            return JSONResponse(status_code=400, content={"detail": f"Đường dẫn file không hợp lệ hoặc không tồn tại: {video_path}"})
        target_path = str(resolved_path)
        filename = resolved_path.name
    elif file:
        # Xử lý file upload lên server
        filename = file.filename
        target_path = str(UPLOAD_DIR / f"{task_id}_{filename}")
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        is_temp_file = True
    else:
        return JSONResponse(status_code=400, content={"detail": "Vui lòng nhập đường dẫn file cục bộ hoặc tải lên một file video!"})

    # Tạo record task trong DB tạm thời
    tasks_db[task_id] = {
        "status": "pending",
        "filename": filename,
        "progress": 0,
        "total": 0,
        "text": "",
        "error": ""
    }

    # Đưa tác vụ vào chạy nền
    background_tasks.add_task(
        run_transcription_task,
        task_id=task_id,
        video_path=target_path,
        filename=filename,
        engine=engine,
        workers=workers,
        is_temp_file=is_temp_file
    )

    return {"task_id": task_id, "status": "pending"}


@app.get("/api/status/{task_id}")
async def get_task_status(task_id: str):
    """Lấy trạng thái tiến trình của tác vụ."""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Không tìm thấy task")
    return tasks_db[task_id]


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*80)
    print("  DỊCH VỤ CHUYỂN VIDEO THÀNH TEXT OFFLINE ĐANG KHỞI CHẠY...")
    print("  Vui lòng mở trình duyệt và truy cập: http://127.0.0.1:8000")
    print("="*80 + "\n")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
