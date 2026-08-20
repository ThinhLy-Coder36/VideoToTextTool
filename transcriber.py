"""
Video-to-Text Transcription Service
=====================================
Chuyển đổi video thành text sử dụng Vosk (offline, không cần AI API).
Hỗ trợ video dài (40+ phút) bằng cách xử lý theo từng chunk.

Yêu cầu:
  - FFmpeg đã cài đặt và có trong PATH
  - Python 3.8+
  - Các thư viện: xem requirements.txt
"""

import os
import sys
import json
import time
import wave
import shutil
import logging
import argparse
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
# Đảm bảo terminal Windows không bị lỗi hiển thị ký tự Unicode tiếng Việt
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
if sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# ---------------------------------------------------------------------------
# Cài đặt logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# Bước 1: Trích xuất audio từ video bằng FFmpeg
# ===========================================================================

def find_ffmpeg() -> str:
    """Tự động tìm kiếm đường dẫn FFmpeg nếu không có trong PATH."""
    # 1. Kiểm tra trong PATH hệ thống
    if shutil.which("ffmpeg"):
        return "ffmpeg"

    # 2. Tìm kiếm trong thư mục Packages/Links của WinGet
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    if local_appdata:
        winget_packages = Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
        if winget_packages.exists():
            ffmpeg_exes = list(winget_packages.glob("**/ffmpeg.exe"))
            if ffmpeg_exes:
                log.info(f"Tìm thấy FFmpeg tại WinGet Packages: {ffmpeg_exes[0]}")
                return str(ffmpeg_exes[0])

        winget_links = Path(local_appdata) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"
        if winget_links.exists():
            log.info(f"Tìm thấy FFmpeg tại WinGet Links: {winget_links}")
            return str(winget_links)

    # 3. Tìm kiếm trong Program Files
    program_files = os.environ.get("ProgramFiles", "")
    if program_files:
        ffmpeg_pf = Path(program_files) / "FFmpeg" / "bin" / "ffmpeg.exe"
        if ffmpeg_pf.exists():
            log.info(f"Tìm thấy FFmpeg tại Program Files: {ffmpeg_pf}")
            return str(ffmpeg_pf)

    return "ffmpeg"


def extract_audio(video_path: str, output_wav: str, sample_rate: int = 16000) -> str:
    """
    Trích xuất audio từ video và chuyển thành WAV PCM 16-bit mono.

    Parameters
    ----------
    video_path  : Đường dẫn file video đầu vào
    output_wav  : Đường dẫn file WAV đầu ra
    sample_rate : Sample rate (Hz) — Vosk thường dùng 16000

    Returns
    -------
    Đường dẫn file WAV đã tạo
    """
    ffmpeg_exe = find_ffmpeg()
    
    # Kiểm tra xem có chạy được không
    if ffmpeg_exe == "ffmpeg" and not shutil.which("ffmpeg"):
        raise EnvironmentError(
            "FFmpeg không tìm thấy trong PATH hoặc thư mục cài đặt mặc định.\n"
            "Tải tại: https://ffmpeg.org/download.html"
        )

    cmd = [
        ffmpeg_exe,
        "-y",                        # overwrite nếu tồn tại
        "-i", video_path,            # file đầu vào
        "-vn",                       # bỏ stream video
        "-acodec", "pcm_s16le",      # PCM 16-bit little-endian
        "-ar", str(sample_rate),     # sample rate
        "-ac", "1",                  # mono
        output_wav,
    ]

    log.info(f"Đang trích xuất audio từ: {video_path}")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg lỗi:\n{result.stderr}")

    log.info(f"Audio đã lưu tại: {output_wav}")
    return output_wav


# ===========================================================================
# Bước 2: Chia audio thành chunks
# ===========================================================================

def split_wav_into_chunks(wav_path: str, chunk_dir: str, chunk_duration_sec: int = 60) -> list[str]:
    """
    Chia file WAV thành nhiều chunk nhỏ để xử lý song song.

    Parameters
    ----------
    wav_path          : File WAV đầu vào
    chunk_dir         : Thư mục lưu các chunk
    chunk_duration_sec: Độ dài mỗi chunk (giây), mặc định 60s

    Returns
    -------
    Danh sách đường dẫn các file chunk theo thứ tự
    """
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_paths = []

    with wave.open(wav_path, "rb") as wf:
        sample_rate   = wf.getframerate()
        n_channels    = wf.getnchannels()
        sampwidth     = wf.getsampwidth()
        total_frames  = wf.getnframes()
        frames_per_chunk = sample_rate * chunk_duration_sec

        chunk_idx = 0
        frames_read = 0

        while frames_read < total_frames:
            frames = wf.readframes(frames_per_chunk)
            if not frames:
                break

            chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_idx:04d}.wav")
            with wave.open(chunk_path, "wb") as cw:
                cw.setnchannels(n_channels)
                cw.setsampwidth(sampwidth)
                cw.setframerate(sample_rate)
                cw.writeframes(frames)

            chunk_paths.append(chunk_path)
            frames_read += frames_per_chunk
            chunk_idx   += 1

    log.info(f"Đã chia thành {len(chunk_paths)} chunks ({chunk_duration_sec}s/chunk)")
    return chunk_paths


# ===========================================================================
# Bước 3: Nhận dạng giọng nói bằng Vosk
# ===========================================================================

def transcribe_chunk_vosk(chunk_path: str, model) -> tuple[int, str]:
    """
    Nhận dạng giọng nói một chunk bằng Vosk.

    Parameters
    ----------
    chunk_path : Đường dẫn file WAV chunk
    model      : Đối tượng vosk.Model đã load

    Returns
    -------
    (chunk_index, text) — index lấy từ tên file
    """
    from vosk import KaldiRecognizer  # import ở đây để tránh lỗi nếu không cài

    chunk_idx = int(Path(chunk_path).stem.split("_")[1])
    text_parts = []

    with wave.open(chunk_path, "rb") as wf:
        sample_rate = wf.getframerate()
        rec = KaldiRecognizer(model, sample_rate)
        rec.SetWords(True)          # kèm thông tin từng từ (tuỳ chọn)

        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text_parts.append(result.get("text", ""))

        # Lấy phần cuối còn lại
        final = json.loads(rec.FinalResult())
        text_parts.append(final.get("text", ""))

    return chunk_idx, " ".join(t for t in text_parts if t)


# ===========================================================================
# Bước 4: Nhận dạng bằng SpeechRecognition (fallback / Google Web Speech)
# ===========================================================================

def transcribe_chunk_sr(chunk_path: str, language: str = "vi-VN") -> tuple[int, str]:
    """
    Fallback: Dùng SpeechRecognition + Google Web Speech API.
    Cần kết nối internet.
    """
    import speech_recognition as sr

    chunk_idx = int(Path(chunk_path).stem.split("_")[1])
    recognizer = sr.Recognizer()

    with sr.AudioFile(chunk_path) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio, language=language)
    except sr.UnknownValueError:
        text = ""
    except sr.RequestError as e:
        log.warning(f"Chunk {chunk_idx}: Google API lỗi — {e}")
        text = ""

    return chunk_idx, text





# ===========================================================================
# Lớp chính: VideoTranscriber
# ===========================================================================

class VideoTranscriber:
    """
    Service chuyển đổi video thành text.

    Parameters
    ----------
    model_path      : Đường dẫn thư mục model Vosk
                      (None = dùng fallback SpeechRecognition)
    language        : Ngôn ngữ cho SpeechRecognition (vd: "vi-VN", "en-US")
    chunk_duration  : Độ dài mỗi chunk tính bằng giây
    max_workers     : Số luồng song song xử lý chunks
    sample_rate     : Sample rate audio (Hz)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        language: str = "vi-VN",
        chunk_duration: int = 60,
        max_workers: int = 4,
        sample_rate: int = 16000,
        whisper_model_size: str = "base",
        whisper_model_instance = None,
    ):
        self.model_path     = model_path
        self.language       = language
        self.chunk_duration = chunk_duration
        self.max_workers    = max_workers
        self.sample_rate    = sample_rate
        self.whisper_model_size = whisper_model_size
        self._vosk_model    = None
        self._whisper_model_instance = whisper_model_instance
        self.key_status_callback = None

        if model_path:
            self._load_vosk_model(model_path)

    # ------------------------------------------------------------------
    def _load_vosk_model(self, model_path: str):
        try:
            from vosk import Model, SetLogLevel
            SetLogLevel(-1)  # Tắt log verbose của Vosk
            log.info(f"Đang load Vosk model từ: {model_path}")
            self._vosk_model = Model(model_path)
            log.info("Vosk model đã sẵn sàng ✓")
        except ImportError:
            log.warning("Thư viện vosk chưa cài. Chạy: pip install vosk")
            self._vosk_model = None
        except Exception as e:
            log.warning(f"Không load được Vosk model: {e}")
            self._vosk_model = None

    # ------------------------------------------------------------------
    def transcribe(self, video_path: str, output_path: Optional[str] = None, progress_callback = None, engine: str = "vosk", save_audio_path: Optional[str] = None, return_segments: bool = False) -> Union[str, dict]:
        """
        Chuyển đổi video thành text.

        Parameters
        ----------
        video_path  : Đường dẫn file video
        output_path : Nếu cung cấp, lưu kết quả vào file này
        progress_callback: Hàm callback nhận vào (done_chunks, total_chunks) để cập nhật tiến trình
        engine      : Công nghệ dịch ("vosk", "whisper", hoặc "google")
        save_audio_path: Nếu cung cấp, sao chép tệp âm thanh WAV trích xuất được vào đây
        return_segments: Nếu True, trả về dict gồm {"text": str, "segments": list[dict]} chứa timestamps

        Returns
        -------
        Toàn bộ text nhận dạng được (hoặc dict chứa text và segments)
        """
        video_path = str(Path(video_path).resolve())
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Không tìm thấy file: {video_path}")

        start_time = time.time()
        log.info(f"{'='*60}")
        log.info(f"Bắt đầu xử lý: {Path(video_path).name} bằng công nghệ: {engine.upper()}")
        log.info(f"{'='*60}")

        with tempfile.TemporaryDirectory(prefix="video2text_") as tmp_dir:
            # ── Bước 1: Trích xuất audio ──────────────────────────────
            wav_path   = os.path.join(tmp_dir, "audio.wav")
            extract_audio(video_path, wav_path, self.sample_rate)

            # Lấy thời lượng video
            duration = self._get_audio_duration(wav_path)
            log.info(f"Thời lượng audio: {duration/60:.1f} phút")

            # Sao lưu tệp âm thanh WAV nếu có chỉ định đường dẫn lưu
            if save_audio_path:
                try:
                    shutil.copy2(wav_path, save_audio_path)
                    log.info(f"Đã sao lưu audio tại: {save_audio_path}")
                except Exception as e:
                    log.warning(f"Không thể sao lưu file âm thanh WAV: {e}")

            # ── Bước 2 & 3: Chia nhỏ âm thanh thành chunks ─────────────
            chunk_dir  = os.path.join(tmp_dir, "chunks")
            chunk_paths = split_wav_into_chunks(wav_path, chunk_dir, self.chunk_duration)

            # ── Bước 4: Nhận dạng song song tùy theo engine ─────────────
            results = {}
            segments_list = []

            if engine.lower() == "groq":
                log.info("Dùng Chang Láo (Groq) — Tốc độ siêu tốc")
                api_keys = self._whisper_model_instance
                if not isinstance(api_keys, list):
                    api_keys = [api_keys] if api_keys else []
                
                # Kiểm tra dung lượng file WAV gốc
                wav_size = os.path.getsize(wav_path)
                max_size_limit = 24 * 1024 * 1024 # 24 MB
                
                if wav_size <= max_size_limit:
                    log.info(f"Dung lượng file WAV ({wav_size/1024/1024:.2f}MB) dưới 24MB. Gửi trực tiếp cả file.")
                    full_text, segments_list = self._transcribe_whisper_via_groq_api(wav_path, api_keys, progress_callback)
                else:
                    log.info(f"Dung lượng file WAV ({wav_size/1024/1024:.2f}MB) vượt quá 24MB. Tự động chuyển sang chế độ dịch theo phân đoạn (chunking) để tránh giới hạn API.")
                    
                    results = {}
                    active_keys = list(api_keys)
                    total_chunks = len(chunk_paths)
                    
                    for chunk_idx, cp in enumerate(chunk_paths):
                        log.info(f"Đang dịch chunk {chunk_idx + 1}/{total_chunks} qua Groq API...")
                        
                        # Khoảng nghỉ ngắn để tránh lỗi quá số lượt yêu cầu trong 1 phút (RPM Rate Limit) của Groq
                        if chunk_idx > 0:
                            time.sleep(1.2)
                            
                        try:
                            chunk_text, chunk_segs = self._transcribe_whisper_via_groq_api(
                                cp, 
                                active_keys, 
                                progress_callback=None
                            )
                            results[chunk_idx] = chunk_segs
                            
                            if progress_callback:
                                try:
                                    progress_callback(chunk_idx + 1, total_chunks)
                                except Exception as e:
                                    log.warning(f"Lỗi progress_callback: {e}")
                        except Exception as e:
                            log.error(f"Thất bại khi dịch chunk {chunk_idx + 1}: {e}")
                            raise e
                            
                    # Hợp nhất kết quả các chunk
                    merged_segments = []
                    for idx in sorted(results.keys()):
                        chunk_start = idx * self.chunk_duration
                        seg_list = results[idx]
                        for seg in seg_list:
                            abs_start = round(chunk_start + seg["start"], 2)
                            abs_end = round(chunk_start + seg["end"], 2)
                            
                            abs_words = []
                            for w in seg.get("words", []):
                                abs_words.append({
                                    "word": w["word"],
                                    "start": round(chunk_start + w["start"], 2),
                                    "end": round(chunk_start + w["end"], 2)
                                })
                                
                            text = seg["text"].strip()
                            if text:
                                if len(text) > 1:
                                    text = text[0].upper() + text[1:]
                                else:
                                    text = text.upper()
                                    
                                merged_segments.append({
                                    "start": abs_start,
                                    "end": abs_end,
                                    "text": text,
                                    "words": abs_words
                                })
                    full_text = "\n".join(seg["text"] for seg in merged_segments)
                    segments_list = merged_segments
            elif engine.lower() == "whisper":
                log.info("Dùng Whisper (offline) — Chạy tuần tự tối ưu hóa CPU")
                model = self._get_whisper_model()
                # Chạy dịch tuần tự trực tiếp trên file wav_path
                full_text, segments_list = self._transcribe_whisper_with_segments(wav_path, model, progress_callback)
            else:
                if engine.lower() == "vosk" and self._vosk_model:
                    log.info(f"Dùng Vosk (offline) — {len(chunk_paths)} chunks, {self.max_workers} luồng")
                    results = self._transcribe_parallel_vosk(chunk_paths, progress_callback)
                else:
                    log.info(f"Dùng SpeechRecognition (Google) — {len(chunk_paths)} chunks, {self.max_workers} luồng")
                    results = self._transcribe_parallel_sr(chunk_paths, progress_callback)
                
                full_text = self._merge_results(results)

                # Ước lượng mốc thời gian cho Vosk và Google dựa trên vị trí chunk
                segments_list = []
                for idx in sorted(results.keys()):
                    text_chunk = results[idx].strip()
                    if text_chunk:
                        formatted_chunk = self._format_by_sentences(text_chunk)
                        sentences = formatted_chunk.split("\n")
                        num_sentences = len(sentences)
                        chunk_start = idx * self.chunk_duration
                        if num_sentences > 0:
                            sec_per_sentence = self.chunk_duration / num_sentences
                            for s_idx, sentence in enumerate(sentences):
                                s_start = chunk_start + (s_idx * sec_per_sentence)
                                s_end = chunk_start + ((s_idx + 1) * sec_per_sentence)
                                segments_list.append({
                                    "start": round(s_start, 2),
                                    "end": round(s_end, 2),
                                    "text": sentence.strip()
                                })

        elapsed = time.time() - start_time

        if not (return_segments and engine.lower() == "whisper"):
            # Đối với Vosk, Google hoặc khi không cần segments, định dạng lại mỗi câu xuống một dòng
            full_text = self._format_by_sentences(full_text)

        word_count = len(full_text.split())
        log.info(f"{'='*60}")
        log.info(f"Hoàn thành trong {elapsed:.1f}s — {word_count} từ")
        log.info(f"{'='*60}")

        # ── Bước 5: Lưu file (tuỳ chọn) ──────────────────────────────
        if output_path:
            self._save_output(full_text, output_path, video_path, elapsed, word_count)

        if return_segments:
            return {"text": full_text, "segments": segments_list}
        return full_text

    # ------------------------------------------------------------------
    def _get_whisper_model(self):
        if not hasattr(self, "_whisper_model_instance") or self._whisper_model_instance is None:
            log.info(f"Đang khởi tạo local Whisper Model (kích thước {self.whisper_model_size})...")
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                raise ImportError("Vui lòng cài đặt: pip install faster-whisper")
            # Thiết lập num_workers = 1 vì CTranslate2 tự động song song hóa đa nhân CPU tối ưu nhất cho 1 worker
            self._whisper_model_instance = WhisperModel(
                self.whisper_model_size,
                device="cpu",
                compute_type="int8",
                num_workers=1
            )
        return self._whisper_model_instance

    # ------------------------------------------------------------------
    def _transcribe_whisper_with_segments(self, wav_path: str, model, progress_callback=None) -> tuple[str, list[dict]]:
        """Nhận dạng toàn bộ file WAV tuần tự bằng Whisper để tối ưu hóa CPU và độ chính xác."""
        log.info("Bắt đầu transcribe tuần tự bằng Whisper...")
        duration = self._get_audio_duration(wav_path)
        
        whisper_lang = self.language.split("-")[0]
        segments, info = model.transcribe(
            wav_path,
            beam_size=5,
            language=whisper_lang,
            word_timestamps=True
        )
        
        segments_list = []
        full_text_parts = []
        
        for segment in segments:
            words_list = []
            if segment.words:
                for w in segment.words:
                    words_list.append({
                        "word": w.word.strip(),
                        "start": round(w.start, 2),
                        "end": round(w.end, 2)
                    })
            
            text = segment.text.strip()
            if text:
                if len(text) > 1:
                    text = text[0].upper() + text[1:]
                else:
                    text = text.upper()
                
                seg_dict = {
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": text,
                    "words": words_list
                }
                segments_list.append(seg_dict)
                full_text_parts.append(text)
                
                if progress_callback:
                    try:
                        done_sec = min(round(segment.end, 2), round(duration, 2))
                        progress_callback(done_sec, round(duration, 2))
                    except Exception as e:
                        log.warning(f"Lỗi progress_callback: {e}")
        
        full_text = "\n".join(full_text_parts)
        return full_text, segments_list

    # ------------------------------------------------------------------
    def _transcribe_whisper_via_groq_api(self, wav_path: str, api_keys: list[str], progress_callback=None) -> tuple[str, list[dict]]:
        """
        Dịch audio bằng Groq Whisper API (mô hình whisper-large-v3) siêu tốc.
        Có cơ chế tự động xoay vòng API Keys khi gặp lỗi Rate Limit (HTTP 429).
        """
        import requests
        
        if not api_keys:
            raise ValueError("Không tìm thấy Groq API Key nào trong cấu hình. Hãy thêm GROQ_API_KEY trong file .env hoặc Settings Space.")
            
        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        log.info(f"Bắt đầu dịch qua Groq Whisper API với danh sách {len(api_keys)} keys...")
        
        # Mở file audio
        with open(wav_path, "rb") as f:
            files = {
                "file": (os.path.basename(wav_path), f, "audio/wav")
            }
            data = [
                ("model", "whisper-large-v3"),
                ("response_format", "verbose_json"),
                ("language", self.language.split("-")[0]),
                ("timestamp_granularities[]", "word"),
                ("timestamp_granularities[]", "segment"),
            ]
            
            last_error = None
            
            # Thử từng key trong danh sách xoay vòng (sử dụng bản sao để lặp, nhưng sửa đổi list gốc khi key lỗi)
            for key in list(api_keys):
                idx = api_keys.index(key) if key in api_keys else 0
                log.info(f"Đang thử sử dụng API Key thứ {idx + 1}...")
                headers = {
                    "Authorization": f"Bearer {key}"
                }
                
                try:
                    # Gửi yêu cầu API (timeout 60s phòng trường hợp file âm thanh lớn cần xử lý trên cloud)
                    response = requests.post(url, headers=headers, files=files, data=data, timeout=60)
                    
                    # Nếu gặp lỗi Rate Limit (429) hoặc lỗi xác thực/quota (400/401/403...)
                    if response.status_code == 429:
                        log.warning(f"API Key thứ {idx + 1} bị lỗi Rate Limit (HTTP 429 - Hết hạn mức). Đang chuyển sang key tiếp theo...")
                        last_error = "Rate Limit (HTTP 429)"
                        if getattr(self, "key_status_callback", None):
                            self.key_status_callback(key, "Rate Limited")
                        if key in api_keys:
                            api_keys.remove(key)
                        # Quay lại file pointer về đầu để gửi lại
                        f.seek(0)
                        continue
                        
                    if response.status_code != 200:
                        log.warning(f"API Key thứ {idx + 1} trả về lỗi HTTP {response.status_code}: {response.text}. Đang chuyển sang key tiếp theo...")
                        last_error = f"HTTP {response.status_code}: {response.text}"
                        if getattr(self, "key_status_callback", None):
                            self.key_status_callback(key, "Rate Limited")
                        if key in api_keys:
                            api_keys.remove(key)
                        f.seek(0)
                        continue
                        
                    # Thành công! Parse kết quả
                    result = response.json()
                    log.info(f"Dịch thành công bằng Groq API với Key thứ {idx + 1} ✓")
                    
                    # Tính toán thời lượng để trừ quota
                    duration = self._get_audio_duration(wav_path)
                    if getattr(self, "key_status_callback", None):
                        self.key_status_callback(key, "Active", duration)
                    
                    full_text = result.get("text", "").strip()
                    raw_segments = result.get("segments", [])
                    all_words = result.get("words", [])
                    
                    segments_list = []
                    for seg in raw_segments:
                        words_list = []
                        # 1. Lấy từ trường "words" bên trong segment nếu có
                        if "words" in seg and seg["words"]:
                            for w in seg["words"]:
                                words_list.append({
                                    "word": w.get("word", "").strip(),
                                    "start": round(w.get("start", 0), 2),
                                    "end": round(w.get("end", 0), 2)
                                })
                        # 2. Hoặc phân bổ từ danh sách "words" ở root JSON
                        elif all_words:
                            seg_start = seg.get("start", 0)
                            seg_end = seg.get("end", 0)
                            for w in all_words:
                                w_start = w.get("start", 0)
                                # Nếu thời gian của từ nằm trong khoảng của segment (cho phép lệch nhỏ 0.05s)
                                if seg_start - 0.05 <= w_start < seg_end + 0.05:
                                    words_list.append({
                                        "word": w.get("word", "").strip(),
                                        "start": round(w_start, 2),
                                        "end": round(w.get("end", 0), 2)
                                    })
                        
                        text = seg.get("text", "").strip()
                        if text:
                            if len(text) > 1:
                                text = text[0].upper() + text[1:]
                            else:
                                text = text.upper()
                                
                            segments_list.append({
                                "start": round(seg.get("start", 0), 2),
                                "end": round(seg.get("end", 0), 2),
                                "text": text,
                                "words": words_list
                            })
                            
                    if progress_callback:
                        duration = self._get_audio_duration(wav_path)
                        try:
                            progress_callback(round(duration, 2), round(duration, 2))
                        except Exception as e:
                            log.warning(f"Lỗi progress_callback: {e}")
                            
                    return full_text, segments_list
                    
                except requests.exceptions.RequestException as req_err:
                    log.warning(f"Lỗi kết nối khi gọi API với Key thứ {idx + 1}: {req_err}. Đang chuyển sang key tiếp theo...")
                    last_error = str(req_err)
                    if key in api_keys:
                        api_keys.remove(key)
                    f.seek(0)
                    continue
            
            raise RuntimeError(f"Tất cả các Groq API Keys trong danh sách đều thất bại. Chi tiết lỗi cuối cùng: {last_error}")

    # ------------------------------------------------------------------
    def _transcribe_parallel_vosk(self, chunk_paths: list[str], progress_callback = None) -> dict[int, str]:
        results = {}
        model   = self._vosk_model
        total   = len(chunk_paths)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(transcribe_chunk_vosk, cp, model): cp
                for cp in chunk_paths
            }
            for future in as_completed(futures):
                idx, text = future.result()
                results[idx] = text
                done = len(results)
                log.info(f"  [{done:3d}/{total}] chunk_{idx:04d} — {len(text.split())} từ")
                if progress_callback:
                    try:
                        progress_callback(done, total)
                    except Exception as e:
                        log.warning(f"Lỗi progress_callback: {e}")

        return results

    # ------------------------------------------------------------------
    def _transcribe_parallel_sr(self, chunk_paths: list[str], progress_callback = None) -> dict[int, str]:
        results = {}
        total   = len(chunk_paths)
        
        # Map "vi" to "vi-VN" and "en" to "en-US"
        google_lang = "vi-VN"
        if "en" in self.language.lower():
            google_lang = "en-US"
        elif "vi" in self.language.lower():
            google_lang = "vi-VN"
        else:
            google_lang = self.language

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(transcribe_chunk_sr, cp, google_lang): cp
                for cp in chunk_paths
            }
            for future in as_completed(futures):
                idx, text = future.result()
                results[idx] = text
                done = len(results)
                log.info(f"  [{done:3d}/{total}] chunk_{idx:04d} — {len(text.split())} từ")
                if progress_callback:
                    try:
                        progress_callback(done, total)
                    except Exception as e:
                        log.warning(f"Lỗi progress_callback: {e}")

        return results

    # ------------------------------------------------------------------
    @staticmethod
    def _merge_results(results: dict[int, str]) -> str:
        """Ghép các đoạn text theo thứ tự chunk index."""
        ordered = [results[k] for k in sorted(results.keys())]
        # Loại bỏ đoạn rỗng, ghép bằng khoảng trắng
        return " ".join(part for part in ordered if part.strip())

    # ------------------------------------------------------------------
    @staticmethod
    def _get_audio_duration(wav_path: str) -> float:
        """Trả về thời lượng file WAV tính bằng giây."""
        with wave.open(wav_path, "rb") as wf:
            return wf.getnframes() / wf.getframerate()

    # ------------------------------------------------------------------
    @staticmethod
    def _save_output(
        text: str,
        output_path: str,
        source_video: str,
        elapsed: float,
        word_count: int,
    ):
        """Lưu text và metadata ra file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# Transcript — {Path(source_video).name}\n")
            f.write(f"# Thời gian xử lý : {elapsed:.1f}s\n")
            f.write(f"# Số từ           : {word_count}\n")
            f.write(f"# Tạo lúc         : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            f.write(text)
        log.info(f"Đã lưu transcript tại: {output_path}")

    @staticmethod
    def _format_by_sentences(text: str) -> str:
        """
        Chia văn bản thành các câu riêng biệt, viết hoa chữ cái đầu và xuống dòng cho mỗi câu.
        """
        if not text:
            return ""
        
        import re
        # Tách câu theo các dấu kết thúc câu (. hoặc ? hoặc !), đồng thời giữ lại các dấu này
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        formatted_sentences = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            # Viết hoa chữ cái đầu tiên của câu
            if len(s) > 1:
                s = s[0].upper() + s[1:]
            else:
                s = s.upper()
            formatted_sentences.append(s)
            
        return "\n".join(formatted_sentences)


# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transcriber",
        description="Chuyển đổi video thành text (offline với Vosk hoặc qua Google Web Speech)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "video",
        help="Đường dẫn file video đầu vào (mp4, mkv, avi, mov, ...)",
    )
    p.add_argument(
        "-o", "--output",
        default=None,
        help="Đường dẫn file text đầu ra (mặc định: <tên_video>.txt)",
    )
    p.add_argument(
        "-m", "--model",
        default=None,
        help=(
            "Đường dẫn thư mục Vosk model (offline).\n"
            "Tải model tại: https://alphacephei.com/vosk/models\n"
            "  Tiếng Việt  : vosk-model-vn-0.4\n"
            "  Tiếng Anh   : vosk-model-en-us-0.22\n"
            "Nếu bỏ qua, dùng Google Web Speech API (cần internet)."
        ),
    )
    p.add_argument(
        "--lang",
        default="vi-VN",
        help="Ngôn ngữ cho Google Speech (vd: vi-VN, en-US). Mặc định: vi-VN",
    )
    p.add_argument(
        "--chunk",
        type=int,
        default=60,
        help="Độ dài mỗi chunk tính bằng giây. Mặc định: 60",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Số luồng xử lý song song. Mặc định: 4",
    )
    p.add_argument(
        "--rate",
        type=int,
        default=16000,
        help="Sample rate audio (Hz). Mặc định: 16000",
    )
    p.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Kích thước mô hình Whisper (mặc định: base)",
    )
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # Xác định file đầu ra
    output = args.output
    if output is None:
        stem   = Path(args.video).stem
        output = str(Path(args.video).parent / f"{stem}_transcript.txt")

    # Khởi tạo transcriber
    transcriber = VideoTranscriber(
        model_path    = args.model,
        language      = args.lang,
        chunk_duration= args.chunk,
        max_workers   = args.workers,
        sample_rate   = args.rate,
        whisper_model_size = args.whisper_model,
    )

    # Chạy
    text = transcriber.transcribe(args.video, output_path=output)

    # In ra màn hình (truncate nếu quá dài)
    print("\n" + "="*60)
    print("TRANSCRIPT (500 ký tự đầu):")
    print("="*60)
    print(text[:500])
    if len(text) > 500:
        print(f"\n... (còn {len(text)-500} ký tự nữa, xem file: {output})")


if __name__ == "__main__":
    main()
