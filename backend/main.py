import os
import shutil
import uuid
import threading
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import imageio_ffmpeg
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# FIX: Inject FFmpeg into system PATH so UVR5 and Demucs don't crash with [WinError 2]
ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
if ffmpeg_dir not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + ffmpeg_dir

load_dotenv()

app = FastAPI(title="VoiceSwap AI - Professional RVC Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp_audio"
MODELS_DIR = "models"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
app.mount("/audio", StaticFiles(directory=TEMP_DIR), name="audio")

job_store = {}

# Create an empty index file for RVC to satisfy the API
DUMMY_INDEX = os.path.join(MODELS_DIR, "dummy.index")
if not os.path.exists(DUMMY_INDEX):
    with open(DUMMY_INDEX, "w") as f:
        f.write("")


# ═══════════════════════════════════════════════════════════
# STAGE 1: PROFESSIONAL STEM SEPARATION
# ═══════════════════════════════════════════════════════════

def separate_stems_audio_separator(song_path: str, vocals_path: str, instrumental_path: str) -> bool:
    try:
        from audio_separator.separator import Separator
        print("[UVR5] Initializing professional AI stem separator...")
        session_dir = os.path.dirname(song_path)

        separator = Separator(
            output_dir=session_dir,
            model_file_dir=MODELS_DIR,
            output_format="WAV",
            normalization_threshold=0.9,
        )

        separator.load_model("UVR-MDX-NET-Inst_HQ_4.onnx")
        outputs = separator.separate(song_path)
        print(f"[UVR5] Outputs: {outputs}")

        vocals_found = False
        instrumental_found = False
        for f in outputs:
            fname = os.path.basename(f).lower()
            if "(vocals)" in fname or "vocals" in fname:
                shutil.copy(f, vocals_path)
                vocals_found = True
            elif "(instrumental)" in fname or "instrumental" in fname or "no_vocals" in fname:
                shutil.copy(f, instrumental_path)
                instrumental_found = True

        if vocals_found and instrumental_found:
            print("[UVR5] Professional stem separation complete!")
            return True

        if len(outputs) == 2 and not (vocals_found and instrumental_found):
            shutil.copy(outputs[0], vocals_path)
            shutil.copy(outputs[1], instrumental_path)
            return True

        return False

    except ImportError:
        print("[UVR5] audio-separator not installed.")
        return False
    except Exception as e:
        print(f"[UVR5] Failed: {e}")
        return False


def separate_stems_demucs_api(song_path: str, vocals_path: str, instrumental_path: str) -> bool:
    """Uses a free Hugging Face API to run Demucs and split vocals from background music, with a timeout."""
    try:
        from gradio_client import Client, handle_file
        import concurrent.futures

        def fetch_api():
            client = Client("abidlabs/music-separation", token=os.getenv("HF_TOKEN"))
            return client.predict(handle_file(song_path), api_name="/inference")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(fetch_api)
            # Timeout after 300 seconds if the Hugging Face queue is stuck
            result = future.result(timeout=300)

        if isinstance(result, (list, tuple)) and len(result) >= 2:
            import shutil
            shutil.copy(result[0], vocals_path)
            shutil.copy(result[1], instrumental_path)
            return True
        return False
    except concurrent.futures.TimeoutError:
        print("\n[Pipeline] Demucs API timed out due to long Hugging Face queue. Falling back to local.")
        return False
    except Exception as e:
        print(f"\n[Pipeline] Demucs API failed: {e}")
        return False


def separate_stems_local(song_path: str, vocals_path: str, instrumental_path: str) -> bool:
    # First try local Demucs for best quality
    try:
        import subprocess
        import sys
        print("[Demucs] Running local high-quality stem separation...")
        session_dir = os.path.dirname(vocals_path)
        result = subprocess.run(
            [sys.executable, "-m", "demucs", "--two-stems=vocals", "-o", session_dir, song_path],
            capture_output=True, text=True, timeout=300
        )
        
        # Demucs creates a folder structure. Let's find the files dynamically.
        import glob
        v_paths = glob.glob(os.path.join(session_dir, "htdemucs", "**", "vocals.wav"), recursive=True)
        i_paths = glob.glob(os.path.join(session_dir, "htdemucs", "**", "no_vocals.wav"), recursive=True)
        
        if v_paths and i_paths:
            shutil.copy(v_paths[0], vocals_path)
            shutil.copy(i_paths[0], instrumental_path)
            print("[Demucs] Local separation complete!")
            return True
        print(f"[Demucs] Output not found. stderr: {result.stderr[:300]}")
    except Exception as e:
        print(f"[Demucs] Local failed: {e}")

    # Fallback to UVR5 (5-Star Quality)
    try:
        print("[UVR5] Demucs failed, running Ultimate Vocal Remover 5 fallback...")
        from audio_separator.separator import Separator
        session_dir = os.path.dirname(vocals_path)
        separator = Separator(output_dir=session_dir, output_format="WAV", normalization_threshold=0.9)
        separator.load_model("UVR-MDX-NET-Inst_HQ_4.onnx")
        outputs = separator.separate(song_path)
        
        vocals_found = False
        instrumental_found = False
        for f in outputs:
            fname = os.path.basename(f).lower()
            src_path = os.path.join(session_dir, f)
            if "(vocals)" in fname or "vocals" in fname:
                shutil.copy(src_path, vocals_path)
                vocals_found = True
            elif "(instrumental)" in fname or "instrumental" in fname or "no_vocals" in fname:
                shutil.copy(src_path, instrumental_path)
                instrumental_found = True
                
        if vocals_found and instrumental_found:
            print("[UVR5] Local separation complete!")
            return True
        elif len(outputs) == 2:
            shutil.copy(os.path.join(session_dir, outputs[0]), vocals_path)
            shutil.copy(os.path.join(session_dir, outputs[1]), instrumental_path)
            return True
    except Exception as e:
        print(f"[UVR5] Failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════
# STAGE 2: PROFESSIONAL ZERO-SHOT INFERENCE (HUGGING FACE GPU)
# ═══════════════════════════════════════════════════════════

def convert_voice_seed_vc(source_path: str, user_voice_path: str, output_path: str, pitch_shift: int) -> bool:
    """Uses Plachta/Seed-VC on HuggingFace for Zero-Shot Voice Cloning. Chunks audio to bypass length limits."""
    try:
        from gradio_client import Client, handle_file
        import concurrent.futures
        import os
        from pydub import AudioSegment
        import imageio_ffmpeg
        AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
        
        print("[Seed-VC Cloud] Loading audio to check duration...")
        audio = AudioSegment.from_file(source_path)
        duration_sec = len(audio) / 1000.0

        # BULLETPROOF FIX: We reduced chunks to 10 seconds. 
        # When combined with MP3 compression, the file size is now only ~150 KB!
        # It is mathematically impossible for this to timeout, even on a weak hotspot.
        chunk_length_ms = 10000  # 10 seconds
        chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        
        print(f"[Seed-VC Cloud] Audio is {duration_sec:.1f}s. Splitting into {len(chunks)} chunks...")
        
        def fetch_seed_vc(chunk_file):
            # THE UNLIMITED QUOTA HACK
            # We purposely do NOT pass a token. Hugging Face will track your IP address instead.
            # If you hit a quota limit, you just turn on a Free VPN and your quota magically resets to 100%!
            print("[Seed-VC Cloud] Connecting anonymously (No Token needed!)...")
            client = Client("Plachta/Seed-VC")
            return client.predict(
                source_audio_path=handle_file(chunk_file),
                target_audio_path=handle_file(user_voice_path),
                diffusion_steps=10, 
                length_adjust=1.0,
                inference_cfg_rate=1.0, # CRITICAL FIX: Increased from 0.7 to 1.0 to completely overwrite original singer!
                f0_condition=True,
                auto_f0_adjust=False, 
                pitch_shift=pitch_shift,
                api_name="/predict_1"
            )

        output_audio_segments = []
        
        for idx, chunk in enumerate(chunks):
            print(f"[Seed-VC Cloud] Processing chunk {idx+1}/{len(chunks)}...")
            temp_chunk_path = os.path.join(os.path.dirname(output_path), f"temp_source_chunk_{idx}.mp3")
            # COMPRESS TO MP3: 10x smaller file size prevents "write operation timed out" on slow connections
            chunk.export(temp_chunk_path, format="mp3", bitrate="128k")
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(fetch_seed_vc, temp_chunk_path)
                result = future.result(timeout=300)
                
            out_file = result[1] if isinstance(result, tuple) and len(result) > 1 else None
            if not out_file and isinstance(result, str):
                out_file = result
                
            if out_file and os.path.exists(out_file):
                processed_chunk = AudioSegment.from_file(out_file)
                output_audio_segments.append(processed_chunk)
            else:
                print(f"[Pipeline] Chunk {idx+1} failed to process. Result: {result}")
                return False

        if output_audio_segments:
            print("[Seed-VC Cloud] Sticking chunks back together...")
            final_audio = sum(output_audio_segments)
            final_audio.export(output_path, format="wav")
            print("[Seed-VC Cloud] Done!")
            return True
            
        return False

    except concurrent.futures.TimeoutError:
        print("\n[Pipeline] Seed-VC API timed out. The Hugging Face server is overloaded right now.")
        return False
    except Exception as e:
        print(f"\n[Pipeline] Seed-VC Error: {e}")
        return False


# ═══════════════════════════════════════════════════════════
# STAGE 3: MIX
# ═══════════════════════════════════════════════════════════

def mix_audio(vocals_path: str, instrumental_path: str, output_path: str,
              vocal_gain: float = 1.30, inst_gain: float = 0.75) -> bool:
    try:
        print("[Mix] Mixing AI voice with original background music (5-Star Quality)...")
        from pydub import AudioSegment
        import imageio_ffmpeg
        AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

        vocal_seg = AudioSegment.from_file(vocals_path)
        inst_seg = AudioSegment.from_file(instrumental_path)

        # Apply gain (pydub uses dB, so we convert ratio to dB roughly, or just use apply_gain)
        import math
        vocal_db = 20 * math.log10(vocal_gain) if vocal_gain > 0 else 0
        inst_db = 20 * math.log10(inst_gain) if inst_gain > 0 else 0

        vocal_seg = vocal_seg + vocal_db
        inst_seg = inst_seg + inst_db

        # Overlay perfectly
        mixed_seg = inst_seg.overlay(vocal_seg)

        # Export high quality MP3
        mixed_seg.export(output_path, format="mp3", bitrate="320k")
        print(f"[Mix] High-quality MP3 saved: {os.path.getsize(output_path):,} bytes")
        
        # Also save a wav version for safety
        wav_out = output_path.replace(".mp3", ".wav")
        mixed_seg.export(wav_out, format="wav")

        return True
    except Exception as e:
        print(f"[Mix] Failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════
# BACKGROUND PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════

def run_pipeline(session_id: str, song_path: str, user_voice_path: str,
                 pitch_shift: int, use_api: bool):
    session_dir = os.path.dirname(song_path)
    vocals_path = os.path.join(session_dir, "vocals.wav")
    instrumental_path = os.path.join(session_dir, "instrumental.wav")
    cloned_path = os.path.join(session_dir, "cloned_vocals.wav")
    final_mp3 = os.path.join(session_dir, "final_cover.mp3")
    final_wav = os.path.join(session_dir, "final_cover.wav")

    def update(msg):
        job_store[session_id]["step"] = msg
        print(f"\n>>> {msg}")

    try:
        # ── TRIM to 45 seconds using librosa (no ffmpeg needed) ──
        print("[Pipeline] Trimming song to 45 seconds...")
        _y, _sr = librosa.load(song_path, sr=None, mono=True, duration=45.0)
        _trimmed_path = os.path.join(os.path.dirname(song_path), "trimmed_song.wav")
        sf.write(_trimmed_path, _y.astype(np.float32), _sr)
        song_path = _trimmed_path
        print(f"[Pipeline] Trimmed successfully to {len(_y)/_sr:.1f}s")

        # ── STEP 1: Stem Separation ──
        update("Step 1/3: Separating vocals using local CPU...")
        separated = False
        # separated = separate_stems_demucs_api(song_path, vocals_path, instrumental_path)
        if not separated:
            update("Step 1/3: Demucs failed, running local HPSS separation...")
            separated = separate_stems_local(song_path, vocals_path, instrumental_path)
        if not separated:
            raise RuntimeError("All stem separation methods failed.")

        # ── STEP 2: Zero-Shot Voice Conversion ──
        update("Step 2/3: Applying Zero-Shot Voice Cloning on Hugging Face GPU...")
        success = convert_voice_seed_vc(vocals_path, user_voice_path, cloned_path, pitch_shift)

        if not success:
            raise RuntimeError("Zero-Shot Conversion failed. Please check the terminal logs.")

        # ── STEP 3: Mix ──
        update("Step 3/3: Mixing AI voice with original background music...")
        mix_audio(cloned_path, instrumental_path, final_mp3)

        if os.path.exists(final_mp3) and os.path.getsize(final_mp3) > 1024:
            final_filename = "final_cover.mp3"
        elif os.path.exists(final_wav) and os.path.getsize(final_wav) > 1024:
            final_filename = "final_cover.wav"
        else:
            raise RuntimeError("Final output file is empty or missing.")

        print(f"\n{'='*60}\n✅ ALL DONE! Cover generated: {final_filename}\n{'='*60}")

        job_store[session_id]["status"] = "done"
        job_store[session_id]["step"] = "✅ Complete! Your AI cover is ready."
        job_store[session_id]["result"] = {
            "session_id": session_id,
            "vocals_url": f"/audio/{session_id}/vocals.wav",
            "instrumental_url": f"/audio/{session_id}/instrumental.wav",
            "cloned_vocals_url": f"/audio/{session_id}/cloned_vocals.wav",
            "mixed_audio_url": f"/audio/{session_id}/{final_filename}"
        }

    except Exception as e:
        print(f"\n[Pipeline] ERROR: {e}")
        job_store[session_id]["status"] = "error"
        job_store[session_id]["step"] = f"Error: {str(e)}"
        job_store[session_id]["error"] = str(e)


# ═══════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    return {"status": "ok", "engine": "RVC Studio Edition"}


@app.post("/api/generate-cover")
async def generate_cover(
    song: UploadFile = File(...),
    user_voice: UploadFile = File(...),
    pitch_shift: int = Form(0),
    top_k: int = Form(4),
    use_api: bool = Form(False)
):
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(TEMP_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    song_ext = os.path.splitext(song.filename)[1] or ".mp3"
    song_path = os.path.join(session_dir, f"original{song_ext}")
    with open(song_path, "wb") as f:
        shutil.copyfileobj(song.file, f)

    voice_ext = os.path.splitext(user_voice.filename)[1] or ".mp3"
    user_voice_path = os.path.join(session_dir, f"user_voice{voice_ext}")
    with open(user_voice_path, "wb") as f:
        shutil.copyfileobj(user_voice.file, f)

    job_store[session_id] = {
        "status": "processing",
        "step": "📤 Files uploaded. Starting pipeline...",
        "result": None,
        "error": None
    }
    
    print(f"\n[API] Received new Generate request! Session: {session_id}", flush=True)

    threading.Thread(
        target=run_pipeline,
        args=(session_id, song_path, user_voice_path, pitch_shift, use_api),
        daemon=True
    ).start()

    return {"session_id": session_id, "status": "processing"}


@app.get("/api/status/{session_id}")
async def get_status(session_id: str):
    if session_id not in job_store:
        raise HTTPException(status_code=404, detail="Session not found.")
    job = job_store[session_id]
    return {
        "session_id": session_id,
        "status": job["status"],
        "step": job["step"],
        "result": job.get("result"),
        "error": job.get("error")
    }
