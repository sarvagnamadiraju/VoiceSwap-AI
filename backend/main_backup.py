import os
import shutil
import uuid
import threading
import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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
    try:
        from gradio_client import Client, handle_file
        print("[Demucs API] Sending to HuggingFace cloud GPU...")
        client = Client("abidlabs/music-separation")
        result = client.predict(handle_file(song_path), api_name="/predict")
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            shutil.copy(result[0], vocals_path)
            shutil.copy(result[1], instrumental_path)
            print("[Demucs API] Done!")
            return True
    except Exception as e:
        print(f"[Demucs API] Failed: {e}")
    return False


def separate_stems_local(song_path: str, vocals_path: str, instrumental_path: str) -> bool:
    try:
        print("[HPSS] Running local fallback separation...")
        y, sr = librosa.load(song_path, sr=None, mono=True)
        harmonic, percussive = librosa.effects.hpss(y, margin=8)
        sf.write(vocals_path, harmonic.astype(np.float32), sr)
        sf.write(instrumental_path, percussive.astype(np.float32), sr)
        return True
    except Exception as e:
        print(f"[HPSS] Failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════
# STAGE 2: PROFESSIONAL RVC INFERENCE (HUGGING FACE GPU)
# ═══════════════════════════════════════════════════════════

def convert_voice_rvc(source_path: str, rvc_model_path: str, output_path: str, pitch_shift: int) -> bool:
    """Uses r3gm/rvc_zero space with actual .pth model file"""
    try:
        from gradio_client import Client, handle_file
        print("[RVC Cloud] Connecting to HuggingFace r3gm/rvc_zero space...")
        client = Client("r3gm/rvc_zero")

        print(f"[RVC Cloud] Sending source: {source_path}")
        print(f"[RVC Cloud] Sending model: {rvc_model_path}")
        
        result = client.predict(
            [handle_file(source_path)],          # audio_files: list
            handle_file(rvc_model_path),         # file_m: RVC model (.pth)
            "rmvpe+",                            # pitch_alg
            pitch_shift,                         # pitch_lvl
            handle_file(DUMMY_INDEX),            # file_index (dummy)
            0.75,                                # index_inf
            3,                                   # r_m_f
            0.25,                                # e_r
            0.5,                                 # c_b_p
            False,                               # active_noise_reduce
            False,                               # audio_effects
            "wav",                               # type_output
            1,                                   # steps
            api_name="/run"
        )

        # Result is typically a list of output filepaths
        out_file = result if isinstance(result, str) else (result[0] if isinstance(result, (list, tuple)) else None)
        
        if out_file and os.path.exists(str(out_file)):
            shutil.copy(str(out_file), output_path)
            print("[RVC Cloud] Voice conversion SUCCESS!")
            return True
        else:
            print(f"[RVC Cloud] Failed to find output file in result: {result}")
            return False

    except Exception as e:
        print(f"[RVC Cloud] Error during inference: {e}")
        return False


# ═══════════════════════════════════════════════════════════
# STAGE 3: MIX
# ═══════════════════════════════════════════════════════════

def mix_audio(vocals_path: str, instrumental_path: str, output_path: str,
              vocal_gain: float = 1.15, inst_gain: float = 0.82) -> bool:
    try:
        print("[Mix] Mixing AI voice with original background music...")
        v, v_sr = sf.read(vocals_path)
        i, i_sr = sf.read(instrumental_path)

        if v_sr != i_sr:
            i = librosa.resample(i.T if len(i.shape) > 1 else i, orig_sr=i_sr, target_sr=v_sr)
            if len(i.shape) > 1:
                i = i.T
            i_sr = v_sr

        if len(v.shape) > 1: v = v.mean(axis=1)
        if len(i.shape) > 1: i = i.mean(axis=1)

        n = max(len(v), len(i))
        vp = np.zeros(n); ip = np.zeros(n)
        vp[:len(v)] = v * vocal_gain
        ip[:len(i)] = i * inst_gain

        mixed = vp + ip
        peak = np.max(np.abs(mixed))
        if peak > 1.0:
            mixed = mixed / peak * 0.95

        wav_out = output_path.replace(".mp3", ".wav")
        sf.write(wav_out, mixed.astype(np.float32), v_sr)
        print(f"[Mix] WAV saved: {os.path.getsize(wav_out):,} bytes")

        try:
            from pydub import AudioSegment
            AudioSegment.from_wav(wav_out).export(output_path, format="mp3", bitrate="192k")
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
                os.remove(wav_out)
                print(f"[Mix] MP3 saved: {os.path.getsize(output_path):,} bytes")
            else:
                if os.path.exists(output_path): os.remove(output_path)
                print("[Mix] MP3 failed, using WAV.")
        except Exception as e:
            print(f"[Mix] MP3 skipped ({e}), using WAV.")

        return True
    except Exception as e:
        print(f"[Mix] Failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════
# BACKGROUND PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════

def run_pipeline(session_id: str, song_path: str, rvc_model_path: str,
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
        # ── STEP 1: Stem Separation ──
        update("🎵 Step 1/3: Separating vocals using UVR5 AI model...")
        separated = False
        separated = separate_stems_audio_separator(song_path, vocals_path, instrumental_path)
        if not separated:
            update("🎵 Step 1/3: Trying cloud Demucs separation...")
            separated = separate_stems_demucs_api(song_path, vocals_path, instrumental_path)
        if not separated:
            update("🎵 Step 1/3: Running local HPSS separation...")
            separated = separate_stems_local(song_path, vocals_path, instrumental_path)
        if not separated:
            raise RuntimeError("All stem separation methods failed.")

        # Note: We do NOT pitch shift the vocals locally here anymore!
        # We pass pitch_shift directly to the RVC model in Step 2, which handles it much better.

        # ── STEP 2: RVC Cloud Voice Conversion ──
        update("🎤 Step 2/3: Applying RVC Voice Model on Hugging Face GPU...")
        success = convert_voice_rvc(vocals_path, rvc_model_path, cloned_path, pitch_shift)

        if not success:
            raise RuntimeError("RVC Cloud Conversion failed. Please check the terminal logs.")

        # ── STEP 3: Mix ──
        update("🎚️ Step 3/3: Mixing AI voice with original background music...")
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
        print(f"\n[Pipeline] ❌ ERROR: {e}")
        job_store[session_id]["status"] = "error"
        job_store[session_id]["step"] = f"❌ Error: {str(e)}"
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
    rvc_model: UploadFile = File(...),
    pitch_shift: int = Form(0),
    top_k: int = Form(4),
    use_api: bool = Form(False)
):
    if not rvc_model.filename.endswith('.pth'):
        raise HTTPException(status_code=400, detail="RVC Voice Model must be a .pth file.")

    session_id = str(uuid.uuid4())
    session_dir = os.path.join(TEMP_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    song_ext = os.path.splitext(song.filename)[1] or ".mp3"
    song_path = os.path.join(session_dir, f"original{song_ext}")
    with open(song_path, "wb") as f:
        shutil.copyfileobj(song.file, f)

    model_path = os.path.join(session_dir, "model.pth")
    with open(model_path, "wb") as f:
        shutil.copyfileobj(rvc_model.file, f)

    job_store[session_id] = {
        "status": "processing",
        "step": "📤 Files uploaded. Starting pipeline...",
        "result": None,
        "error": None
    }

    threading.Thread(
        target=run_pipeline,
        args=(session_id, song_path, model_path, pitch_shift, use_api),
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
