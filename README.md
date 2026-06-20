# 🎙️ VoiceSwap AI

> **Ever wondered what your favorite song would sound like in your own voice?**

VoiceSwap AI is a full-stack Generative AI application that transforms any song into the user's voice. Simply upload a song, record your voice or upload a voice sample, and let AI recreate the track as if you were the one singing it.

---

## 🚀 Features

- 🎵 **Song Upload** – Upload any MP3/WAV song for voice transformation.
- 🎤 **Flexible Voice Input** – Record your voice live or upload a voice sample.
- 🧠 **AI Voice Conversion** – Recreates songs in the user's voice using RVC-based models.
- 🎧 **Vocal Separation** – Separates vocals and instrumentals using UVR/MDX models.
- ⚡ **Asynchronous Processing** – Executes AI tasks efficiently using FastAPI.
- 🎶 **Audio Reconstruction** – Combines converted vocals with the original instrumental to produce a natural-sounding output.

---

## 🧠 System Design

**Frontend:** React + Vite + TailwindCSS (interactive and responsive user interface)

**Backend:** FastAPI (handles uploads, processing, workflow orchestration, and asynchronous tasks)

**AI Pipeline:**
- **UVR / MDX** → Vocal separation
- **RVC-based Model** → Voice conversion
- **PyDub** → Audio merging & final export

---

## 🎧 How It Works

1. Upload a song (MP3/WAV).
2. Record your voice or upload a voice sample.
3. The AI separates vocals from the original track.
4. The extracted vocals are converted into your voice.
5. The converted vocals are merged with the instrumental.
6. Download your personalized version of the song.

---

## 💻 Run Locally

### 🔧 Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### 🎨 Frontend

```bash
cd frontend
npm install
npm run dev
```

Open your browser and visit:

```text
http://localhost:5173
```

---

## ✨ Key Highlights

- 🎙️ Transform songs into your own voice using Generative AI.
- ⚡ End-to-end asynchronous audio processing pipeline.
- 🎧 Real-time voice recording and voice sample upload support.
- 🏗️ Modular full-stack architecture built with React and FastAPI.
- 🤖 AI-powered vocal separation, voice conversion, and audio reconstruction.

---

## 🎯 Result

Generate a personalized version of any song in your own voice with just a few clicks.
