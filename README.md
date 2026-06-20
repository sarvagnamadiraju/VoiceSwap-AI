# 🎙️ VoiceSwap AI

> **Ever wondered what your favorite song would sound like in your own voice?**

VoiceSwap AI is a full-stack Generative AI application that lets users transform any song into their own voice. Simply upload a song, record your voice or upload a voice sample, and let AI recreate the track as if you were the one singing it.

---

## 🚀 Features

- 🎵 **Song Upload** – Upload any MP3/WAV song for voice transformation.
- 🎤 **Flexible Voice Input** – Record your voice live or upload a voice sample.
- 🧠 **AI Voice Conversion** – Recreates songs in the user's voice using RVC-based models.
- 🎧 **Vocal Separation** – Separates vocals and instrumentals using UVR/MDX models.
- ⚡ **Asynchronous Processing** – Executes AI tasks efficiently using FastAPI.
- 🎶 **Audio Reconstruction** – Combines converted vocals with the original instrumental to produce a natural-sounding output.

---

## 🧠 System Architecture

**Frontend**
- React
- Vite
- TailwindCSS

**Backend**
- FastAPI (Python)

**AI Pipeline**
- **UVR / MDX** → Vocal Separation
- **RVC** → Voice Conversion
- **PyDub** → Audio Mixing & Export

---

## 🎧 How It Works

1. 🎵 Upload a song (MP3/WAV).
2. 🎤 Record your voice or upload a voice sample.
3. 🧠 The AI separates the vocals from the music.
4. 🎶 The vocals are converted into your voice.
5. 📥 Download the generated song.

---

## 💻 Run Locally

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## ✨ Key Highlights

- Full-stack AI application built with React and FastAPI.
- End-to-end audio processing and voice conversion pipeline.
- Interactive voice recording and audio upload support.
- Modular architecture for scalable AI workflows.
- Modern, responsive user interface.

---

## 🛠️ Tech Stack

- **Frontend:** React, Vite, TailwindCSS
- **Backend:** FastAPI, Python
- **AI/ML:** PyTorch, RVC, UVR/MDX
- **Audio Processing:** PyDub

---

## 🎯 Output

Generate a personalized version of any song in your own voice with just a few clicks.
