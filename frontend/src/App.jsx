import React, { useState, useRef, useEffect } from 'react';
import { 
  Music, 
  UploadCloud, 
  Disc, 
  Download, 
  RefreshCw, 
  CheckCircle2, 
  AlertCircle, 
  Mic,
  Square,
  Volume2,
  Trash2
} from 'lucide-react';

export default function App() {
  const [backendUrl] = useState('http://127.0.0.1:8002');
  
  // Inputs
  const [songFile, setSongFile] = useState(null);
  const [userVoiceFile, setUserVoiceFile] = useState(null);
  
  // Recording State
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  // Studio Execution State: 'idle' | 'processing' | 'done'
  const [studioState, setStudioState] = useState('idle');
  const [progressStep, setProgressStep] = useState(0);
  const [errorMsg, setErrorMsg] = useState('');
  const [liveStep, setLiveStep] = useState('Starting...');
  const pollIntervalRef = useRef(null);
  
  // Outputs
  const [sessionId, setSessionId] = useState('');
  const [finalCoverUrl, setFinalCoverUrl] = useState('');

  // Status labels during processing
  const progressLabels = [
    { label: 'Audio Upload', desc: 'Uploading song and microphone recording...' },
    { label: 'Separating Stems', desc: 'Isolating original vocal track from background music using UVR5...' },
    { label: 'Zero-Shot AI Cloning', desc: 'Instantly transferring your voice timbre to the song...' },
    { label: 'Remixing & Mastering', desc: 'Blending your new vocals into the instrumental track...' }
  ];

  const handleSongSelect = (e) => {
    const file = e.target.files[0];
    if (file) setSongFile(file);
  };

  const handleVoiceUpload = (e) => {
    const file = e.target.files[0];
    if (file) setUserVoiceFile(file);
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const file = new File([audioBlob], 'microphone_recording.webm', { type: 'audio/webm' });
        setUserVoiceFile(file);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      
      // Auto-stop recording after 5 seconds to make the demo very fast!
      setTimeout(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
          mediaRecorderRef.current.stop();
          setIsRecording(false);
        }
      }, 5000);

    } catch (err) {
      setErrorMsg("Microphone access denied. Please allow microphone permissions or upload an audio file instead.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };



  const pollStatus = (sessionId) => {
    if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    pollIntervalRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${backendUrl}/api/status/${sessionId}`);
        if (!res.ok) return;
        const data = await res.json();

        if (data.step) setLiveStep(data.step);

        const step = (data.step || '').toLowerCase();
        if (step.includes('separating')) setProgressStep(1);
        else if (step.includes('generating custom')) setProgressStep(2);
        else if (step.includes('mixing')) setProgressStep(3);

        if (data.status === 'done' && data.result) {
          clearInterval(pollIntervalRef.current);
          setFinalCoverUrl(`${backendUrl}${data.result.mixed_audio_url}`);
          setStudioState('done');
        } else if (data.status === 'error') {
          clearInterval(pollIntervalRef.current);
          setErrorMsg(data.error || 'Something went wrong. Please try again.');
          setStudioState('idle');
        }
      } catch (err) {
        console.warn('Poll blip:', err);
      }
    }, 2000);
  };

  const handleCreateCover = async () => {
    if (!songFile) return setErrorMsg('Please upload a song file.');
    if (!userVoiceFile) return setErrorMsg('Please record your voice or upload an audio file.');

    setStudioState('processing');
    setProgressStep(0);
    setLiveStep('Uploading your files...');
    setErrorMsg('');

    const formData = new FormData();
    formData.append('song', songFile);
    formData.append('user_voice', userVoiceFile);

    try {
      const res = await fetch(`${backendUrl}/api/generate-cover`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error('Failed to start job.');

      const data = await res.json();
      setSessionId(data.session_id);
      pollStatus(data.session_id);

    } catch (err) {
      setErrorMsg(err.message || 'Could not reach the backend. Is the server running?');
      setStudioState('idle');
    }
  };

  const handleReset = () => {
    setSongFile(null);
    setUserVoiceFile(null);
    setFinalCoverUrl('');
    setStudioState('idle');
    setErrorMsg('');
  };

  return (
    <div className="min-h-screen flex flex-col font-sans antialiased text-slate-200">
      
      {/* Header */}
      <header className="glass-panel border-b border-white/5 py-4 px-6 sticky top-0 z-40 backdrop-blur-md">
        <div className="max-w-6xl mx-auto flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <div className="bg-indigo-600/30 p-2.5 rounded-2xl border border-indigo-500/40 text-indigo-400">
              <Disc className="w-6 h-6 animate-spin-slow" />
            </div>
            <div>
              <h1 className="text-lg font-bold bg-gradient-to-r from-white via-slate-200 to-indigo-400 bg-clip-text text-transparent">
                VoiceSwap AI
              </h1>
              <p className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider">Professional RVC Studio</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Studio Workspace */}
      <main className="flex-grow max-w-4xl w-full mx-auto p-4 md:p-8 flex flex-col justify-center">
        
        {errorMsg && (
          <div className="glass-panel border-red-500/20 bg-red-500/5 p-4 rounded-2xl flex items-start space-x-3 mb-6">
            <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
            <div>
              <h4 className="text-sm font-bold text-red-200">Error</h4>
              <p className="text-xs text-red-300 mt-1">{errorMsg}</p>
            </div>
          </div>
        )}

        {/* STATE: IDLE */}
        {studioState === 'idle' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              
              {/* Card 1: Original Song */}
              <div className="glass-panel rounded-3xl p-6 flex flex-col justify-between hover:border-indigo-500/20 transition-all">
                <div>
                  <h2 className="text-sm font-bold text-slate-100 flex items-center mb-4">
                    <span className="w-5 h-5 rounded-lg bg-indigo-500/10 text-indigo-400 flex items-center justify-center mr-2 text-xs font-bold border border-indigo-500/20">1</span>
                    Upload Original Song
                  </h2>
                  <p className="text-xs text-slate-400 leading-relaxed mb-6">
                    Upload the music track you want to convert.
                  </p>
                  <label className="flex flex-col items-center justify-center border-2 border-dashed border-white/5 hover:border-indigo-500/40 rounded-2xl p-6 bg-[#0f1422]/20 cursor-pointer">
                    <Music className="w-8 h-8 text-indigo-400 mb-2" />
                    <span className="text-xs text-slate-300 font-bold text-center">
                      {songFile ? songFile.name : 'Select Audio File'}
                    </span>
                    <input type="file" accept=".mp3,.wav" onChange={handleSongSelect} className="hidden" />
                  </label>
                </div>
              </div>

              {/* Card 2: User Voice */}
              <div className="glass-panel rounded-3xl p-6 flex flex-col justify-between hover:border-indigo-500/20 transition-all">
                <div>
                  <h2 className="text-sm font-bold text-slate-100 flex items-center mb-4">
                    <span className="w-5 h-5 rounded-lg bg-indigo-500/10 text-indigo-400 flex items-center justify-center mr-2 text-xs font-bold border border-indigo-500/20">2</span>
                    Your Voice (Target)
                  </h2>
                  <p className="text-xs text-slate-400 leading-relaxed mb-4">
                    Record your voice for 10 seconds or upload a clean voice recording.
                  </p>

                  {/* Mic Recording Controls */}
                  <div className="flex space-x-2 mb-4">
                    {!isRecording ? (
                      <button onClick={startRecording} className="flex-1 bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 text-red-400 py-3 rounded-xl flex items-center justify-center space-x-2 transition-all">
                        <Mic className="w-4 h-4" />
                        <span className="text-xs font-bold">Record Mic</span>
                      </button>
                    ) : (
                      <button onClick={stopRecording} className="flex-1 bg-red-500 hover:bg-red-600 border border-red-500 text-white py-3 rounded-xl flex items-center justify-center space-x-2 transition-all animate-pulse">
                        <Square className="w-4 h-4 fill-current" />
                        <span className="text-xs font-bold">Stop Recording</span>
                      </button>
                    )}
                  </div>                  <div className="relative flex items-center py-2">
                    <div className="flex-grow border-t border-white/5"></div>
                    <span className="flex-shrink-0 mx-4 text-[10px] text-slate-500 font-bold uppercase">OR</span>
                    <div className="flex-grow border-t border-white/5"></div>
                  </div>

                  <label className="flex flex-col items-center justify-center border-2 border-dashed border-white/5 hover:border-indigo-500/40 rounded-xl p-4 bg-[#0f1422]/20 cursor-pointer mt-2">
                    <UploadCloud className="w-5 h-5 text-indigo-400 mb-1" />
                    <span className="text-xs text-slate-300 text-center">
                      Upload Audio (.wav, .mp3)
                    </span>
                    <input type="file" accept=".wav,.mp3,.webm,.m4a" onChange={handleVoiceUpload} className="hidden" />
                  </label>

                  {userVoiceFile && (
                    <div className="mt-4 flex items-center justify-between bg-slate-900/60 p-2.5 rounded-xl border border-indigo-500/20">
                      <div className="flex items-center space-x-2 truncate">
                        <Mic className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                        <span className="text-xs text-slate-300 truncate max-w-[180px]">{userVoiceFile.name}</span>
                      </div>
                      <button onClick={() => setUserVoiceFile(null)} className="p-1 hover:text-red-400 text-slate-500 transition-colors">
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>

            <button
              onClick={handleCreateCover}
              disabled={!songFile || !userVoiceFile}
              className="w-full mt-4 bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 text-white font-bold py-4 px-6 rounded-2xl flex items-center justify-center space-x-2.5 shadow-xl disabled:opacity-50"
            >
              <Disc className="w-5 h-5 animate-spin-slow" />
              <span className="text-sm tracking-wide">Generate Song in My Voice</span>
            </button>
          </div>
        )}

        {/* STATE: PROCESSING */}
        {studioState === 'processing' && (
          <div className="max-w-md w-full mx-auto glass-panel rounded-3xl p-6 text-center my-12 relative overflow-hidden">
            <h3 className="text-lg font-bold text-slate-100">Cloning Voice...</h3>
            <p className="text-xs text-slate-400 mt-1">Applying Zero-Shot Timbre Transfer</p>
            
            <div className="mt-8 space-y-4 text-left">
              {progressLabels.map((step, idx) => {
                const isActive = progressStep === idx;
                const isCompleted = progressStep > idx;
                return (
                  <div key={idx} className={`flex items-start space-x-3 p-3 rounded-xl transition-all ${isActive ? 'bg-indigo-500/10 border border-indigo-500/20' : ''}`}>
                    <div className="mt-0.5 flex-shrink-0">
                      {isCompleted ? <CheckCircle2 className="w-5 h-5 text-indigo-400" /> : 
                       isActive ? <div className="w-5 h-5 rounded-full border-2 border-indigo-500/50 border-t-indigo-500 animate-spin"></div> : 
                       <div className="w-5 h-5 rounded-full border-2 border-white/5"></div>}
                    </div>
                    <div>
                      <h4 className="text-xs font-bold text-slate-200">{step.label}</h4>
                      <p className="text-[10px] text-slate-400 mt-0.5">{step.desc}</p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* STATE: DONE */}
        {studioState === 'done' && (
          <div className="glass-panel rounded-3xl p-6 md:p-8 max-w-xl mx-auto w-full text-center space-y-6">
            <CheckCircle2 className="w-12 h-12 text-emerald-400 mx-auto" />
            <h2 className="text-xl font-bold text-white">Your Custom AI Song is Ready!</h2>
            
            <div className="bg-[#0f1422] p-5 rounded-2xl border border-white/5 space-y-4">
              <audio src={finalCoverUrl} controls className="w-full h-10 bg-slate-900 rounded-lg" />
              <a href={finalCoverUrl} download="My_AI_Cover.mp3" className="w-full py-3.5 px-4 bg-emerald-600 hover:bg-emerald-500 text-white font-bold rounded-xl text-sm flex items-center justify-center space-x-2">
                <Download className="w-4 h-4" />
                <span>Download MP3</span>
              </a>
            </div>

            <button onClick={handleReset} className="w-full text-slate-400 hover:text-white text-xs font-bold flex items-center justify-center space-x-1.5 mt-4">
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Make Another Song</span>
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
