import whisper
import json
import os
import numpy as np
import soundfile as sf
import webrtcvad
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering
from scipy.signal import medfilt
import sklearn

class AudioProcessor:
    def __init__(self, model_size="base", device="auto"):
        """
        Initialize Whisper and manual diarization models.
        """
        # --- Configurable Parameters (Internal) ---
        self.SR = 16000
        self.WINDOW = 0.5 
        self.HOP = 0.1 # Reduced hop for finer resolution
        self.MIN_TURN = 0.5
        self.SMOOTH_KERNEL = 5 # Increased smoothing
        self.OVERLAP_ENERGY_THRESHOLD = 0.005 # Lowered base threshold
        
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize Whisper
        print(f"Loading Whisper model: {model_size}...")
        self.model = whisper.load_model(model_size, device=self.device)
        print("Whisper model loaded.")
        
        # Initialize VAD
        self.vad = webrtcvad.Vad(2) # Mode 2: Aggressive
        
        # Initialize Encoder
        print("Loading VoiceEncoder...")
        self.encoder = VoiceEncoder(device=self.device)
        print("VoiceEncoder loaded.")

    def _vad_speech_ratio(self, audio, sr, vad, frame_ms=30):
        frame_len = int(sr * frame_ms / 1000)
        if len(audio) < frame_len:
            return 0.0

        speech_frames = 0
        total_frames = 0

        for i in range(0, len(audio) - frame_len, frame_len):
            frame = audio[i:i+frame_len]
            # Convert float32 to int16 bytes for webrtcvad
            pcm = (frame * 32767).astype(np.int16).tobytes()
            try:
                if vad.is_speech(pcm, sr):
                    speech_frames += 1
                total_frames += 1
            except:
                continue

        return speech_frames / max(total_frames, 1)

    def process_audio(self, file_path):
        """
        Process audio file: Transcribe & Diarize using manual VAD + Clustering implementation.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # ensure 16kHz mono generic fix (reusing previous logic roughly)
        try:
             # Try loading with soundfile to check sr/channels
             data, samplerate = sf.read(file_path)
             if samplerate != 16000 or (len(data.shape) > 1 and data.shape[1] > 1):
                  raise ValueError("Incorrect format")
        except Exception:
             print("Audio format mismatch (not 16kHz Mono). Converting...")
             from pydub import AudioSegment
             audio = AudioSegment.from_file(file_path)
             audio = audio.set_frame_rate(16000)
             audio = audio.set_channels(1)
             file_path = file_path.replace(".wav", "_fixed.wav").replace(".mp3", "_fixed.wav")
             if not file_path.endswith("_fixed.wav"): file_path += "_fixed.wav"
             audio.export(file_path, format="wav")
             print(f"Converted to {file_path}")

        # 1. Load & Preprocess
        print(f"Preprocessing {file_path}...")
        wav = preprocess_wav(file_path)
        duration = len(wav) / self.SR

        # 2. Sliding Window & VAD
        windows, times = [], []
        t = 0.0 
        while t + self.WINDOW <= duration:
            s_idx = int(t * self.SR)
            e_idx = int((t + self.WINDOW) * self.SR)
            chunk = wav[s_idx:e_idx]

            speech_ratio = self._vad_speech_ratio(chunk, self.SR, self.vad)
            if speech_ratio > 0.3: 
                windows.append(chunk)
                times.append((t, t + self.WINDOW))

            t += self.HOP
            
        if not windows:
            print("No speech detected by VAD.")
            # Fallback: Just transcribe without speakers
            whisper_result = self.model.transcribe(file_path)
            return [{"start": s["start"], "end": s["end"], "speaker": "Unknown", "text": s["text"]} for s in whisper_result["segments"]]

        # 3. Embedding
        print("Generating Embeddings...")
        embeddings = np.array([self.encoder.embed_utterance(w) for w in windows])

        # 4. Clustering (Spectral often works better for Speaker Diarization than Agglomerative)
        print("Clustering Speakers...")
        from sklearn.cluster import SpectralClustering
        try:
            cluster = SpectralClustering(n_clusters=2, affinity='cosine', assign_labels='discretize', random_state=42)
            labels = cluster.fit_predict(embeddings)
        except Exception:
            # Fallback to Agglomerative if Spectral fails (e.g. singular matrix)
            cluster = AgglomerativeClustering(n_clusters=2, metric="cosine", linkage="average")
            labels = cluster.fit_predict(embeddings)

        # 5. Smoothing
        labels = medfilt(labels, kernel_size=self.SMOOTH_KERNEL)

        # 6. Min Turn Logic
        final_labels = labels.copy()
        if len(labels) > 0:
            last_label = labels[0]
            last_time = times[0][0]

            for i in range(1, len(labels)):
                if labels[i] != last_label:
                    if times[i][0] - last_time < self.MIN_TURN:
                        final_labels[i] = last_label
                    else:
                        last_label = labels[i]
                        last_time = times[i][0]
                        
        # Identify Roles Heuristically
        # Assumption: Employee speaks first? Or Employee speaks MOST?
        # Let's try: Employee speaks first. If ambiguous, check total duration.
        # But usually in support calls, "Hello, this is X" comes first.
        
        roles = {0: "Speaker A", 1: "Speaker B"}
        if len(final_labels) > 0:
            first_speaker_id = final_labels[0]
            second_speaker_id = 1 - first_speaker_id
            roles[first_speaker_id] = "Employee"
            roles[second_speaker_id] = "Customer"
            print(f"Heuristic Role Map: {roles}")

        # 7. Transcription (Whisper)
        print(f"Transcribing {file_path}...")
        whisper_result = self.model.transcribe(file_path)

        # 8. Merge
        diarized_transcript = []
        
        for seg in whisper_result["segments"]:
            mid = (seg["start"] + seg["end"]) / 2
            
            # Find closest time window index
            if times:
                idx = min(range(len(times)), key=lambda i: abs((times[i][0] + times[i][1]) / 2 - mid))
                speaker_id = final_labels[idx]
                
                # Check energy for overlap check (only if requested, but let's be careful not to hide speech)
                # User asked to "remove overlap", which we define as low energy or conflicting segments?
                # Actually, simple clustering usually assigns 1 label. Real overlap is hard in mono.
                # The previous energy check might have filtered silent parts or bad segments.
                # Let's trust VAD/Clustering for now but filter very short/low energy if needed.
                # Re-using User's energy logic but ONLY for labeling "overlap" not hiding unless very low.
                
                energy = np.mean(windows[idx] ** 2)
                # Relaxed threshold to avoid cutting quiet valid speech
                if energy < 0.005: 
                     # Only skip if extremely quiet/silence
                     continue

                speaker = roles.get(speaker_id, f"Speaker {speaker_id}")
            else:
                speaker = "Unknown"

            diarized_transcript.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": speaker,
                "text": seg["text"].strip()
            })

        return diarized_transcript

    def export_to_json(self, transcript_data, output_path):
        with open(output_path, 'w') as f:
            json.dump(transcript_data, f, indent=4)
        return output_path
