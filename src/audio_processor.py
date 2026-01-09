import whisper
import json
import os
import senko 

# Audio Processor Module

class AudioProcessor:
    def __init__(self, model_size="base", device="auto"):
        """
        Initialize Whisper and Senko models.
        """
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
        
        # Initialize Senko
        print("Loading Senko model...")
        self.diarizer = senko.Diarizer(device=self.device, warmup=True, quiet=False)
        print("Senko model loaded.")

    def process_audio(self, file_path):
        """
        Process audio file: Transcribe & Diarize using user's specific logic.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # Check/Convert Audio if needed
        # Some WAVs miss RIFF headers or are raw PCM that Whisper/Senko dislikes.
        # We try to load it with pydub and export as a clean WAV.
        try:
             # Basic check: verify file header or try lightweight load
             # But easiest is just to wrap the critical logic below in try-except and convert if it fails
             pass
        except:
             pass

        # 1. Diarization (Senko)
        print(f"Diarizing {file_path}...")
        try:
             dia_result = self.diarizer.diarize(file_path, generate_colors=False)
        except Exception as e:
             # Check for RIFF header error OR the specific 16kHz format error
             err_msg = str(e).lower()
             if "riff" in err_msg or "header" in err_msg or "correct format" in err_msg or "16khz" in err_msg:
                  print("Audio format issue (Header or 16kHz/Mono mismatch) detected. Converting...")
                  from pydub import AudioSegment
                  
                  # Load and enforce 16kHz Mono
                  audio = AudioSegment.from_file(file_path)
                  audio = audio.set_frame_rate(16000)
                  audio = audio.set_channels(1)
                  
                  file_path = file_path.replace(".wav", "_fixed.wav").replace(".mp3", "_fixed.wav")
                  if not file_path.endswith("_fixed.wav"):
                       file_path += "_fixed.wav"
                       
                  audio.export(file_path, format="wav")
                  print(f"Converted to {file_path}. Retrying...")
                  
                  # Retry with fixed file
                  dia_result = self.diarizer.diarize(file_path, generate_colors=False)
             else:
                  raise e
        
        senko_segments = dia_result["merged_segments"] # User's code key

        # 2. Transcription (Whisper)
        print(f"Transcribing {file_path}...")
        whisper_result = self.model.transcribe(file_path)

        # 3. Merge (User's Midpoint Logic)
        diarized_transcript = []

        for seg in whisper_result["segments"]:
            mid_time = (seg["start"] + seg["end"]) / 2

            # Find speaker
            speaker_label = "Unknown"
            for s in senko_segments:
                # User logic: check if midpoint is within senko segment
                if s["start"] <= mid_time <= s["end"]:
                    speaker_label = s["speaker"]
                    break 

            diarized_transcript.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": speaker_label,
                "text": seg["text"].strip()
            })

        return diarized_transcript

    def export_to_json(self, transcript_data, output_path):
        with open(output_path, 'w') as f:
            json.dump(transcript_data, f, indent=4)
        return output_path