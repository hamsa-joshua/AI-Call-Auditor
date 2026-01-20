from faster_whisper import WhisperModel
import json
import os

class AudioProcessor:
    def __init__(self, model_size="base", device="auto"):
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")
        # faster-whisper uses 'cuda' or 'cpu', matching our logic
        self.model = WhisperModel(model_size, device=self.device, compute_type="int8") # int8 is faster on CPU
        print("Faster-Whisper model loaded")

    def process_audio(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        segments, info = self.model.transcribe(
            file_path,
            vad_filter=True,    
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        transcript = []
        speaker_toggle = 0

        for seg in segments:
            # simple speaker alternation heuristic
            speaker = "Employee" if speaker_toggle % 2 == 0 else "Customer"
            speaker_toggle += 1

            transcript.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "speaker": speaker,
                "text": seg.text.strip()
            })

        return transcript

    def export_to_json(self, transcript, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, indent=4, ensure_ascii=False)
        return output_path
