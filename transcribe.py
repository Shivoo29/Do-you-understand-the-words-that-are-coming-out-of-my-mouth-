"""Transcribe every train/test clip once with Whisper and cache to transcripts.csv."""
import csv
from pathlib import Path

import librosa
from faster_whisper import WhisperModel

DATA = Path("data/Dataset_Final")
OUT = Path("transcripts.csv")

# Resume support: skip clips already transcribed.
done = set()
if OUT.exists():
    with OUT.open() as f:
        done = {(r["split"], r["filename"]) for r in csv.DictReader(f)}  # names repeat across train/test

clips = [(s, p) for s in ("train", "test") for p in sorted((DATA / s).glob("*.wav")) if (s, p.name) not in done]
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

new = not OUT.exists()
with OUT.open("a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["filename", "split", "text", "avg_logprob", "no_speech_prob", "duration"])
    for i, (split, path) in enumerate(clips, 1):
        # No VAD filter / no condition_on_previous_text: keep disfluencies & errors, they ARE the signal.
        segs, info = model.transcribe(librosa.load(path, sr=16000)[0],  # ponytail: librosa sidesteps PyAV version clash
                                        language="en", beam_size=5, condition_on_previous_text=False)
        segs = list(segs)
        text = " ".join(s.text.strip() for s in segs)
        lp = sum(s.avg_logprob for s in segs) / len(segs) if segs else 0.0
        nsp = max((s.no_speech_prob for s in segs), default=1.0)
        w.writerow([path.name, split, text, round(lp, 4), round(nsp, 4), round(info.duration, 2)])
        f.flush()
        print(f"{i}/{len(clips)} {path.name}: {text[:70]}", flush=True)
