"""Per-layer WavLM embeddings per clip (mean + std pooled over time). Cached to audio_<model>.npz.

Usage: python audio_embed.py [base-plus|large]
Audio carries what the transcript hides: hesitations, self-corrections, pronunciation of word endings.
"""
import sys

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import AutoFeatureExtractor, WavLMModel

name = sys.argv[1] if len(sys.argv) > 1 else "large"
hf = f"microsoft/wavlm-{name}"
t = pd.read_csv("transcripts.csv")
fe = AutoFeatureExtractor.from_pretrained(hf)
model = WavLMModel.from_pretrained(hf).cuda().eval().half()
embs = []  # (clips, layers, 2*hidden)
with torch.no_grad():
    for i, r in enumerate(t.itertuples(), 1):
        wav = librosa.load(f"data/Dataset_Final/{r.split}/{r.filename}", sr=16000)[0]
        x = fe(wav, sampling_rate=16000, return_tensors="pt").input_values.cuda().half()
        h = torch.stack(model(x, output_hidden_states=True).hidden_states).squeeze(1)  # (layers, time, hidden)
        embs.append(torch.cat([h.mean(1), h.std(1)], -1).float().cpu().numpy())
        if i % 100 == 0:
            print(f"{i}/{len(t)}", flush=True)
np.savez(f"audio_{name}.npz", emb=np.array(embs, dtype=np.float16), filename=t.filename.values, split=t.split.values)
