"""Fine-tune WavLM to regress the grammar score directly from audio, 5-fold.

Out-of-fold predictions (train) and fold-averaged predictions (test) are saved to finetune_preds.csv
and used as one stacked feature in the final model, the same way the frozen-embedding ridge is.

Usage: [FOLDS=5] [CROP_S=15] [EPOCHS=12] python finetune.py [base-plus|large] [seed]
Folds are fixed (KFold random_state=0) so out-of-fold predictions from different seeds line up and can be averaged.
"""
import os
import sys

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from transformers import WavLMModel

NAME = sys.argv[1] if len(sys.argv) > 1 else "base-plus"
FOLDS = int(os.environ.get("FOLDS", 5))
SR, CROP, EPOCHS = 16000, int(os.environ.get("CROP_S", 15)) * 16000, int(os.environ.get("EPOCHS", 12))
BS = 2 if NAME == "large" else 4  # large needs smaller batches to fit 12 GB
DATA = "data/Dataset_Final"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
torch.manual_seed(SEED)
np.random.seed(SEED)

tr = pd.read_csv(f"{DATA}/train.csv")
te = pd.read_csv(f"{DATA}/test.csv")
load = lambda split, f: librosa.load(f"{DATA}/{split}/{f}", sr=SR)[0]
wav_tr = [load("train", f) for f in tr.filename]
wav_te = [load("test", f) for f in te.filename]
y = tr.label.values.astype(np.float32)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(f"microsoft/wavlm-{NAME}", layerdrop=0.0)  # layerdrop would randomly drop hidden states
        self.wavlm.freeze_feature_encoder()  # CNN front-end stays frozen; transformer layers adapt
        self.wavlm.gradient_checkpointing_enable()
        with torch.no_grad():  # count hidden states from a dry run rather than trusting the config
            n = len(self.wavlm(torch.zeros(1, SR), output_hidden_states=True).hidden_states)
        self.layer_w = torch.nn.Parameter(torch.zeros(n))  # learned mix of all layers
        self.head = torch.nn.Linear(self.wavlm.config.hidden_size, 1)

    def forward(self, x):
        h = torch.stack(self.wavlm(x, output_hidden_states=True).hidden_states)  # (L, B, T, H)
        h = (torch.softmax(self.layer_w, 0)[:, None, None, None] * h).sum(0).mean(1)
        return self.head(h).squeeze(-1) + 3.0  # start near the label mean


def norm(w):
    return (w - w.mean()) / (w.std() + 1e-7)


def crop(w, rng):
    if len(w) <= CROP:
        return np.pad(w, (0, CROP - len(w)))
    s = rng.integers(0, len(w) - CROP)
    return w[s:s + CROP]


@torch.no_grad()
def predict(model, wavs):
    """Average over 4 evenly spaced crops so the whole clip is seen."""
    model.eval()
    out = []
    for w in wavs:
        starts = np.linspace(0, max(len(w) - CROP, 0), 4).astype(int)
        x = torch.tensor(np.stack([norm(np.pad(w[s:s + CROP], (0, CROP - len(w[s:s + CROP])))) for s in starts])).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(x).float().mean().item())
    return np.array(out)


oof, test_pred = np.zeros(len(tr)), np.zeros(len(te))
for fold, (a, b) in enumerate(KFold(FOLDS, shuffle=True, random_state=0).split(tr)):
    model = Model().cuda()
    opt = torch.optim.AdamW([
        {"params": model.wavlm.parameters(), "lr": 2e-5},
        {"params": [model.layer_w, *model.head.parameters()], "lr": 1e-3},
    ], weight_decay=0.01)
    steps = EPOCHS * int(np.ceil(len(a) / BS))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[2e-5, 1e-3], total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    rng = np.random.default_rng(fold + 100 * SEED)
    for ep in range(EPOCHS):
        model.train()
        perm = rng.permutation(a)
        for i in range(0, len(perm), BS):
            idx = perm[i:i + BS]
            x = torch.tensor(np.stack([norm(crop(wav_tr[j], rng)) for j in idx])).cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.mse_loss(model(x).float(), torch.tensor(y[idx]).cuda())
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
        p = predict(model, [wav_tr[j] for j in b])
        print(f"fold {fold} epoch {ep} val RMSE {np.sqrt(((np.clip(p, 0, 5) - y[b]) ** 2).mean()):.3f}", flush=True)
    oof[b] = p
    test_pred += predict(model, wav_te) / FOLDS
    del model, opt
    torch.cuda.empty_cache()

print(f"OOF RMSE {np.sqrt(((np.clip(oof, 0, 5) - y) ** 2).mean()):.3f}  r {np.corrcoef(oof, y)[0, 1]:.3f}")
pd.concat([
    pd.DataFrame({"filename": tr.filename, "split": "train", "ft_pred": oof}),
    pd.DataFrame({"filename": te.filename, "split": "test", "ft_pred": test_pred}),
]).to_csv(f"finetune_preds_{NAME}" + (f"_seed{SEED}" if SEED else "")
         + ("" if (FOLDS, CROP // SR, EPOCHS) == (5, 15, 12) else f"_f{FOLDS}c{CROP // SR}e{EPOCHS}") + ".csv", index=False)
