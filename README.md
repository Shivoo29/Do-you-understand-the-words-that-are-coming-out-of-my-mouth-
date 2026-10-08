# Do you understand the words that are coming out of my mouth?!

![Detective Lee, moments before being asked the most important question in speech assessment](assets/do-you-understand.png)

> **Carter:** *"Do you understand the words that are coming out of my mouth?"*
> **This repo:** *"Yes. Also, your subject–verb agreement is a solid 4 out of 5."*

A **grammar scoring engine for spoken English**, built for the SHL Hiring Assessment 2026. Feed it a 45–60 second audio clip and it returns a grammar score from 0 to 5, the same scale human raters use. No karate required.

| | |
|---|---|
| **5-fold CV** | RMSE **0.464**, Pearson r **0.927** |
| **Training RMSE** (in-sample) | **0.377** |
| **Public leaderboard** | RMSE **0.390** |

The full report (approach, ablation, plots, training RMSE) lives in [`solution.ipynb`](solution.ipynb). Carter would not read it. You should.

---

## How it works (the "Lee explains the case" section)

Grammar can be judged in two ways: **what** someone said and **how** they said it. The model listens to both and asks two small regressors (Ridge + LightGBM) to agree on a number.

| Clue | What we use | Where |
|---|---|---|
| 🎙️ What was said | Whisper large-v3 transcript, keeping the *uh*s and *um*s on purpose | `transcribe.py` |
| 📏 How fluently | 8 features: words/sec, sentence length, filler rate, ASR confidence… | notebook |
| ✅ Is it grammatical? | RoBERTa trained on CoLA, scoring every sentence | notebook |
| 🧑‍⚖️ What would a strict examiner say? | An LLM applies the SHL rubric and returns 1–5 | `llm_score.py` |
| 🔊 What the transcript hides | Frozen WavLM-large (layers 13/14/19/21) → Ridge, out-of-fold | `audio_embed.py` |
| 🥋 Trained specifically for this | WavLM-base-plus fine-tuned to predict the score, 30 s crops, 10 folds | `finetune.py` |

**Why the audio matters:** speech recognition politely "fixes" people's grammar while transcribing. That's great for subtitles and terrible for grading. The audio models hear the hesitations and self-corrections the transcript quietly smooths away, which is why they produce the biggest jump below.

### Every clue earns its place

| Features (cumulative) | CV RMSE ↓ | CV r ↑ |
|---|---|---|
| fluency | 0.857 | 0.722 |
| + CoLA | 0.843 | 0.733 |
| + LLM judge | 0.784 | 0.774 |
| + WavLM-large (frozen) | 0.504 | 0.914 |
| + WavLM fine-tuned | 0.466 | 0.927 |
| **Ridge + LightGBM average** | **0.464** | **0.927** |

### Things that didn't make the cut
- **Fine-tuning WavLM-large:** on a 12 GB GPU it only fit a batch size of 2 and lost to the smaller model on every fold.
- **Averaging two fine-tuning seeds:** no CV gain. One seed was simply worse (OOF 0.569 vs 0.535), a reminder that 769 clips is not a lot.
- **Gemini and Groq as judges:** free tiers ran out (Gemini after 20 clips), and Groq's judge correlated far less with the labels (r ≈ 0.47 vs 0.68).

---

## Reproduce it ("Follow me, Carter")

The competition data is **not** included, because the rules say so. Put it at `data/Dataset_Final/{train,test}/*.wav` along with the CSVs.

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv -r requirements.txt
source .venv/bin/activate
cp .env.example .env            # add an API key for whichever LLM judge you run

python transcribe.py            # -> transcripts.csv                         (~25 min, GPU)
python llm_score.py claude      # -> llm_scores_claude.csv   (also: groq | gemini)
python audio_embed.py large     # -> audio_large.npz                         (~10 min, GPU)
FOLDS=10 CROP_S=30 EPOCHS=14 python finetune.py base-plus   # -> finetune_preds_base-plus_f10c30e14.csv (~4 h, GPU)
jupyter nbconvert --to notebook --execute solution.ipynb   # -> submission.csv
```

Every step caches its output and resumes if interrupted. The small caches (LLM scores, fine-tune predictions) are committed. Two are not:
- `transcripts.csv`: derived from the competition data, which the rules forbid sharing.
- `audio_large.npz`: 97 MB.

Run `transcribe.py` and `audio_embed.py large` before the notebook.

## Honesty corner
`llm_scores_claude.csv` was **not** produced through the API. The Anthropic account had no credits, so the scores were produced by Claude Opus 5.5 in a Claude Code session, using the same rubric prompt as `llm_score.py` and **without seeing any train labels**. Running `python llm_score.py claude` regenerates them through the API; values may differ slightly.

**Hardware:** RTX 4070 Super (12 GB), Python 3.12.

---

<sub>Image: still from *Rush Hour* (1998, New Line Cinema), used here as a reference/commentary. All rights belong to their owners.</sub>
