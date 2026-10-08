# Grammar Scoring Engine for Spoken English (SHL Hiring Assessment 2026)

Predicts a 0–5 grammar score for 45–60 s speech clips.

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

