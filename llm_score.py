"""Ask an LLM judge to grade each transcript on the grammar rubric. Cached per provider to llm_scores_<provider>.csv.

Usage: python llm_score.py groq|gemini|claude
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

PROVIDER = sys.argv[1] if len(sys.argv) > 1 else "groq"
OUT = Path(f"llm_scores_{PROVIDER}.csv")
for line in Path(".env").read_text().splitlines():
    k, _, v = line.partition("=")
    os.environ.setdefault(k.strip(), v.strip())

PROMPT = """You grade spoken English grammar. Below is an automatic transcript of a 45-60 second spoken answer.
The transcript may hide some errors (speech recognition tends to clean up grammar), so judge sentence structure,
syntax, tense/agreement errors, incomplete sentences and complexity of structures you can see.

Rubric:
1 - struggles with sentence structure and syntax; limited control of simple structures, memorized patterns.
2 - limited understanding; simple structures with consistent basic mistakes; may leave sentences incomplete.
3 - decent grasp of sentence structure but errors in grammar, or decent grammar but errors in syntax/structure.
4 - strong understanding; good control of grammar; occasional minor errors that don't hurt understanding.
5 - high grammatical accuracy, adept control of complex grammar, seldom noticeable mistakes.

Transcript:
\"\"\"{text}\"\"\"

Reply with JSON only: {{"score": <number 1-5, half points allowed>, "errors": <count of grammar errors you see>}}"""


def ask(prompt):
    """One judge call -> raw JSON text. Each provider's SDK, smallest working call."""
    if PROVIDER == "groq":
        r = client.chat.completions.create(model="openai/gpt-oss-120b", temperature=0,
                                           response_format={"type": "json_object"},
                                           messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content
    if PROVIDER == "gemini":
        r = client.models.generate_content(model="gemini-2.5-flash", contents=prompt,
                                           config={"response_mime_type": "application/json", "temperature": 0})
        return r.text
    r = client.messages.create(model="claude-opus-5-5", max_tokens=2000,
                               output_config={"effort": "low"},
                               messages=[{"role": "user", "content": prompt}])
    text = next(b.text for b in r.content if b.type == "text")
    return text[text.find("{"): text.rfind("}") + 1]


if PROVIDER == "groq":
    import groq
    client, RateLimit = groq.Groq(), groq.RateLimitError
elif PROVIDER == "gemini":
    from google import genai
    from google.genai import errors
    client, RateLimit = genai.Client(api_key=os.environ["GEMINI_API_KEY"]), errors.ClientError  # 429 arrives as ClientError
else:
    import anthropic
    client, RateLimit = anthropic.Anthropic(), anthropic.RateLimitError

done = set()
if OUT.exists():
    done = {(r["split"], r["filename"]) for r in csv.DictReader(OUT.open())}
t = pd.read_csv("transcripts.csv").fillna({"text": ""})
todo = t[[(s, f) not in done for s, f in zip(t.split, t.filename)]]

new = not OUT.exists()
with OUT.open("a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["filename", "split", "llm_score", "llm_errors"])
    for i, r in enumerate(todo.itertuples(), 1):
        for attempt in range(8):
            try:
                d = json.loads(ask(PROMPT.format(text=r.text or "(no speech)")))
                score, errs = float(d["score"]), float(d.get("errors", -1))
                break
            except RateLimit:
                time.sleep(15 * (attempt + 1))  # free-tier tokens/min cap: back off and retry
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                score, errs = float("nan"), float("nan")  # bad reply: leave blank, model imputes
                break
        else:
            raise SystemExit("still rate-limited after retries; rerun later, progress is saved")
        w.writerow([r.filename, r.split, score, errs])
        f.flush()
        print(f"{i}/{len(todo)} {r.split}/{r.filename}: {score}", flush=True)
