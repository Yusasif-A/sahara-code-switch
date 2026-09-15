"""
Publish the exact audio used in the benchmark to a Hugging Face dataset repo.

The challenge asks entrants to submit the code-switched audio they benchmarked
on. This uploads only the clips that actually appear in a results file — not
everything in samples/ — so the dataset and the reported numbers describe the
same 16 recordings and anyone can re-run the comparison against them.

Each clip ships with its ground-truth transcript, its language, and a
metadata.csv that Hugging Face reads as an audio dataset.

    python push_dataset.py --dry-run    # list what would be uploaded
    python push_dataset.py              # upload

Provenance note: the audio originates from intronhealth/AfriSwitch, which is a
gated dataset. It is republished here because the organisers asked entrants to
submit the clips they used, and the organisers own the source — but the README
credits it explicitly rather than presenting the recordings as ours.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import shutil
import sys
from pathlib import Path

from config import settings

REPO_ID = "yusasif/intron-stt_tts-benchmark"
STAGING = Path(__file__).with_name("_dataset_upload")
SAMPLES = Path(__file__).with_name("samples")

LANGUAGE_NAMES = {
    "ha": "Hausa",
    "ig": "Igbo",
    "pcm": "Nigerian Pidgin",
    "yo": "Yoruba",
    "en": "English",
}


def clips_actually_benchmarked() -> dict[str, set[str]]:
    """Clip ids that appear in at least one results file, and where."""
    used: dict[str, set[str]] = {}
    for path in sorted(set(glob.glob("*results*.json") + glob.glob("tts_*.json"))):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows = data.get("results", []) if isinstance(data, dict) else data
        for row in rows:
            name = row.get("sample") or row.get("phrase")
            if name:
                used.setdefault(name, set()).add(path)
    return used


def language_of(clip: str) -> str:
    lang_file = SAMPLES / f"{clip}.lang"
    if lang_file.exists():
        raw = lang_file.read_text(encoding="utf-8").strip().lower()
        return {"yoruba": "yo", "hausa": "ha", "igbo": "ig", "pidgin": "pcm"}.get(raw, raw)
    for code, word in (("yo", "yoruba"), ("ha", "hausa"), ("ig", "igbo"), ("pcm", "pidgin")):
        if word in clip:
            return code
    return "en"


def build(used: dict[str, set[str]]) -> list[dict]:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    (STAGING / "data").mkdir(parents=True)

    rows: list[dict] = []
    for clip in sorted(used):
        wav, txt = SAMPLES / f"{clip}.wav", SAMPLES / f"{clip}.txt"
        if not wav.exists() or not txt.exists():
            print(f"    skipping {clip}: missing audio or transcript")
            continue
        shutil.copy2(wav, STAGING / "data" / wav.name)
        language = language_of(clip)
        rows.append(
            {
                "file_name": f"data/{wav.name}",
                "clip_id": clip,
                "language": language,
                "language_name": LANGUAGE_NAMES.get(language, language),
                "transcript": txt.read_text(encoding="utf-8").strip(),
                "source_dataset": "intronhealth/AfriSwitch",
                "used_in": ";".join(sorted(used[clip])),
            }
        )

    with (STAGING / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_language: dict[str, int] = {}
    for row in rows:
        by_language[row["language_name"]] = by_language.get(row["language_name"], 0) + 1
    counts = "\n".join(f"| {k} | {v} |" for k, v in sorted(by_language.items()))

    (STAGING / "README.md").write_text(
        f"""---
license: other
task_categories:
- automatic-speech-recognition
- text-to-speech
language:
- ha
- ig
- yo
- en
tags:
- code-switching
- nigerian-languages
- speech-benchmark
---

# Code-switched benchmark audio

The exact {len(rows)} recordings used to benchmark speech models for the Sahara
CodeSwitch Africa Challenge. Published so the reported numbers can be checked
against the audio that produced them.

Every clip is intra-sentential code-switching — one speaker moving between a
Nigerian language and English inside a single utterance, which is the case the
challenge is judged on.

| Language | Clips |
|---|---|
{counts}

## Source

Audio is from [intronhealth/AfriSwitch](https://huggingface.co/datasets/intronhealth/AfriSwitch),
published by Intron Health. It is reproduced here only because entrants were
asked to submit the clips they benchmarked on; all credit for the recordings
belongs to Intron. Use of the audio is governed by the terms of the source
dataset.

## Contents

- `data/*.wav` — 16 kHz mono
- `metadata.csv` — `file_name`, `clip_id`, `language`, `transcript`,
  `source_dataset`, and `used_in` (which results file each clip appears in)

## Models benchmarked

ASR: Intron Sahara, ElevenLabs Scribe, Deepgram nova-2-phonecall, our fine-tuned model.
TTS: Intron Sahara, ElevenLabs multilingual v2, our fine-tuned multilingual model.

Metrics follow Intron's AfriHealth MultiBench: WER and CER, normalised and
unnormalised, reported per language.
""",
        encoding="utf-8",
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish benchmark audio to Hugging Face.")
    parser.add_argument("--dry-run", action="store_true", help="build locally, do not upload")
    parser.add_argument("--repo", default=REPO_ID)
    parser.add_argument("--private", action="store_true", help="create the repo private")
    args = parser.parse_args()

    token = settings.stt.hf_token
    if not token:
        print("\n  HF_TOKEN is not set in .env\n")
        sys.exit(1)

    used = clips_actually_benchmarked()
    if not used:
        print("\n  No results files found — run a benchmark first.\n")
        sys.exit(1)

    print(f"\n  {len(used)} clips were benchmarked. Staging ...")
    rows = build(used)
    print(f"  Built {STAGING.name}/ with {len(rows)} clips + metadata.csv + README.md")

    if args.dry_run:
        print("\n  --dry-run: nothing uploaded. Inspect the folder, then re-run without it.\n")
        return

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    print(f"  Creating/confirming dataset repo {args.repo} ...")
    api.create_repo(
        repo_id=args.repo, repo_type="dataset", private=args.private, exist_ok=True
    )
    print("  Uploading ...")
    api.upload_folder(
        folder_path=str(STAGING), repo_id=args.repo, repo_type="dataset"
    )
    print(f"\n  Done: https://huggingface.co/datasets/{args.repo}\n")


if __name__ == "__main__":
    main()
