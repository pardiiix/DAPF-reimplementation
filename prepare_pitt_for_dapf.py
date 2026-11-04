#!/usr/bin/env python3
"""Convert DementiaBank Pitt Cookie Theft CHAT transcripts into DAPF source CSV.

Run on your Mac (from anywhere):
    python3 prepare_pitt_for_dapf.py

Input (extracted corpus):
    .../DAPF-main/data/pitt_zip/
Outputs:
    .../DAPF-main/data/pitt/pitt.csv                (all 477 eligible recordings)
    .../DAPF-main/data/pitt/pitt_paper2b_105.csv    (52 AD + 53 controls)
    .../DAPF-main/data/pitt/pitt_paper2b_105_manifest.csv
    .../DAPF-main/data/pitt/pitt_paper2b_105_sampling.json

The Paper 2b sample is diagnosis-stratified, includes at most one recording
per participant, and uses a fixed seed (42).  The original 477-row dataset
is retained.  Do not construe this matched-size sampling as an exact
replication of Farzana and Parde's published DB cohort.

The DAPF loader expects these exact columns:
    filename, id, text, ad, domain, domain1, domain2

Eligibility:
    @ID: PAR diagnosis ProbableAD in Dementia/cookie -> 'dementia'
    @ID: PAR diagnosis Control in Control/cookie -> 'healthy'
    Other diagnoses and non-Cookie tasks are NOT used.

No audio or clinical metadata enters the classifier text. CHAT annotations are
removed but spoken fillers and repeated/false-start words are largely retained.
A manifest and exclusions table allow an audit of the conversion.

This is a transparent preprocessing implementation, not a claim to reconstruct
Farzana & Parde's original unpublished preprocessing decisions exactly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
from collections import Counter
from pathlib import Path

# Your Mac directories. These can also be overridden at the command line.
PROJECT = Path("/Users/pardisranjbar/Desktop/Dementia Detection/projects/final/DAPF-main")
DEFAULT_INPUT = PROJECT / "data" / "pitt_zip"
DEFAULT_OUTPUT = PROJECT / "data" / "pitt"
PAPER2B_SIZE = 105
PAPER2B_AD_COUNT = 52
PAPER2B_SEED = 42

# Source-domain text for DAPF's domain-specific manual templates.
# Check the specific template_id before training: the loader chooses one of
# domain/domain1/domain2 depending on template_id. These are reasonable
# descriptions, not verified original DAPF domain-string values.
DOMAIN = "picture description"
DOMAIN1 = "Cookie Theft picture description"
DOMAIN2 = "description of the Cookie Theft picture"

CSV_FIELDS = ["filename", "id", "text", "ad", "domain", "domain1", "domain2"]
TIME_MARKER = re.compile(r"\x15[^\x15]*\x15")
CHAT_ANNOTATION = re.compile(r"\[[^\]]*\]")
SPOKEN_FILLER = re.compile(r"&-([A-Za-z]+)")
NONVERBAL = re.compile(r"&=[^\s]+")
PHONOLOGICAL_FRAGMENT = re.compile(r"&\+[^\s]+")
PARTIAL_WORD = re.compile(r"\b([A-Za-z]+)\(([A-Za-z]+)\)")


def extract_par_diagnosis(chat_text: str, file_name: str) -> str:
    """Get PAR's diagnosis from its @ID CHAT metadata, not directory alone."""
    diagnoses = []
    for line in chat_text.splitlines():
        if not line.startswith("@ID:"):
            continue
        fields = line.partition(":")[2].strip().split("|")
        if len(fields) > 5 and fields[2].strip().upper() == "PAR":
            diagnoses.append(fields[5].strip())
    if len(diagnoses) != 1:
        raise ValueError(f"Expected exactly one PAR @ID for {file_name}; found {diagnoses}")
    return diagnoses[0]


def extract_par_utterances(chat_text: str) -> list[str]:
    """Read the main *PAR tier; ignore interviewer, %mor/%gra/%wor, etc."""
    utterances = []
    continuation_of_par = False
    for line in chat_text.splitlines():
        if line.startswith("*PAR:"):
            utterances.append(line.split(":", 1)[1].strip())
            continuation_of_par = True
        elif line.startswith("*") or line.startswith("%") or line.startswith("@"):
            continuation_of_par = False
        elif continuation_of_par and line[:1] in ("\t", " ") and line.strip():
            # CHAT may wrap a long speaker turn onto one or more indented lines.
            utterances[-1] += " " + line.strip()
        else:
            continuation_of_par = False
    return utterances


def clean_chat_utterance(utterance: str) -> str:
    """Convert the participant's CHAT tier into readable plain text.

    We remove CHAT control codes but retain real spoken hesitations, e.g.
    '&-uh' -> 'uh'. We also retain the spoken words before [//] repairs,
    because those false starts may be informative in dementia detection.
    """
    s = TIME_MARKER.sub(" ", utterance)
    s = SPOKEN_FILLER.sub(r"\1", s)
    s = NONVERBAL.sub(" ", s)       # &=laughs is an event, not a spoken word
    s = PHONOLOGICAL_FRAGMENT.sub(" ", s)
    s = CHAT_ANNOTATION.sub(" ", s) # [+ exc], [//], [: correction], etc.
    s = PARTIAL_WORD.sub(r"\1\2", s)  # fallin(g) -> falling
    s = s.replace("_", " ")             # out_of -> out of

    # Remove CHAT overlap, repair, and turn-boundary symbols.
    s = re.sub(r"\+(?:\.\.\.|//?|<?|!?|\^|\+)", " ", s)
    s = re.sub(r"[‡≠⌈⌉⌊⌋<>↑↓↫↻°∬§]", " ", s)
    s = s.replace("&", " ")
    s = re.sub(r"\b(?:xxx|yyy|www)\b(?:@\w+)?", " ", s, flags=re.I)
    s = re.sub(r"\b0(?=[A-Za-z])\w*", " ", s)  # zero-prefixed unspoken items
    s = re.sub(r"\b[A-Za-z]+@\w+\b", lambda m: m.group().split("@", 1)[0], s)

    # Keep normal word punctuation, contractions, and speech-like text.
    s = re.sub(r"[^\w\s'.,!?;:\-]", " ", s)
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip(" -\t\n\r")
    return s


def is_cookie_file(p: Path) -> bool:
    return any(x.lower() == "cookie" for x in p.parts[:-1]) and p.suffix.lower() == ".cha"


def parent_group(p: Path) -> str:
    # A normal Pitt extracted path ends in Dementia/cookie/name.cha or
    # Control/cookie/name.cha, possibly beneath an extra Pitt/ directory.
    parts = [x.lower() for x in p.parts]
    try:
        i = len(parts) - 1 - parts[::-1].index("cookie")
    except ValueError:
        return ""
    return parts[i - 1] if i else ""


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]],
              *, protect_existing: bool = False) -> None:
    """Write deterministically. Never replace an existing full dataset silently."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    payload = buffer.getvalue().encode("utf-8")
    if path.exists() and path.read_bytes() == payload:
        return
    if protect_existing and path.exists():
        raise RuntimeError(
            f"The existing full-data file differs: {path}\n"
            "No files were replaced. Compare inputs first; if intentional, "
            "use --allow-overwrite-full to regenerate full-data outputs."
        )
    path.write_bytes(payload)


def select_paper2b_sample(
    rows: list[dict[str, str]], seed: int, ad_count: int = PAPER2B_AD_COUNT,
    total_count: int = PAPER2B_SIZE,
) -> list[dict[str, str]]:
    """Stratify by label, sample unique speaker IDs, then one visit per speaker."""
    rng = random.Random(seed)
    people: dict[str, dict[str, list[dict[str, str]]]] = {
        "dementia": {}, "healthy": {}
    }
    label_by_id: dict[str, str] = {}
    for row in rows:
        person, label = row["id"], row["ad"]
        if person in label_by_id and label_by_id[person] != label:
            raise ValueError(
                f"Participant {person} has conflicting eligible labels: "
                f"{label_by_id[person]} and {label}. Resolve before sampling."
            )
        label_by_id[person] = label
        people[label].setdefault(person, []).append(row)

    counts = {"dementia": ad_count, "healthy": total_count - ad_count}
    chosen = []
    for label in ("dementia", "healthy"):
        ids = sorted(people[label])
        if len(ids) < counts[label]:
            raise ValueError(
                f"Cannot sample {counts[label]} unique {label} participants; "
                f"only {len(ids)} are available."
            )
        chosen_people = sorted(rng.sample(ids, counts[label]))
        for person in chosen_people:
            candidates = sorted(people[label][person], key=lambda r: r["filename"])
            chosen.append(rng.choice(candidates))

    chosen.sort(key=lambda r: (r["filename"], r["ad"]))
    assert len(chosen) == total_count
    assert len({r["filename"] for r in chosen}) == total_count
    assert len({r["id"] for r in chosen}) == total_count
    assert Counter(r["ad"] for r in chosen) == Counter(counts)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--paper2b-seed", type=int, default=PAPER2B_SEED)
    parser.add_argument("--paper2b-size", type=int, default=PAPER2B_SIZE)
    parser.add_argument("--paper2b-ad-count", type=int, default=PAPER2B_AD_COUNT)
    parser.add_argument("--allow-overwrite-full", action="store_true",
                        help="Explicitly allow replacing differing full-data CSVs")
    args = parser.parse_args()
    if args.paper2b_size < 2 or not (0 < args.paper2b_ad_count < args.paper2b_size):
        parser.error("Sample size must be >1, with both classes represented")

    source = args.input_dir.expanduser()
    destination = args.output_dir.expanduser()
    if not source.is_dir():
        raise FileNotFoundError(f"Input directory not found: {source}")

    all_cha = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() == ".cha")
    cookie_files = [p for p in all_cha if is_cookie_file(p)]
    if not cookie_files:
        raise RuntimeError(f"No .cha files found inside a cookie/ folder under {source}")

    included = []
    manifest = []
    excluded = []
    filenames_seen = set()
    for cha in cookie_files:
        original = cha.read_text(encoding="utf-8-sig")
        original_diagnosis = extract_par_diagnosis(original, str(cha))
        category = parent_group(cha)
        if category == "dementia" and original_diagnosis == "ProbableAD":
            label = "dementia"
        elif category == "control" and original_diagnosis == "Control":
            label = "healthy"
        else:
            excluded.append({
                "source_file": str(cha.relative_to(source)),
                "original_diagnosis": original_diagnosis,
                "folder_group": category,
                "reason": "Not an eligible ProbableAD or Control recording",
            })
            continue

        file_id = cha.stem               # e.g. 001-0 (stable recording ID)
        speaker_id = file_id.split("-")[0]  # e.g. 001 (participant ID)
        if file_id in filenames_seen:
            raise ValueError(f"Duplicate recording identifier {file_id}: {cha}")
        filenames_seen.add(file_id)

        turns = extract_par_utterances(original)
        cleaned_turns = [clean_chat_utterance(turn) for turn in turns]
        text = " ".join(turn for turn in cleaned_turns if turn).strip()
        if not text:
            raise ValueError(f"No participant speech could be extracted from {cha}")
        if any(tok in text for tok in ("@ID:", "*PAR:", "%mor:", "%gra:", "\x15")):
            raise ValueError(f"Unremoved CHAT annotation in {cha}")

        included.append({
            "filename": file_id,
            "id": speaker_id,
            "text": text,
            "ad": label,
            "domain": DOMAIN,
            "domain1": DOMAIN1,
            "domain2": DOMAIN2,
        })
        manifest.append({
            "filename": file_id,
            "id": speaker_id,
            "source_file": str(cha.relative_to(source)),
            "original_diagnosis": original_diagnosis,
            "ad": label,
            "par_turns": str(len(turns)),
            "words": str(len(text.split())),
        })

    if not included:
        raise RuntimeError("No eligible recordings after diagnosis filtering")
    if any(not all(row[field] for field in CSV_FIELDS) for row in included):
        raise ValueError("Required DAPF data fields cannot be blank")

    # Sort by original filename to make the conversion deterministic.
    included.sort(key=lambda r: (r["filename"], r["ad"]))
    manifest.sort(key=lambda r: (r["filename"], r["ad"]))
    excluded.sort(key=lambda r: r["source_file"])

    # Select before writing outputs; invalid sampling should not write anything.
    sample = select_paper2b_sample(
        included, seed=args.paper2b_seed,
        ad_count=args.paper2b_ad_count, total_count=args.paper2b_size,
    )
    source_by_filename = {r["filename"]: r for r in manifest}
    sample_manifest = [source_by_filename[r["filename"]] for r in sample]

    destination.mkdir(parents=True, exist_ok=True)
    output_csv = destination / "pitt.csv"
    full_manifest_fields = [
        "filename", "id", "source_file", "original_diagnosis", "ad",
        "par_turns", "words",
    ]
    exclusion_fields = ["source_file", "original_diagnosis", "folder_group", "reason"]

    # Protect all previous full-cohort CSVs by default. Existing byte-identical
    # outputs are preserved without even changing their file timestamps.
    full_outputs = [
        (output_csv, CSV_FIELDS, included),
        (destination / "pitt_manifest.csv", full_manifest_fields, manifest),
        (destination / "pitt_excluded.csv", exclusion_fields, excluded),
    ]
    if not args.allow_overwrite_full:
        for path, fields, rows in full_outputs:
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(buffer, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            if path.exists() and path.read_bytes() != buffer.getvalue().encode("utf-8"):
                raise RuntimeError(
                    f"Existing full-dataset file differs: {path}. No files changed. "
                    "Investigate differences or use --allow-overwrite-full deliberately."
                )
    for path, fields, rows in full_outputs:
        write_csv(path, fields, rows)

    sample_name = f"pitt_paper2b_{args.paper2b_size}"
    sample_csv = destination / f"{sample_name}.csv"
    sample_manifest_file = destination / f"{sample_name}_manifest.csv"
    write_csv(sample_csv, CSV_FIELDS, sample)
    write_csv(sample_manifest_file, full_manifest_fields, sample_manifest)

    sampling_report = {
        "method": "diagnosis_stratified_participant_sampling_one_transcript_each",
        "seed": args.paper2b_seed,
        "source_dataset": "pitt.csv",
        "full_rows": len(included),
        "full_unique_participants": len({r["id"] for r in included}),
        "sample_rows": len(sample),
        "sample_unique_participants": len({r["id"] for r in sample}),
        "sample_counts": dict(Counter(r["ad"] for r in sample)),
        "sample_csv": sample_csv.name,
        "sample_csv_sha256": hashlib.sha256(sample_csv.read_bytes()).hexdigest(),
        "notes": [
            "Only ProbableAD and Control Cookie Theft recordings are eligible.",
            "Participants sampled independently within diagnostic class.",
            "One transcript selected randomly for each chosen participant.",
            "This is an intentionally smaller, balanced source dataset, not the original published DAPF source cohort.",
        ],
    }
    (destination / f"{sample_name}_sampling.json").write_text(
        json.dumps(sampling_report, indent=2) + "\n", encoding="utf-8"
    )

    labels = Counter(r["ad"] for r in included)
    print(f"Cookie Theft .cha files found: {len(cookie_files)}")
    print(f"Other task .cha files ignored: {len(all_cha) - len(cookie_files)}")
    print(f"Dementia (ProbableAD) transcripts: {labels['dementia']}")
    print(f"Healthy (Control) transcripts:   {labels['healthy']}")
    print(f"Total included transcripts:     {len(included)}")
    print(f"Unique participants included:   {len(set(r['id'] for r in included))}")
    print(f"Cookie Theft files excluded:    {len(excluded)}")
    print(f"\nDAPF source CSV saved to: {output_csv}")
    print("Manifest and exclusion audit saved to the same directory.")
    print("For full Pitt: --data_dir ./data/ --src_data pitt/pitt")
    sample_labels = Counter(r["ad"] for r in sample)
    print(f"\nPaper 2b source sample: {len(sample)} transcripts / "
          f"{len({r['id'] for r in sample})} unique participants")
    print(f"  dementia={sample_labels['dementia']}, healthy={sample_labels['healthy']}")
    print(f"  Seed: {args.paper2b_seed}; one recording per participant")
    print(f"  Sample CSV: {sample_csv}")
    print(f"  Manifest: {sample_manifest_file}")
    print(f"  Sampling report: {destination / (sample_name + '_sampling.json')}")
    print(f"For Paper 2b: --data_dir ./data/ --src_data pitt/{sample_name}")
    if (labels["dementia"], labels["healthy"]) != (234, 243):
        print("WARNING: Counts differ from the previously inspected Pitt.zip (234/243). Review the audit.")


if __name__ == "__main__":
    main()
