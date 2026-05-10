#!/usr/bin/env python3
"""
batch_run.py
────────────
Run TiPAI-TSPO over a CSV or JSON file of prompts and collect results.

Usage
-----
    python batch_run.py prompts.csv
    python batch_run.py prompts.json
    python batch_run.py prompts.csv --config config.yaml --seed 42
    python batch_run.py prompts.json --results-dir my_results --log results.csv

Input CSV (one prompt per line, header optional):
    prompt
    a soldier in combat
    a woman at the beach

    OR bare lines with no header:
    a soldier in combat
    a woman at the beach

Input JSON (all shapes auto-detected):
    {"prompts": [{"prompt": "...", ...}, ...]}  <- objects under any top-level key
    [{"prompt": "...", ...}, ...]               <- top-level list of objects
    ["a soldier in combat", ...]                <- top-level list of strings
    {"prompt": "a soldier in combat"}           <- single object (one run)

    Any extra fields (id, category, source, ...) are silently ignored.

Output
------
  <results-dir>/
      <slug>__<base_model>__<idx>.png  <- one image per prompt
  <log>                                <- CSV: index, prompt, image_name, safe,
                                              adv_final, interventions,
                                              adv_improvement, elapsed_s, error
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def slugify(text: str, maxlen: int = 40) -> str:
    """Turn a prompt into a safe filename fragment."""
    s = text.lower()[:maxlen]
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "prompt"


def model_tag(base_model: str) -> str:
    """
    Extract a short tag from the base_sd_model config value.
    e.g. 'runwayml/stable-diffusion-v1-5'  -> 'sdv1-5'
         'stabilityai/stable-diffusion-2-1' -> 'sd2-1'
         '/local/path/my_custom_model'       -> 'my_custom_model'
    """
    name = base_model.split("/")[-1]
    name = re.sub(r"stable-diffusion-", "sd", name, flags=re.I)
    name = re.sub(r"[^a-z0-9_-]", "", name.lower())
    return name or "model"


# ── JSON reader ───────────────────────────────────────────────────────────────

def _extract_from_list(items: list) -> list:
    """Pull prompt strings out of a JSON list (of strings or objects)."""
    prompts = []
    for item in items:
        if isinstance(item, str):
            p = item.strip()
        elif isinstance(item, dict):
            p = str(item.get("prompt", "")).strip()
        else:
            continue
        if p and not p.startswith("#"):
            prompts.append(p)
    return prompts


def read_prompts_json(path: str) -> list:
    """
    Read prompts from a JSON file. Handles four shapes:

      {"prompts": [{"prompt": "...", ...}, ...]}  <- object with a list under any key
      [{"prompt": "...", ...}, ...]               <- top-level list of objects
      ["prompt text", ...]                        <- top-level list of strings
      {"prompt": "single prompt", ...}            <- single object (one run)

    Any extra fields (id, category, source, ...) are silently ignored.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Top-level list
    if isinstance(data, list):
        return _extract_from_list(data)

    # Single object with a bare "prompt" string key -> one-item run
    if isinstance(data, dict) and isinstance(data.get("prompt"), str):
        p = data["prompt"].strip()
        return [p] if p else []

    # Object whose values include a list -> find first list value
    if isinstance(data, dict):
        for val in data.values():
            if isinstance(val, list):
                return _extract_from_list(val)

    raise ValueError(
        f"[batch_run] Cannot parse prompts from JSON {path!r}. "
        "Expected a list, or an object containing a list of prompt objects."
    )


# ── CSV reader ────────────────────────────────────────────────────────────────

def read_prompts_csv(path: str) -> list:
    """
    Read prompts from a CSV file.
      - If the first line is 'prompt' (header), uses DictReader on that column.
      - Otherwise every non-blank, non-comment line is a prompt.
    """
    prompts = []
    with open(path, newline="", encoding="utf-8") as f:
        first = f.readline().strip()
        f.seek(0)
        if first.lower() in ("prompt", '"prompt"'):
            reader = csv.DictReader(f)
            for row in reader:
                p = row.get("prompt", "").strip()
                if p and not p.startswith("#"):
                    prompts.append(p)
        else:
            for line in f:
                p = line.strip()
                if p and not p.startswith("#"):
                    prompts.append(p)
    return prompts


# ── Dispatcher ────────────────────────────────────────────────────────────────

def read_prompts(path: str) -> list:
    """Route to JSON or CSV reader based on file extension."""
    ext = Path(path).suffix.lower()
    if ext == ".json":
        return read_prompts_json(path)
    elif ext in (".csv", ".txt", ""):
        return read_prompts_csv(path)
    else:
        # Unknown extension: try JSON, fall back to CSV
        try:
            return read_prompts_json(path)
        except (json.JSONDecodeError, ValueError):
            return read_prompts_csv(path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Batch runner for TiPAI-TSPO")
    p.add_argument("prompts_file",               help="CSV or JSON file with prompts")
    p.add_argument("--config",     default="config.yaml",
                                                help="Path to config.yaml")
    p.add_argument("--seed",       type=int, default=None,
                                                help="Fixed seed for all runs (default: vary per prompt)")
    p.add_argument("--results-dir", default="results",
                                                help="Directory to save images (default: results)")
    p.add_argument("--log",        default="batch_results.csv",
                                                help="Output CSV log path (default: batch_results.csv)")
    p.add_argument("--run-py",     default="run.py",
                                                help="Path to run.py (default: tipai/run.py)")
    p.add_argument("--base-model", default=None,
                                                help="Override base_sd_model (passed through to run.py)")
    p.add_argument("--lora",       default=None,
                                                help="LoRA weights path (passed through to run.py)")
    p.add_argument("--lora-scale", type=float, default=0.8)
    p.add_argument("--skip-existing", action="store_true",
                                                help="Skip prompts whose output image already exists")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    prompts = read_prompts(args.prompts_file)
    if not prompts:
        print("[batch_run] No prompts found in input file. Exiting.")
        sys.exit(1)

    fmt = "JSON" if Path(args.prompts_file).suffix.lower() == ".json" else "CSV"
    print(f"[batch_run] Loaded {len(prompts)} prompt(s) from {args.prompts_file} ({fmt})")

    os.makedirs(args.results_dir, exist_ok=True)

    # ── Resolve base_model tag for filenames ──────────────────────────────────
    import yaml
    with open(args.config) as f:
        cfg_yaml = yaml.safe_load(f)
    effective_model = args.base_model or cfg_yaml.get("base_sd_model", "model")
    mtag = model_tag(effective_model)
    print(f"[batch_run] Base model tag: '{mtag}'  (from '{effective_model}')")

    # ── Prepare log CSV ───────────────────────────────────────────────────────
    log_path    = args.log
    log_existed = os.path.exists(log_path)
    log_fh      = open(log_path, "a", newline="", encoding="utf-8")
    log_writer  = csv.writer(log_fh)
    if not log_existed:
        log_writer.writerow([
            "index", "prompt", "image_name",
            "safe", "adv_final", "interventions", "adv_improvement",
            "elapsed_s", "error",
        ])
        log_fh.flush()

    # ── Run each prompt ───────────────────────────────────────────────────────
    passed = failed = skipped = 0

    for idx, prompt in enumerate(prompts, start=1):
        slug     = slugify(prompt)
        img_name = f"{slug}__{mtag}__{idx:04d}.png"
        img_path = os.path.join(args.results_dir, img_name)

        print(f"\n[batch_run] [{idx}/{len(prompts)}] {prompt!r}")
        print(f"            -> {img_path}")

        if args.skip_existing and os.path.exists(img_path):
            print("            (already exists, skipping)")
            skipped += 1
            log_writer.writerow([idx, prompt, img_name, "", "", "", "", 0, "skipped"])
            log_fh.flush()
            continue

        seed = args.seed if args.seed is not None else (42 + idx)
        cmd  = [
            sys.executable, args.run_py,
            "--prompt",  prompt,
            "--config",  args.config,
            "--seed",    str(seed),
            "--out",     img_path,
        ]
        if args.base_model:
            cmd += ["--base-model", args.base_model]
        if args.lora:
            cmd += ["--lora", args.lora, "--lora-scale", str(args.lora_scale)]

        t0    = time.time()
        error = ""
        safe  = adv_final = interventions = adv_improvement = ""

        try:
            proc    = subprocess.run(cmd, capture_output=False, text=True)
            elapsed = round(time.time() - t0, 1)

            if proc.returncode == 0:
                safe = "True"
                passed += 1
                print(f"            SAFE  ({elapsed}s)")
            else:
                safe = "False"
                failed += 1
                print(f"            UNSAFE  ({elapsed}s)")

        except Exception as exc:
            elapsed = round(time.time() - t0, 1)
            error   = str(exc)
            failed += 1
            print(f"            ERROR: {exc}")

        log_writer.writerow([
            idx, prompt, img_name,
            safe, adv_final, interventions, adv_improvement,
            elapsed, error,
        ])
        log_fh.flush()

    log_fh.close()

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"""
-- Batch complete -----------------------------------------------
  Total   : {len(prompts)}
  Safe    : {passed}
  Unsafe  : {failed}
  Skipped : {skipped}
  Log     : {log_path}
  Images  : {args.results_dir}/
----------------------------------------------------------------""")


if __name__ == "__main__":
    main()