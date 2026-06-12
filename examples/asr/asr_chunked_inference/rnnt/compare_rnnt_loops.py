"""
compare_rnnt_loops.py

Compares RNNT inner-loop structure between:
  LOCAL : nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py
  MASTER: nemo/collections/asr/parts/submodules/transducer_decoding/rnnt_label_looping.py

Usage:
    python compare_rnnt_loops.py
    python compare_rnnt_loops.py --context 5   # lines of context around each loop
"""

import argparse
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to NeMo_fc root, or override with --local / --master)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]  # NeMo_fc/

LOCAL_FILE  = REPO_ROOT / "nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py"
MASTER_FILE = REPO_ROOT / "nemo/collections/asr/parts/submodules/transducer_decoding/rnnt_label_looping.py"

# ---------------------------------------------------------------------------
# Loop patterns to search
# ---------------------------------------------------------------------------
LOCAL_PATTERNS = [
    (r"for time_idx in range",                             "OUTER time-step loop"),
    (r"while not_blank.*symbols_added",                    "INNER label loop  (while not_blank)"),
    (r"while need_loop.*symbols_added",                    "INNER label loop  (while need_loop — CUDA graph)"),
]

MASTER_PATTERNS = [
    (r"while active_mask\.any\(\)",                        "OUTER loop        (while active_mask.any)"),
    (r"while advance_mask\.any\(\)",                       "INNER label loop  (while advance_mask.any)"),
    (r"while self\.state\.active_mask_any",                "OUTER loop        (CUDA graph — active_mask_any)"),
    (r"while self\.state\.advance_mask_any",               "INNER label loop  (CUDA graph — advance_mask_any)"),
]


def find_loops(filepath: Path, patterns: list, context: int = 3) -> list[dict]:
    lines = filepath.read_text().splitlines()
    results = []
    for pat, label in patterns:
        rx = re.compile(pat)
        for i, line in enumerate(lines):
            if rx.search(line):
                start = max(0, i - 1)
                end   = min(len(lines), i + context + 1)
                results.append({
                    "label":   label,
                    "lineno":  i + 1,
                    "line":    line.strip(),
                    "snippet": lines[start:end],
                })
    return results


def print_section(title: str, filepath: Path, loops: list[dict], context: int):
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {title}")
    print(f"  File: {filepath}")
    print(sep)

    # Summary count
    outer = sum(1 for l in loops if "OUTER" in l["label"])
    inner = sum(1 for l in loops if "INNER" in l["label"])
    print(f"  OUTER loops : {outer}")
    print(f"  INNER loops : {inner}")
    print(f"  Total       : {len(loops)}")
    print()

    for loop in loops:
        print(f"  Line {loop['lineno']:>4}  [{loop['label']}]")
        print(f"           {loop['line']}")
        if context > 0:
            print("         Context:")
            for ctx_line in loop["snippet"]:
                print(f"           {ctx_line}")
        print()


def print_diff_summary(local_loops: list[dict], master_loops: list[dict]):
    sep = "-" * 72
    print(f"\n{sep}")
    print("  DIFF SUMMARY")
    print(sep)

    local_outer  = sum(1 for l in local_loops  if "OUTER" in l["label"])
    local_inner  = sum(1 for l in local_loops  if "INNER" in l["label"])
    master_outer = sum(1 for l in master_loops if "OUTER" in l["label"])
    master_inner = sum(1 for l in master_loops if "INNER" in l["label"])

    print(f"  {'':30s} {'LOCAL':>8}  {'MASTER':>8}")
    print(f"  {'OUTER loops':30s} {local_outer:>8}  {master_outer:>8}")
    print(f"  {'INNER loops':30s} {local_inner:>8}  {master_inner:>8}")
    print(f"  {'Total loops':30s} {len(local_loops):>8}  {len(master_loops):>8}")
    print()

    print("  KEY STRUCTURAL DIFFERENCES")
    print()
    diffs = [
        ("Outer loop variable",
         "for time_idx in range(out_len)",
         "while active_mask.any()  [per-sample tensor]"),
        ("Blank handling",
         "not_blank = False  → exit inner loop",
         "time_indices += blank_mask  → advance frame, keep looping"),
        ("Inner loop condition",
         "while not_blank AND symbols_added < max_symbols",
         "while advance_mask.any()  [any sample still needs non-blank]"),
        ("max_symbols enforcement",
         "Counter stops inner loop entry",
         "Post-loop force-blank: time_indices += force_blank_mask"),
        ("Per-sample independence",
         "All batch items at same time_idx (padded)",
         "time_indices[] per sample — items can be at different frames"),
        ("State on blank",
         "Restore prev hidden + label (batched variant lines 906-924)",
         "No restore — state advances; active_mask gates updates"),
        ("CUDA graph support",
         "No",
         "Yes — separate cuda_graph_impl path (lines 763-774)"),
        ("Fusion model integration",
         "No",
         "Yes — fusion scores combined inside inner loop"),
    ]
    for name, local_val, master_val in diffs:
        print(f"  [{name}]")
        print(f"    LOCAL : {local_val}")
        print(f"    MASTER: {master_val}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Compare RNNT inner loops: local vs NeMo master")
    parser.add_argument("--local",   default=str(LOCAL_FILE),  help="Path to local rnnt_greedy_decoding.py")
    parser.add_argument("--master",  default=str(MASTER_FILE), help="Path to master rnnt_label_looping.py")
    parser.add_argument("--context", type=int, default=3,      help="Lines of context shown per loop (default 3)")
    args = parser.parse_args()

    local_path  = Path(args.local)
    master_path = Path(args.master)

    for p in [local_path, master_path]:
        if not p.exists():
            print(f"ERROR: file not found: {p}")
            raise SystemExit(1)

    local_loops  = find_loops(local_path,  LOCAL_PATTERNS,  args.context)
    master_loops = find_loops(master_path, MASTER_PATTERNS, args.context)

    print_section("LOCAL  — rnnt_greedy_decoding.py",  local_path,  local_loops,  args.context)
    print_section("MASTER — rnnt_label_looping.py",    master_path, master_loops, args.context)
    print_diff_summary(local_loops, master_loops)


if __name__ == "__main__":
    main()
