#!/usr/bin/env python3
import re, sys, difflib, argparse
from pathlib import Path

# ───── ANSI 색 ─────
class Palette:
    RED = "\033[31m"; GRN = "\033[32m"; DIM = "\033[2m"; RST = "\033[0m"
    @staticmethod
    def strip():  # unset color
        Palette.RED = Palette.GRN = Palette.DIM = Palette.RST = ""

# ───── parse header/pass name ─────
HEADER_RE     = re.compile(r"^\s*//.*?IR\ Dump\ (?:Before|After)\b.*")
PASS_NAME_RE  = re.compile(r"IR Dump (?:Before|After)\s+(.*?)\s*(?:\(|$)")
strip_indent  = str.lstrip  # normalizer for comparison

def pass_name(line: str) -> str:
    m = PASS_NAME_RE.search(line)
    return (m.group(1).strip() if m else line.strip()) or "initial"

def parse_blocks(path: Path):
    blocks, header, buf = [], None, []
    for raw in path.read_text(encoding="utf8", errors="replace").splitlines():
        if HEADER_RE.match(raw):
            if header is not None:
                blocks.append((header, buf))
            header, buf = pass_name(raw), []
        else:
            buf.append(raw)
    if header is not None:
        blocks.append((header, buf))
    return blocks

# ───── diff calculation + two formats(terminal, patch) generation ─────
def generate_diff(a_name, a_lines, b_name, b_lines, context=3):
    """returns (pretty_lines, patch_chunks)"""
    a_key = [strip_indent(l) for l in a_lines]
    b_key = [strip_indent(l) for l in b_lines]
    sm    = difflib.SequenceMatcher(None, a_key, b_key)

    pretty, patch = [], []
    # unified-diff header
    patch.append(f"--- {a_name}")
    patch.append(f"+++ {b_name}")

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        # ---- patch format ----
        old_span = f"{i1+1},{i2-i1}"
        new_span = f"{j1+1},{j2-j1}"
        if tag != "equal":
            patch.append(f"@@ -{old_span} +{new_span} @@")
        if tag in ("equal", "replace", "delete"):
            for l in a_lines[i1:i2]:
                if tag == "equal":
                    patch.append(f" {l}")
                else:
                    patch.append(f"-{l}")
        if tag in ("replace", "insert"):
            for l in b_lines[j1:j2]:
                patch.append(f"+{l}")

        # ---- terminal pretty ----
        if tag == "equal":
            for l in a_lines[i1:i2][:context]:
                pretty.append(f"{Palette.DIM} {l}{Palette.RST}")
            if i2 - i1 > 2*context:
                pretty.append(f"{Palette.DIM} …{Palette.RST}")
            for l in a_lines[i2-context:i2]:
                pretty.append(f"{Palette.DIM} {l}{Palette.RST}")
        else:
            if tag in ("replace", "delete"):
                for l in a_lines[i1:i2]:
                    pretty.append(f"{Palette.RED}-{l}{Palette.RST}")
            if tag in ("replace", "insert"):
                for l in b_lines[j1:j2]:
                    pretty.append(f"{Palette.GRN}+{l}{Palette.RST}")

    return pretty, patch

def compare_blocks(blocks, patch_path: Path|None, patch_only: bool = False):
    if len(blocks) < 2:
        print("less than two blocks - no comparison")
        return

    patch_lines: list[str] = []
    for (h1, b1), (h2, b2) in zip(blocks, blocks[1:]):
        pretty, patch = generate_diff(h1, b1, h2, b2)
        if not patch_only:
            print(f"\n=== {h1}  →  {h2} ===")
            print("\n".join(pretty))

        # patch file construction
        patch_lines.append(f"diff --git a/{h1} b/{h2}")
        patch_lines.extend(patch)
        patch_lines.append("")  # end of line

    if patch_path:
        patch_path.write_text("\n".join(patch_lines), encoding="utf8")
        # print(f"diff file saved: {patch_path}")

# ────── CLI ──────
def main():
    ap = argparse.ArgumentParser(
        description="ignore indent MLIR pass diff + VSCode-friendly patch"
    )
    ap.add_argument("dump", help="compile_dump.mlir")
    ap.add_argument("--patch", metavar="FILE",
                    help="save diff in unified-diff format to FILE (.patch/.diff recommended)")
    ap.add_argument("--no-color", action="store_true",
                    help="disable terminal color")
    ap.add_argument("--patch-only", default=False, action="store_true",
                    help="generate patch file only")
    args = ap.parse_args()

    if args.no_color or not sys.stdout.isatty():
        Palette.strip()

    compare_blocks(parse_blocks(Path(args.dump)),
                   Path(args.patch) if args.patch else None,
                   args.patch_only)

if __name__ == "__main__":
    main()
