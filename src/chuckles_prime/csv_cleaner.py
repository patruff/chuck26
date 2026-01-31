"""CSV cleaner for human_parodies.csv -- handles 5+ format zones in the raw data."""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path


def _fix_encoding(text: str) -> str:
    """Fix common encoding artifacts in the CSV data."""
    replacements = {
        "A(c)": "\u00e9",   # e-acute (risqué, Pokémon)
        "AY=": "\u00e5",    # a-ring (Skarsgård)
        "A\u00a9": "\u00e9",
        "\u00c3\u00a9": "\u00e9",
        "â€™": "'",
        "â€œ": '"',
        "â€\x9d": '"',
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _strip_quotes_and_whitespace(text: str) -> str:
    """Strip leading/trailing whitespace and outer quotes from a field."""
    text = text.strip()
    # Strip outer triple-quotes first, then double-quotes, then single quotes
    for q in ['"""', '""', '"', "'"]:
        if text.startswith(q) and text.endswith(q) and len(text) > len(q):
            text = text[len(q):-len(q)].strip()
    return text


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace chars to single space."""
    return re.sub(r"\s+", " ", text).strip()


def _clean_field(text: str) -> str:
    """Apply all cleaning steps to a field value."""
    text = _fix_encoding(text)
    text = _strip_quotes_and_whitespace(text)
    # Remove stray ** markdown artifacts
    text = text.replace("**", "")
    text = _normalize_whitespace(text)
    return text


def clean_human_parodies(input_path: str | Path, output_path: str | Path) -> int:
    """Clean human_parodies.csv and write a normalized 3-column CSV.

    Handles 5 format zones found in the raw data:
      Zone 1 (rows 2-13):    Standard 3-column CSV
      Zone 2 (rows 14-24):   Markdown-numbered ``N. **input,output**``
      Zone 3 (rows 25-68):   Slash-separated ``input / output: Explanation: ...``
      Zone 4 (rows 69-~500): Triple-quoted with ``Parody: ""output""**``
      Zone 5 (rows ~501+):   Standard 3-column CSV again

    Args:
        input_path: Path to raw human_parodies.csv.
        output_path: Path to write cleaned CSV.

    Returns:
        Number of unique rows written.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Regex patterns for zone detection
    # Zone 2: After CSV split, first_field looks like "13. **Chinchilla"
    zone2_first_re = re.compile(r"^\d+\.\s*\*\*(.+)$")
    zone2_second_re = re.compile(r"^(.+?)\*\*$")
    zone3_re = re.compile(
        r'^(.+?)\s*/\s*(.+?):\s*(?:Detailed\s+)?Explanation:\s*(.+)',
    )
    zone4_parody_re = re.compile(r'Parody:\s*"*(.+?)"*\**$')
    # For the "Original:" / "Parody" partial rows (498-500 area)
    original_re = re.compile(r'^Original:\s*"*(.+?)"*$')

    seen: set[tuple[str, str]] = set()
    rows: list[tuple[str, str, str]] = []

    raw_text = input_path.read_text(encoding="utf-8", errors="replace")
    lines = raw_text.splitlines()

    for line_num, raw_line in enumerate(lines, start=1):
        # Skip header row
        if line_num == 1:
            continue

        line = raw_line.strip()
        if not line:
            continue

        inp = out = expl = ""

        # --- Try Zone 2: Markdown-numbered rows ---
        # Parse the line as CSV first to get fields
        try:
            fields = list(csv.reader(io.StringIO(raw_line)))[0]
        except (csv.Error, IndexError):
            continue

        if not fields or not fields[0].strip():
            continue

        first_field = fields[0].strip()

        # Zone 2: "13. **Chinchilla" (first field) + "Chintrilla**" (second field)
        # The comma inside **input,output** splits it across two CSV fields
        z2_first_match = zone2_first_re.match(first_field)
        z2_second_match = (
            zone2_second_re.match(fields[1].strip())
            if z2_first_match and len(fields) >= 2
            else None
        )
        if z2_first_match and z2_second_match:
            inp = z2_first_match.group(1).strip()
            out = z2_second_match.group(1).strip()
            expl = " ".join(f.strip() for f in fields[2:] if f.strip())
        # Zone 3: "Chinchilla / Chintrilla: Detailed Explanation: ..."
        elif zone3_re.match(first_field):
            z3_match = zone3_re.match(first_field)
            inp = z3_match.group(1).strip()
            out = z3_match.group(2).strip()
            # Explanation may be split across multiple CSV columns
            expl_parts = [z3_match.group(3).strip()]
            expl_parts.extend(f.strip() for f in fields[1:] if f.strip())
            expl = " ".join(expl_parts)
        # Zone 4: Triple-quoted with Parody format
        elif first_field.startswith('"""') or first_field.startswith('"'):
            # Check for "Original:" format (partial rows ~498-500)
            cleaned_first = _strip_quotes_and_whitespace(first_field)
            orig_match = original_re.match(cleaned_first)
            if orig_match:
                # These are orphan rows with just Original/Parody labels; skip
                continue

            # Standard Zone 4: first field is """input""", second has Parody: ""output""**
            inp = _strip_quotes_and_whitespace(first_field)

            if len(fields) >= 2:
                second_field = fields[1].strip()
                parody_match = zone4_parody_re.search(second_field)
                if parody_match:
                    out = parody_match.group(1).strip()
                else:
                    # Some Zone 4 rows have just the parody text
                    out = _strip_quotes_and_whitespace(second_field)

            if len(fields) >= 3:
                expl = " ".join(f.strip() for f in fields[2:] if f.strip())
        else:
            # Zone 1 / Zone 5: Standard 3-column CSV (fallback)
            if len(fields) >= 3:
                inp = fields[0].strip()
                out = fields[1].strip()
                expl = " ".join(f.strip() for f in fields[2:] if f.strip())
            elif len(fields) == 2:
                inp = fields[0].strip()
                out = fields[1].strip()
            else:
                continue

        # Clean all fields
        inp = _clean_field(inp)
        out = _clean_field(out)
        expl = _clean_field(expl)

        # Strip "Parody" or "Parody:" prefix if it leaked into output
        if out.lower().startswith("parody:"):
            out = out[7:].strip()
        if out.lower().startswith("parody"):
            remainder = out[6:].strip()
            if remainder.startswith(":"):
                out = remainder[1:].strip()

        # Skip if input or output is empty
        if not inp or not out:
            continue

        # Skip rows where input == output (no actual parody)
        if inp.lower() == out.lower():
            continue

        # Dedup by normalized (input, output) pair
        dedup_key = (inp.lower().strip(), out.lower().strip())
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        rows.append((inp, out, expl))

    # Write clean CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["input", "output", "explanation"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        for inp, out, expl in rows:
            writer.writerow({"input": inp, "output": out, "explanation": expl})

    return len(rows)
