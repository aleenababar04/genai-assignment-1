"""Pack the report for Overleaf and check the LaTeX source for common mistakes.

Run from the repository root:

    python scripts/make_overleaf_zip.py

It writes report_overleaf.zip containing only what the report needs
(main.tex, references.bib and the figures that main.tex uses), and prints:
  - figures that main.tex asks for but that do not exist (they would show as
    red placeholder boxes in the PDF);
  - labels that are referenced but never defined, and citations that are not
    in references.bib (these show up as "??" in the PDF);
  - unbalanced \\begin/\\end environments or braces;
  - the number of red [TODO] markers still left.

There is no LaTeX on this machine, so this static check is not a compile: the
real test is to upload the zip to Overleaf (see docs/testing_and_outputs_guide.md).
"""

import io
import re
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
OUT = ROOT / "report_overleaf.zip"


def strip_comments(text):
    """Remove LaTeX comments (a % that is not escaped), line by line."""
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in text.splitlines())


MAX_WIDTH = 1600  # pixels; enough for a full-width figure in a two-column IEEE page


def shrink_png(path):
    """PNG bytes of the image, scaled down to MAX_WIDTH if it is wider (the original file is not changed)."""
    from PIL import Image

    image = Image.open(path)
    if image.width > MAX_WIDTH:
        height = round(image.height * MAX_WIDTH / image.width)
        image = image.resize((MAX_WIDTH, height), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def main():
    source =(REPORT / "main.tex").read_text(encoding="utf-8")
    code = strip_comments(source)
    bib = (REPORT / "references.bib").read_text(encoding="utf-8")

    # ---- files the report uses ----
    figures = set(re.findall(r"\\figOrTodo\{([^}]+)\}", code))
    inputs = {name + ".tex" for name in re.findall(r"\\input\{([^}]+)\}", code)}
    wanted = {f for f in figures | inputs if not f.startswith("#")}
    present = sorted(f for f in wanted if (REPORT / f).exists())
    missing = sorted(f for f in wanted if not (REPORT / f).exists())

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(REPORT / "main.tex", "main.tex")
        z.write(REPORT / "references.bib", "references.bib")
        for f in present:
            if f.endswith(".png"):
                z.writestr(f, shrink_png(REPORT / f))  # smaller images compile faster
            else:
                z.write(REPORT / f, f)
    size_mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT.name}: main.tex, references.bib and {len(present)} figure files, {size_mb:.1f} MB")

    # ---- static checks ----
    problems = 0
    if missing:
        print(f"\nFigures asked for by main.tex that do not exist yet ({len(missing)}; they appear as red boxes):")
        for f in missing:
            print(f"  {f}")

    labels = set(re.findall(r"\\label\{([^}]+)\}", code))
    refs = set()
    for group in re.findall(r"\\(?:ref|eqref|autoref|cref)\{([^}]+)\}", code):
        refs.update(g.strip() for g in group.split(","))
    undefined = sorted(refs - labels)
    if undefined:
        problems += len(undefined)
        print(f"\nReferenced but never defined labels (would print ??): {undefined}")

    keys = set(re.findall(r"@\w+\{([^,\s]+),", bib))
    cited = set()
    for group in re.findall(r"\\cite[a-z]*\{([^}]+)\}", code):
        cited.update(g.strip() for g in group.split(","))
    unknown = sorted(cited - keys)
    if unknown:
        problems += len(unknown)
        print(f"\nCited but not in references.bib (would print [?]): {unknown}")
    unused = sorted(keys - cited)
    if unused:
        print(f"\nIn references.bib but never cited (harmless, not printed): {unused}")

    begins = Counter(re.findall(r"\\begin\{([^}]+)\}", code))
    ends = Counter(re.findall(r"\\end\{([^}]+)\}", code))
    for env in sorted(set(begins) | set(ends)):
        if begins[env] != ends[env]:
            problems += 1
            print(f"\nUnbalanced environment {env}: {begins[env]} begin, {ends[env]} end")

    plain = re.sub(r"\\[{}]", "", code)  # ignore escaped braces
    if plain.count("{") != plain.count("}"):
        problems += 1
        print(f"\nUnbalanced braces: {plain.count('{')} '{{' against {plain.count('}')} '}}'")

    todos = len(re.findall(r"\\TODO\{", code)) - 1  # minus its use inside the \figOrTodo macro
    print(f"\nRed [TODO] markers left in the text: {todos}")
    print("Static check: " + ("no problems found." if problems == 0 else f"{problems} problem(s) above."))


if __name__ == "__main__":
    main()
