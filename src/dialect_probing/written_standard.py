from pathlib import Path

LEXICONS = {
    "nob": "20220201_norsk_ordbank_nob_2005/fullformsliste.txt",
    "nno": "20220201_norsk_ordbank_nno_2012/fullformer_2012.txt",
}

def load_forms(path: Path) -> set[str]:
    forms = set()
    with open(path, encoding="latin-1") as f:
        next(f)
        for line in f:
            cols = line.split("\t")
            if len(cols) > 2:
                forms.add(cols[2].strip().lower())
    return forms

def load_lexicons(ordbank_dir: Path) -> tuple[set[str], set[str]]:
    nob = load_forms(Path(ordbank_dir) / LEXICONS["nob"])
    nno = load_forms(Path(ordbank_dir) / LEXICONS["nno"])
    return nob - nno, nno - nob

def classify(words: list[str], nob_only: set[str], nno_only: set[str]) -> tuple[str, int, int]:
    b = sum(w in nob_only for w in words)
    n = sum(w in nno_only for w in words)
    if n > b:
        return "nynorsk", b, n
    if b > n:
        return "bokmål", b, n
    return "uklar", b, n
