from collections import defaultdict

from dialect_probing.written_standard import LEXICONS

CROSS = {("sjøl", "selv"), ("heilt", "helt"), ("heile", "hele"), ("òg", "også"), ("ifra", "fra"), ("sia", "siden"),
         ("anna", "annet"), ("trondhjem", "trondheim"), ("ikkje", "ikke"), ("noko", "noe"), ("mykje", "mye"),
         ("berre", "bare"), ("heim", "hjem"), ("heime", "hjemme"), ("meir", "mer"), ("kva", "hva"), ("frå", "fra"),
         ("no", "nå"), ("dei", "de"), ("ho", "hun"), ("eg", "jeg"), ("ein", "en"), ("eit", "et"), ("vore", "vært"),
         ("vart", "ble"), ("sjå", "se"), ("kjem", "kommer"), ("korleis", "hvordan"), ("kor", "hvor")}

def load_lexicon(ordbank_dir) -> tuple[dict, dict]:
    forms, tags = defaultdict(set), defaultdict(set)
    with open(ordbank_dir / LEXICONS["nob"], encoding="latin-1") as f:
        next(f)
        for line in f:
            c = line.split("\t")
            if len(c) > 3:
                w = c[2].lower()
                forms[w].add((c[1], c[3].replace(" normert", "").strip()))
                tags[w].add(c[3].split()[0])
    return forms, tags

def same_variant(forms: dict, ref: str, hyp: str) -> bool:
    return bool(forms.get(ref, set()) & forms.get(hyp, set())) or (ref, hyp) in CROSS
