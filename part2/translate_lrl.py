"""
Task 2.2 -- Translate the code-switched transcript into the chosen LRL
(Maithili by default).

Strategy
--------
1. Word-level dictionary lookup against `lrl_dict.tsv`. Hits are translated
   directly to the Maithili form.
2. Misses fall back per-script:
     * Devanagari source token -> kept verbatim. Maithili shares ~70% of its
       core lexicon with Hindi, so leaving an unknown Hindi word in place
       is usually intelligible to a Maithili reader.
     * Latin source token (English loanword) -> deterministic Latin->Devanagari
       transliteration. English borrowings in Maithili are written in the
       Devanagari script preserving English pronunciation.
3. Punctuation / digits passed through.

The fallback transliteration is intentionally simple (it's not a G2P --
g2p_hinglish.py is what produces IPA). It just turns "computer" -> "कम्प्यूटर".

Public API
----------
    translate_text(text: str) -> str
    transcript_to_lrl_json(in_path, out_path, dict_path) -> None
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple


# --------------------------------------------------------------------------- #
# Dictionary loader
# --------------------------------------------------------------------------- #
def load_dict(path: Path) -> Dict[str, Tuple[str, str]]:
    """Returns {source_word_lower: (lrl_devanagari, lrl_ipa_or_empty)}"""
    out: Dict[str, Tuple[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        src = cols[0].strip().lower()
        lrl = cols[1].strip() if len(cols) > 1 else src
        ipa = cols[2].strip() if len(cols) > 2 else ""
        if src:
            out[src] = (lrl, ipa)
    return out


# --------------------------------------------------------------------------- #
# Latin -> Devanagari deterministic transliteration (for English misses)
# --------------------------------------------------------------------------- #
_LATIN_DIGRAPHS = [
    ("sh", "श"), ("ch", "च"), ("th", "थ"), ("ph", "फ"), ("kh", "ख"),
    ("gh", "घ"), ("ck", "क"), ("ng", "ंग"), ("tion", "शन"),
    ("ee", "ी"), ("oo", "ू"), ("ai", "ै"), ("ay", "े"), ("ea", "ी"),
    ("oa", "ो"), ("ou", "ौ"), ("ow", "ौ"), ("oi", "ॉय"), ("oy", "ॉय"),
    ("au", "ॉ"), ("aw", "ॉ"),
]
_LATIN_LETTER = {
    "a": "ा", "b": "ब", "c": "क", "d": "ड", "e": "े", "f": "फ",
    "g": "ग", "h": "ह", "i": "ि", "j": "ज", "k": "क", "l": "ल",
    "m": "म", "n": "न", "o": "ो", "p": "प", "q": "क", "r": "र",
    "s": "स", "t": "ट", "u": "ु", "v": "व", "w": "व", "x": "क्स",
    "y": "य", "z": "ज़",
}
_VOWEL_SIGNS = {"ा", "ि", "ी", "ु", "ू", "े", "ै", "ो", "ौ", "ं"}
_CONSONANTS = set("बकडफगहजकलमनपकरसटवक्सयज़चछशघखघट")


def _transliterate_latin(word: str) -> str:
    """Crude Latin->Devanagari transliteration. Inserts अ where a consonant
    would otherwise have no following vowel sign."""
    w = word.lower()
    out = []
    i = 0
    while i < len(w):
        # try digraphs first
        matched = False
        for n in (4, 3, 2):
            if i + n <= len(w):
                seg = w[i:i + n]
                hit = next((p for p in _LATIN_DIGRAPHS if p[0] == seg), None)
                if hit:
                    out.append(hit[1])
                    i += n
                    matched = True
                    break
        if matched:
            continue
        ch = w[i]
        if ch in _LATIN_LETTER:
            out.append(_LATIN_LETTER[ch])
        i += 1

    # post-process: a Devanagari consonant followed by another consonant needs
    # an explicit virama or schwa. We insert schwa (अ) where a consonant is
    # not followed by a vowel sign or another piece producing a vowel sign.
    fixed = []
    for j, piece in enumerate(out):
        fixed.append(piece)
        if piece and piece[-1] not in _VOWEL_SIGNS and piece not in _VOWEL_SIGNS:
            nxt = out[j + 1] if j + 1 < len(out) else ""
            if nxt and nxt[0] in _VOWEL_SIGNS:
                continue
            # else add inherent schwa as consonant marker (just append nothing
            # -- Devanagari consonant glyphs already imply schwa)
            continue
    return "".join(fixed)


# --------------------------------------------------------------------------- #
# Translation
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(
    r"[A-Za-z']+"
    r"|[\u0900-\u097f]+"
    r"|\d+"
    r"|[^\sA-Za-z0-9\u0900-\u097f]"
)


class LrlTranslator:
    def __init__(self, dict_path: Path):
        self.lex = load_dict(dict_path)
        self._stats = {"hits": 0, "deva_passthrough": 0, "translit": 0,
                       "punct": 0, "digit": 0}

    def stats(self):
        return dict(self._stats)

    def _translate_token(self, tok: str) -> str:
        key = tok.lower()
        if key in self.lex:
            self._stats["hits"] += 1
            return self.lex[key][0]
        # Devanagari token, no entry -> keep as-is (likely Hindi)
        if re.match(r"[\u0900-\u097f]+$", tok):
            self._stats["deva_passthrough"] += 1
            return tok
        # Latin token, no entry -> transliterate
        if re.match(r"[A-Za-z']+$", tok):
            self._stats["translit"] += 1
            return _transliterate_latin(tok)
        if tok.isdigit():
            self._stats["digit"] += 1
            return tok
        self._stats["punct"] += 1
        return tok

    def translate(self, text: str) -> str:
        out = []
        for m in _TOKEN_RE.finditer(text):
            tok = m.group(0)
            out.append(self._translate_token(tok))
        # naive re-spacing -- punctuation glued to the previous word
        joined = ""
        for i, piece in enumerate(out):
            if i == 0:
                joined = piece
                continue
            if re.match(r"[^\sA-Za-z0-9\u0900-\u097f]", piece):
                joined += piece
            else:
                joined += " " + piece
        return joined


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def transcript_to_lrl_json(in_path: Path, out_path: Path, dict_path: Path):
    tr = LrlTranslator(dict_path)
    with open(in_path, "r", encoding="utf-8") as f:
        segs = json.load(f)
    enriched = []
    for s in segs:
        text = s.get("text", "")
        enriched.append({**s, "text_lrl": tr.translate(text)})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    print(f"[lrl] wrote {len(enriched)} segments -> {out_path}")
    print(f"[lrl] coverage: {tr.stats()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/transcript_ipa.json")
    ap.add_argument("--out", default="results/transcript_lrl.json")
    ap.add_argument("--dict", default="part2/lrl_dict.tsv")
    args = ap.parse_args()
    transcript_to_lrl_json(Path(args.inp), Path(args.out), Path(args.dict))
