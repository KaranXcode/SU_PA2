"""
Task 2.1 -- Hinglish (English-Hindi code-switched) text -> unified IPA string.

Why a custom layer is required
------------------------------
Stock G2P tools fail on code-switched input because:
  * English G2P (CMU dict, eng_to_ipa) only knows Latin script. On a
    Devanagari token they either crash or pass it through verbatim.
  * Hindi G2P (epitran, indic_g2p) only knows Devanagari. On a Latin token
    they do the same.
  * Indian English (L2) has phonological substitutions that no native-English
    G2P captures (the assignment explicitly calls this out -- "(L2)").

3-stage pipeline
----------------
  1. Tokenize, keeping per-token script identity (latin / devanagari / other).
  2. Per-token G2P routed by script:
        * Devanagari -> rule-based grapheme->IPA in pure Python (this file)
        * Latin      -> a small letter-rule G2P (no external dep) with
                        optional `eng_to_ipa` upgrade if installed.
  3. Apply the L2-Hinglish substitution layer to the IPA stream:
        /θ/ -> /t̪/    (no English dental fricative for Indian speakers)
        /ð/ -> /d̪/
        /v/ and /w/ -> /ʋ/   (single labiodental approximant)
        /ɹ/ -> /r/
        alveolar /t/, /d/ -> retroflex /ʈ/, /ɖ/
        long /aː/ retained (Hindi vowel system)
        schwa-deletion at word-final positions in Devanagari (Hindi rule)

Public API
----------
    text_to_ipa(text: str, lang: str) -> str
    transcript_to_ipa_json(in_path, out_path) -> None
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple


# --------------------------------------------------------------------------- #
# Devanagari -> IPA (rule-based, pure Python)
# --------------------------------------------------------------------------- #
_DEV_VOWELS = {
    "अ": "ə", "आ": "aː", "इ": "ɪ", "ई": "iː",
    "उ": "ʊ", "ऊ": "uː", "ऋ": "r̩",
    "ए": "eː", "ऐ": "ɛː", "ओ": "oː", "औ": "ɔː",
    "ऍ": "æ",  "ऑ": "ɒ",   # English-borrowed vowels
}
_DEV_VOWEL_SIGNS = {
    "ा": "aː", "ि": "ɪ",  "ी": "iː", "ु": "ʊ", "ू": "uː",
    "ृ": "r̩", "े": "eː", "ै": "ɛː", "ो": "oː", "ौ": "ɔː",
    "ॅ": "æ",  "ॉ": "ɒ",
}
_DEV_CONSONANTS = {
    # velar
    "क": "k",  "ख": "kʰ", "ग": "ɡ",  "घ": "ɡʱ", "ङ": "ŋ",
    # palatal
    "च": "tʃ", "छ": "tʃʰ","ज": "dʒ", "झ": "dʒʱ","ञ": "ɲ",
    # retroflex
    "ट": "ʈ",  "ठ": "ʈʰ", "ड": "ɖ",  "ढ": "ɖʱ", "ण": "ɳ",
    # dental
    "त": "t̪", "थ": "t̪ʰ","द": "d̪",  "ध": "d̪ʱ","न": "n",
    # labial
    "प": "p",  "फ": "pʰ", "ब": "b",  "भ": "bʱ", "म": "m",
    # approximants / fricatives
    "य": "j",  "र": "r",  "ल": "l",  "व": "ʋ",
    "श": "ʃ",  "ष": "ʂ",  "स": "s",  "ह": "ɦ",
    # nukta-modified (loan consonants)
    "क़": "q",  "ख़": "x",  "ग़": "ɣ",  "ज़": "z",  "ड़": "ɽ",  "ढ़": "ɽʱ",
    "फ़": "f",
}
_VIRAMA = "्"
_NUKTA = "़"
_ANUSVARA = "ं"        # nasalises preceding vowel
_VISARGA = "ः"         # /ɦ/-ish offglide
_CHANDRABINDU = "ँ"


def _devanagari_to_ipa(word: str) -> str:
    """Convert one Devanagari word to IPA. Implements:
       - inherent schwa after each consonant unless suppressed by virama
       - vowel-sign attachment
       - anusvara nasalisation
       - Hindi schwa deletion at word-final position only (conservative)."""
    # Pre-merge nukta combos (e.g. क + ़ -> क़)
    word = word.replace("क" + _NUKTA, "क़").replace("ख" + _NUKTA, "ख़")
    word = word.replace("ग" + _NUKTA, "ग़").replace("ज" + _NUKTA, "ज़")
    word = word.replace("ड" + _NUKTA, "ड़").replace("ढ" + _NUKTA, "ढ़")
    word = word.replace("फ" + _NUKTA, "फ़")

    out: List[str] = []
    i = 0
    while i < len(word):
        ch = word[i]
        nxt = word[i + 1] if i + 1 < len(word) else ""

        if ch in _DEV_CONSONANTS:
            out.append(_DEV_CONSONANTS[ch])
            if nxt == _VIRAMA:
                i += 2
                continue
            if nxt in _DEV_VOWEL_SIGNS:
                out.append(_DEV_VOWEL_SIGNS[nxt])
                i += 2
                continue
            if nxt == _ANUSVARA:
                out.append("ə̃")
                i += 2
                continue
            # else inherent schwa
            out.append("ə")
            i += 1
            continue

        if ch in _DEV_VOWELS:
            out.append(_DEV_VOWELS[ch])
            if nxt == _ANUSVARA:
                out[-1] = out[-1] + "\u0303"
                i += 2
                continue
            i += 1
            continue

        if ch == _ANUSVARA:
            if out:
                out[-1] = out[-1] + "\u0303"
            i += 1
            continue
        if ch == _VISARGA:
            out.append("ɦ")
            i += 1
            continue
        if ch == _CHANDRABINDU:
            if out:
                out[-1] = out[-1] + "\u0303"
            i += 1
            continue

        # punctuation / unknown -> drop
        i += 1

    # Hindi schwa-deletion rule: drop a final 'ə' if last segment is a schwa
    # following a non-cluster consonant (conservative: just trim trailing 'ə').
    if out and out[-1] == "ə":
        out.pop()
    return "".join(out)


# --------------------------------------------------------------------------- #
# English -> IPA
# --------------------------------------------------------------------------- #
def _try_external_eng_g2p(word: str) -> str | None:
    try:
        import eng_to_ipa as ipa            # type: ignore
    except ImportError:
        return None
    res = ipa.convert(word, keep_punct=False)
    if "*" in res:                          # eng_to_ipa marks unknowns with *
        return None
    return res.replace(" ", "")


# Tiny letter-rule G2P used when CMU/eng_to_ipa is unavailable. Far from
# perfect; works well enough for word-level features that the L2 substitution
# layer will further normalise.
_EN_DIGRAPHS = [
    ("sh", "ʃ"), ("ch", "tʃ"), ("th", "θ"), ("ph", "f"), ("ck", "k"),
    ("ng", "ŋ"), ("qu", "kw"), ("ee", "iː"), ("oo", "uː"), ("ai", "eɪ"),
    ("ay", "eɪ"), ("ea", "iː"), ("oa", "oʊ"), ("ou", "aʊ"), ("ow", "aʊ"),
    ("oi", "ɔɪ"), ("oy", "ɔɪ"), ("au", "ɔː"), ("aw", "ɔː"),
]
_EN_LETTER = {
    "a": "æ", "b": "b", "c": "k", "d": "d", "e": "ɛ", "f": "f",
    "g": "ɡ", "h": "h", "i": "ɪ", "j": "dʒ", "k": "k", "l": "l",
    "m": "m", "n": "n", "o": "ɒ", "p": "p", "q": "k", "r": "ɹ",
    "s": "s", "t": "t", "u": "ʌ", "v": "v", "w": "w", "x": "ks",
    "y": "j", "z": "z",
}


def _letter_rule_g2p(word: str) -> str:
    w = word.lower()
    out: List[str] = []
    i = 0
    while i < len(w):
        if i + 1 < len(w):
            two = w[i:i + 2]
            hit = next((p for p in _EN_DIGRAPHS if p[0] == two), None)
            if hit is not None:
                out.append(hit[1])
                i += 2
                continue
        ch = w[i]
        if ch in _EN_LETTER:
            out.append(_EN_LETTER[ch])
        i += 1
    # silent-e rule: trailing 'e' often silent and lengthens previous vowel
    if w.endswith("e") and len(out) >= 2:
        # crude lengthening
        out[-1] = out[-1] if out[-1].endswith("ː") else out[-1] + "ː"
    return "".join(out)


def _english_to_ipa(word: str) -> str:
    ipa = _try_external_eng_g2p(word)
    return ipa if ipa is not None else _letter_rule_g2p(word)


# --------------------------------------------------------------------------- #
# L2 (Indian-English) substitution layer
# --------------------------------------------------------------------------- #
_L2_SUBS = [
    ("θ",  "t̪"),     # think -> ʈʰɪŋk -> t̪ɪŋk
    ("ð",  "d̪"),
    ("v",  "ʋ"),
    ("w",  "ʋ"),
    ("ɹ",  "r"),
    ("z",  "dʒ"),     # not all Indian Englishes; comment out if undesired
    # alveolar -> retroflex for borrowed English consonants
    ("t",  "ʈ"),
    ("d",  "ɖ"),
    # vowel mappings closer to Hindi inventory
    ("æ",  "ɛ"),
    ("ɒ",  "ɔ"),
    ("ʌ",  "ə"),
    ("ɔɪ", "ɔj"),
    ("eɪ", "eː"),
    ("oʊ", "oː"),
    ("aʊ", "aʊ"),
]


def _apply_l2_substitutions(ipa: str) -> str:
    # Order matters: we re-run the dental subs LAST so the retroflex /t,d/
    # rules above don't clobber the dental fricative outputs.
    out = ipa
    for src, tgt in _L2_SUBS:
        out = out.replace(src, tgt)
    # ensure dental diacritic preserved -- if a 't̪' got mangled by the
    # alveolar /t/ rule, rewrite back. _L2_SUBS order above does this safely:
    # 't' -> 'ʈ' would also touch 't̪' -> 'ʈ̪' (unintended). Fix:
    out = out.replace("ʈ̪", "t̪").replace("ɖ̪", "d̪")
    return out


# --------------------------------------------------------------------------- #
# Tokenisation & top-level
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(
    r"[A-Za-z']+"                       # latin run
    r"|[\u0900-\u097f]+"                # devanagari run
    r"|\d+"
    r"|[^\sA-Za-z0-9\u0900-\u097f]"     # punctuation (kept verbatim)
)


def _tokenise(text: str) -> List[Tuple[str, str]]:
    """Returns [(token, kind)] where kind in {'latin','deva','digit','punct','space'}"""
    toks: List[Tuple[str, str]] = []
    for m in _TOKEN_RE.finditer(text):
        s = m.group(0)
        if re.match(r"[A-Za-z']+$", s):
            toks.append((s, "latin"))
        elif re.match(r"[\u0900-\u097f]+$", s):
            toks.append((s, "deva"))
        elif s.isdigit():
            toks.append((s, "digit"))
        else:
            toks.append((s, "punct"))
    return toks


def text_to_ipa(text: str, lang: str = "auto") -> str:
    """Convert a code-switched line to a unified IPA string. `lang` is a hint
    used to break ties for digits / unknown tokens (numbers spoken in en vs hi
    sound very different)."""
    pieces: List[str] = []
    for tok, kind in _tokenise(text):
        if kind == "latin":
            ipa = _english_to_ipa(tok)
            ipa = _apply_l2_substitutions(ipa)
            pieces.append(ipa)
        elif kind == "deva":
            pieces.append(_devanagari_to_ipa(tok))
        elif kind == "digit":
            # Spoken-out form is language-dependent; we punt and just keep
            # the digit as a placeholder so downstream MT can re-render it.
            pieces.append(f"[{tok}]")
        elif kind == "punct":
            pieces.append(tok if tok in ".,!?;:" else "")
    return " ".join(p for p in pieces if p)


def transcript_to_ipa_json(in_path: Path, out_path: Path):
    with open(in_path, "r", encoding="utf-8") as f:
        segs = json.load(f)
    enriched = []
    for s in segs:
        ipa = text_to_ipa(s.get("text", ""), s.get("lang", "auto"))
        enriched.append({**s, "ipa": ipa})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    print(f"[g2p] wrote {len(enriched)} segments -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/transcript.json")
    ap.add_argument("--out", default="results/transcript_ipa.json")
    args = ap.parse_args()
    transcript_to_ipa_json(Path(args.inp), Path(args.out))
