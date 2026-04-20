"""
Build a word-level N-gram language model from the Speech Course Syllabus.
Stored as log-probabilities with Laplace (add-k) smoothing. Small, pickle-able.

Usage:
    python ngram_lm.py build --corpus syllabus.txt --out checkpoints/lm.pkl
    python ngram_lm.py score --lm  checkpoints/lm.pkl --text "mfcc cepstrum"
"""
from __future__ import annotations

import argparse
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path


def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u0900-\u097f'\- ]+", " ", text)  # keep Devanagari too
    return [t for t in text.split() if t]


class NgramLM:
    def __init__(self, n: int = 3, k: float = 0.1):
        self.n = n
        self.k = k                        # add-k smoothing
        self.ngram_counts = [Counter() for _ in range(n + 1)]
        self.context_counts = [Counter() for _ in range(n + 1)]
        self.vocab = set()

    def fit(self, sentences):
        for toks in sentences:
            toks = ["<s>"] * (self.n - 1) + toks + ["</s>"]
            self.vocab.update(toks)
            for order in range(1, self.n + 1):
                for i in range(len(toks) - order + 1):
                    ng = tuple(toks[i:i + order])
                    self.ngram_counts[order][ng] += 1
                    if order > 1:
                        self.context_counts[order][ng[:-1]] += 1

    def logprob_word(self, word: str, context: tuple) -> float:
        """log P(word | context) with stupid back-off + add-k smoothing."""
        V = max(1, len(self.vocab))
        for order in range(min(self.n, len(context) + 1), 0, -1):
            ctx = tuple(context[-(order - 1):]) if order > 1 else ()
            ng = ctx + (word,)
            num = self.ngram_counts[order][ng]
            den = self.context_counts[order][ctx] if order > 1 \
                  else sum(self.ngram_counts[1].values())
            if num > 0:
                return math.log((num + self.k) / (den + self.k * V))
        # unseen at all orders
        den = sum(self.ngram_counts[1].values()) or 1
        return math.log(self.k / (den + self.k * V))

    def score_sentence(self, toks):
        toks = ["<s>"] * (self.n - 1) + toks + ["</s>"]
        total = 0.0
        for i in range(self.n - 1, len(toks)):
            total += self.logprob_word(toks[i], tuple(toks[i - self.n + 1:i]))
        return total


def build(corpus_path: Path, out_path: Path, n: int = 3):
    text = corpus_path.read_text(encoding="utf-8")
    sents = [tokenize(s) for s in re.split(r"[\.\n]+", text) if s.strip()]
    lm = NgramLM(n=n)
    lm.fit(sents)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(lm, f)
    print(f"Built {n}-gram LM: vocab={len(lm.vocab)}, "
          f"bigrams={len(lm.ngram_counts[2])}, "
          f"trigrams={len(lm.ngram_counts[3])}. Saved -> {out_path}")


def technical_terms(corpus_path: Path, min_len: int = 4):
    """Extract unique technical terms from the syllabus (lowercased)."""
    text = corpus_path.read_text(encoding="utf-8")
    toks = set(tokenize(text))
    # drop stopwords-like short tokens
    return sorted(t for t in toks if len(t) >= min_len and t.isascii())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--corpus", required=True)
    b.add_argument("--out", default="checkpoints/lm.pkl")
    b.add_argument("--n", type=int, default=3)

    s = sub.add_parser("score")
    s.add_argument("--lm", required=True)
    s.add_argument("--text", required=True)

    t = sub.add_parser("terms")
    t.add_argument("--corpus", required=True)

    args = ap.parse_args()
    if args.cmd == "build":
        build(Path(args.corpus), Path(args.out), args.n)
    elif args.cmd == "score":
        with open(args.lm, "rb") as f:
            lm = pickle.load(f)
        print(lm.score_sentence(tokenize(args.text)))
    elif args.cmd == "terms":
        for t in technical_terms(Path(args.corpus)):
            print(t)
