#!/usr/bin/env python3
"""Splice-variant generator for Piper calibration data.

Creates calibration sentences from real Vietnamese sentence fragments to reach
the >=200-sample runbook requirement while keeping a diverse length
distribution and realistic Vietnamese phonotactics.
"""

import random


def make_splice_variants(sentences: list, target: int, rng: random.Random) -> list:
    """Create splice variants (concatenate sentence fragments) to reach target count."""
    variants = []
    word_lists = [s.split() for s in sentences if s.split()]

    # 1) Whole sentences first
    pool = list(sentences)
    rng.shuffle(pool)
    variants.extend(pool)

    # 2) Half-sentence splits (long sentences -> 2 shorter ones)
    for s in sentences:
        words = s.split()
        if len(words) >= 8:
            mid = len(words) // 2
            variants.append(" ".join(words[:mid]))
            variants.append(" ".join(words[mid:]))

    # 3) Splice: head of one sentence + tail of another
    attempts = 0
    while len(variants) < target and attempts < target * 40:
        attempts += 1
        a = rng.choice(word_lists)
        b = rng.choice(word_lists)
        if len(a) < 2 or len(b) < 2:
            continue
        cut_a = rng.randint(1, max(1, len(a) - 1))
        cut_b = rng.randint(1, max(1, len(b) - 1))
        frag = a[:cut_a] + b[-cut_b:]
        if len(frag) >= 4:
            new_s = " ".join(frag)
            if new_s not in variants:
                variants.append(new_s)

    # Dedupe + cap
    seen = set()
    out = []
    for s in variants:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:target]
