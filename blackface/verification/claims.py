"""Repérage des phrases factuelles vérifiables, par règles linguistiques."""

from __future__ import annotations

import re

from ..language import content_words

MAX_CLAIMS = 5
MIN_WORDS, MAX_WORDS = 5, 45

SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[\"«“A-ZÀ-ÖØ-Þ0-9])")
NUMBER = re.compile(r"\d")
MONTHS = re.compile(
    r"\b(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre|"
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche|hier|aujourd'hui)\b",
    re.IGNORECASE,
)
FACT_CUES = re.compile(
    r"\b(selon|a annoncé|annonce|a déclaré|déclare|confirme|a confirmé|est mort|est décédé|a été|ont été|"
    r"a tué|tués|morts|blessés|victimes|interdit|lancé|signé|élu|nommé|arrêté|condamné|découvert|"
    r"according to|announced|said|confirmed|died|killed|dead|banned|launched|signed|elected|arrested)\b",
    re.IGNORECASE,
)
OPINION_CUES = re.compile(
    r"\b(je pense|je crois|à mon avis|selon moi|il faut|il faudrait|devrait|devraient|j'adore|je déteste|"
    r"magnifique|incroyable|horrible|scandaleux|honteux|i think|i believe|in my opinion|should|must|"
    r"amazing|terrible|awesome)\b",
    re.IGNORECASE,
)


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]


def score_sentence(sentence: str) -> int:
    """Plus le score est élevé, plus la phrase ressemble à un fait vérifiable."""
    tokens = sentence.split()
    if not MIN_WORDS <= len(tokens) <= MAX_WORDS or sentence.endswith("?"):
        return 0
    score = 0
    if NUMBER.search(sentence):
        score += 2
    if MONTHS.search(sentence):
        score += 1
    proper_nouns = sum(1 for word in tokens[1:] if word[:1].isupper())
    score += min(proper_nouns, 3)
    if FACT_CUES.search(sentence):
        score += 2
    if OPINION_CUES.search(sentence):
        score -= 4
    return score


def extract_claims(text: str, context: str | None = None) -> list[str]:
    """Jusqu'à MAX_CLAIMS affirmations : le contexte saisi d'abord, puis les phrases les plus factuelles."""
    claims: list[str] = []
    if context and context.strip():
        claims.append(context.strip())
    scored = [(score_sentence(s), index, s) for index, s in enumerate(split_sentences(text or ""))]
    best = sorted((item for item in scored if item[0] >= 3), key=lambda item: (-item[0], item[1]))
    for _score, _index, sentence in sorted(best[: MAX_CLAIMS - len(claims)], key=lambda item: item[1]):
        if sentence not in claims:
            claims.append(sentence)
    return claims[:MAX_CLAIMS]


def search_query(claim: str, max_words: int = 10) -> str:
    """Requête de recherche : les mots porteurs de sens de l'affirmation."""
    return " ".join(content_words(claim)[:max_words]) or claim
