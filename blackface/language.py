"""Outils linguistiques simples (français / anglais), sans dépendance externe."""

from __future__ import annotations

import re

STOPWORDS = {
    "fr": set(
        "le la les un une des du de d l au aux et ou mais donc or ni car que qui quoi dont où ce cet cette ces "
        "son sa ses leur leurs mon ma mes ton ta tes notre nos votre vos il elle ils elles on nous vous je tu "
        "se s en y ne pas plus très est sont été être a ont avait avoir fait faire pour par sur sous dans avec "
        "sans entre vers chez comme aussi tout tous toute toutes cela ça c qu n j m t après avant selon".split()
    ),
    "en": set(
        "the a an and or but so of to in on at by for with from as is are was were be been being has have had "
        "do does did this that these those it its he she they them his her their we you i not no very also "
        "all any which who whom whose what when where why how than then there here after before according".split()
    ),
}

WORD_PATTERN = re.compile(r"[\wÀ-ÿ-]+", re.UNICODE)
ELISION = re.compile(r"\b[ldjnmtsc]['’]|\bqu['’]", re.IGNORECASE)
ENTITY_PATTERN = re.compile(r"\b[A-ZÀ-ÖØ-Þ][\wÀ-ÿ-]+(?:\s+(?:de\s+|du\s+|des\s+|d['’])?[A-ZÀ-ÖØ-Þ][\wÀ-ÿ-]+)*")


def words(text: str) -> list[str]:
    """Mots du texte, en séparant les élisions (« l'Allemagne » → « Allemagne »)."""
    return WORD_PATTERN.findall(ELISION.sub(" ", text))


def entities(text: str) -> list[str]:
    """Groupes de mots à majuscule (noms propres probables), sans les mots outils en tête de phrase."""
    stop = STOPWORDS["fr"] | STOPWORDS["en"] | {"selon", "non", "oui"}
    found: list[str] = []
    for match in ENTITY_PATTERN.finditer(ELISION.sub(" ", text)):
        parts = match.group(0).split()
        while parts and parts[0].casefold() in stop:
            parts.pop(0)
        entity = " ".join(parts)
        if entity and entity not in found:
            found.append(entity)
    return found


def detect_language(text: str) -> str:
    """Renvoie « fr », « en » ou « autre » selon la proportion de mots outils."""
    tokens = [w.casefold() for w in WORD_PATTERN.findall(text.replace("'", " ").replace("’", " "))[:2000]]
    if not tokens:
        return "autre"
    scores = {lang: sum(t in stop for t in tokens) / len(tokens) for lang, stop in STOPWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 0.08 else "autre"


def content_words(text: str) -> list[str]:
    """Mots porteurs de sens (sans mots outils), dans l'ordre d'apparition."""
    stop = STOPWORDS["fr"] | STOPWORDS["en"]
    return [w for w in words(text) if w.casefold() not in stop and len(w) > 1]


def overlap(a: str, b: str) -> float:
    """Part des mots porteurs de sens de `a` présents dans `b` (0 à 1)."""
    wa = {w.casefold() for w in content_words(a)}
    wb = {w.casefold() for w in content_words(b)}
    return len(wa & wb) / len(wa) if wa else 0.0
