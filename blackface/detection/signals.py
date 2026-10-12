"""Indices de détection et leur combinaison en un verdict unique."""

from __future__ import annotations

from dataclasses import dataclass, field

# Probabilité retenue quand les métadonnées déclarent explicitement une génération IA.
DECLARED_AI_PROBABILITY = 0.98


@dataclass
class Signal:
    """Un indice : probabilité que le contenu soit généré par IA (0 à 1)."""

    name: str
    probability: float | None
    weight: float
    detail: str
    error: str | None = None


@dataclass
class DetectionResult:
    media_type: str
    signals: list[Signal] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    extracted_text: str | None = None

    @property
    def probability(self) -> float | None:
        return combine(self.signals)

    @property
    def verdict(self) -> str:
        return verdict_label(self.probability)


def combine(signals: list[Signal]) -> float | None:
    """Moyenne pondérée des indices disponibles; une déclaration IA explicite l'emporte."""
    usable = [s for s in signals if s.probability is not None and s.weight > 0]
    if not usable:
        return None
    if any(s.probability >= DECLARED_AI_PROBABILITY for s in usable if s.name == "Métadonnées"):
        return DECLARED_AI_PROBABILITY
    total = sum(s.weight for s in usable)
    return sum(s.probability * s.weight for s in usable) / total


def verdict_label(probability: float | None) -> str:
    if probability is None:
        return "Indéterminé"
    if probability >= 0.8:
        return "Très probablement généré par IA"
    if probability >= 0.6:
        return "Probablement généré par IA"
    if probability > 0.4:
        return "Incertain"
    if probability > 0.2:
        return "Probablement authentique"
    return "Très probablement authentique"
