"""Vérification des faits : affirmations → recherche → comparaison des sources par le modèle NLI."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Protocol, Sequence

from ..language import detect_language, overlap
from .claims import extract_claims, search_query, split_sentences
from .search import Evidence, MultiSearch
from .sources import CATEGORY_WEIGHTS

VERDICT_LABELS: dict[str, str] = {
    "VRAI": "Vrai",
    "PLUTOT_VRAI": "Plutôt vrai",
    "TROMPEUR": "Trompeur",
    "CONTESTE": "Sources divergentes",
    "FAUX": "Faux",
    "NON_VERIFIABLE": "Non vérifiable",
}

# Seuils du modèle NLI et de pertinence.
STANCE_THRESHOLD = 0.6
MIN_RELEVANCE = 0.3
RATED_RELEVANCE = 0.4
RATING_STRENGTH = 0.9
# Nombre maximal de sources examinées par le modèle pour chaque affirmation (vitesse).
MAX_SCORED_SOURCES = 4

FALSE_RATINGS = re.compile(
    r"\b(faux|fausse|false|fake|intox|infond|inexact|incorrect|wrong|pants on fire|erron|bidon|"
    r"canular|hoax|fabricated|invent)", re.IGNORECASE)
MISLEADING_RATINGS = re.compile(
    r"\b(trompeur|misleading|partly|partiellement|manipul|détourn|hors contexte|out of context|"
    r"exagér|exaggerat|missing context|imprécis)", re.IGNORECASE)
TRUE_RATINGS = re.compile(r"\b(vrai|true|correct|exact|accurate|confirmé)\b", re.IGNORECASE)
# Une source qui *rapporte* une rumeur (pour l'examiner ou la démentir) ne la confirme pas.
RUMOR_MARKERS = re.compile(
    r"(rumeur|rumor|rumour|intox|infox|fake news|complot|conspira|théorie|hoax|canular|vrai ou faux|"
    r"fact[- ]?check|désinformation|disinformation|misinformation|démenti|\?)", re.IGNORECASE)


class StanceModel(Protocol):
    def predict(self, premises: Sequence[str], hypothesis: str) -> list[tuple[float, float]]: ...


@dataclass
class SourceStance:
    evidence: Evidence
    stance: str  # « confirme », « contredit », « trompeur » ou « neutre »
    strength: float
    relevance: float
    passage: str = ""  # phrase de la source la plus proche de l'affirmation


@dataclass
class ClaimResult:
    claim: str
    query: str
    verdict: str = "NON_VERIFIABLE"
    confidence: str = "faible"
    explanation: str = ""
    answer: str = ""  # réponse rédigée : verdict + ce que disent réellement les sources
    sources: list[SourceStance] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return VERDICT_LABELS[self.verdict]


@dataclass
class FactCheckReport:
    claims: list[ClaimResult] = field(default_factory=list)
    error: str | None = None

    @property
    def summary(self) -> str:
        verdicts = [c.verdict for c in self.claims]
        if not verdicts:
            return "Aucune affirmation n'a pu être vérifiée."
        if "FAUX" in verdicts:
            return "Le contenu contient au moins une information démentie par des sources fiables."
        if "TROMPEUR" in verdicts:
            return "Le contenu contient des informations jugées trompeuses par des fact-checkeurs."
        if "CONTESTE" in verdicts:
            return "Les sources fiables divergent sur certaines affirmations."
        if all(v == "NON_VERIFIABLE" for v in verdicts):
            return "Les sources fiables consultées ne permettent pas de vérifier ce contenu."
        if "NON_VERIFIABLE" in verdicts:
            return "Les affirmations vérifiables sont confirmées, d'autres restent non vérifiées."
        return "Les affirmations sont confirmées par des sources fiables."

    @property
    def answer(self) -> str:
        """Réponse rédigée pour l'ensemble du contenu, affirmation par affirmation."""
        if len(self.claims) == 1:
            return self.claims[0].answer
        return "\n\n".join(f"« {c.claim} » — {c.answer}" for c in self.claims)

    def as_dict(self) -> dict:
        return {"resume": self.summary, "reponse": self.answer, "erreur": self.error,
                "affirmations": [asdict(c) for c in self.claims]}


class FactChecker:
    def __init__(self, search: MultiSearch, model_factory) -> None:
        self.search = search
        self.model_factory = model_factory  # fabrique : le modèle NLI n'est chargé qu'au besoin

    def check(self, content: str, context: str | None = None) -> FactCheckReport:
        claims = extract_claims(content, context)
        if not claims:
            return FactCheckReport(error=(
                "Aucune phrase factuelle vérifiable n'a été repérée. "
                "Saisissez l'affirmation précise dans le champ prévu."
            ))
        return FactCheckReport(claims=[self.check_claim(claim) for claim in claims])

    def check_claim(self, claim: str) -> ClaimResult:
        language = detect_language(claim)
        result = ClaimResult(claim=claim, query=search_query(claim))
        evidence, result.warnings = self.search.search(claim, language)
        for index, item in enumerate(evidence, start=1):
            item.id = f"S{index}"
        if evidence:
            result.sources = self.assess_sources(claim, evidence)
        decide(result)
        return result

    def assess_sources(self, claim: str, evidence: list[Evidence]) -> list[SourceStance]:
        premises = [key_passage(claim, item) for item in evidence]
        relevances = [overlap(claim, premise) for premise in premises]
        # Les verdicts de fact-checkeurs sur la même affirmation sont gardés : ils ne coûtent rien.
        rated = [i for i, item in enumerate(evidence) if item.rating and relevances[i] >= RATED_RELEVANCE]
        # Le modèle (lent sur processeur) n'examine que les sources les plus pertinentes.
        candidates = [i for i, r in enumerate(relevances) if r >= MIN_RELEVANCE and i not in rated]
        to_score = sorted(candidates, key=lambda i: relevances[i], reverse=True)[:MAX_SCORED_SOURCES]
        scores = dict(zip(to_score, self.model_factory().predict([premises[i] for i in to_score], claim))) if to_score else {}
        stances = []
        for index in sorted(rated + to_score):
            item = evidence[index]
            relevance = relevances[index]
            entail, contra = scores.get(index, (0.0, 0.0))
            stance, strength = "neutre", 0.0
            if item.rating and relevance >= RATED_RELEVANCE:
                # Verdict explicite d'un fact-checkeur sur la même affirmation.
                if MISLEADING_RATINGS.search(item.rating):
                    stance, strength = "trompeur", RATING_STRENGTH
                elif FALSE_RATINGS.search(item.rating):
                    stance, strength = "contredit", RATING_STRENGTH
                elif TRUE_RATINGS.search(item.rating):
                    stance, strength = "confirme", RATING_STRENGTH
            if stance == "neutre" and relevance >= MIN_RELEVANCE:
                if entail >= STANCE_THRESHOLD and entail > contra and not RUMOR_MARKERS.search(premises[index]):
                    stance, strength = "confirme", entail
                elif contra >= STANCE_THRESHOLD and contra > entail:
                    stance, strength = "contredit", contra
            passage = best_sentence(claim, item, stance)
            stances.append(SourceStance(item, stance, round(strength, 3), round(relevance, 2), passage))
        return stances


CAPTION = re.compile(r"^(Image|Photo|Capture d'écran|Screenshot|Crédit|Credit)\b|\((AFP|Reuters|AP)\s*/|©|Getty Images", re.IGNORECASE)
NEGATION = re.compile(
    r"(\bne\b|\bn['’]|\bpas\b|\baucun|\bjamais\b|\bnon\b|\bfaux\b|\bfausse|\bdément|\bnot\b|\bno\b|\bnever\b|\bfalse\b|\bdenied\b)",
    re.IGNORECASE)
SITE_SUFFIX = re.compile(r"\s+[-|–—]\s+[^-|–—]{2,40}$")


def best_sentence(claim: str, item: Evidence, stance: str = "neutre", max_chars: int = 220) -> str:
    """Phrase de la source qui répond le plus directement à l'affirmation.

    Pour une source qui contredit, on cherche la phrase de démenti (et non celle qui reformule la rumeur) ;
    le titre de l'article est candidat, car il résume souvent le mieux sa conclusion.
    """
    if item.rating:
        return f"{item.title} (verdict : {item.rating})"[:max_chars]
    # Les extraits web contiennent des légendes, crédits photo et coupures (« ... ») : on les écarte.
    pieces = [p.strip() for chunk in re.split(r"\s*(?:\.\.\.|…|\n|\|)\s*", item.excerpt) for p in split_sentences(chunk)]
    title = SITE_SUFFIX.sub("", item.title).strip()
    # Un titre Wikipédia n'est qu'un nom d'article, pas une phrase.
    titles = [] if item.category == "Encyclopédie" else [title]
    candidates = [
        p for p in [*titles, *pieces]
        if len(p.split()) >= 5 and not CAPTION.search(p) and (p[0].isupper() or p[0].isdigit() or p[0] in "«\"“")
    ] or [title]

    def score(sentence: str) -> float:
        value = overlap(claim, sentence)
        negated = bool(NEGATION.search(sentence))
        if stance == "contredit":
            value += 0.5 if negated else 0.0
            value -= 0.3 if re.search(r"[\"«“]", sentence) else 0.0  # citation probable de la rumeur
        elif stance == "confirme":
            value -= 0.3 if negated else 0.0
        return value - len(sentence) / 5000  # à score égal, la phrase la plus courte

    sentence = max(candidates, key=score)
    if sentence == title and not sentence.endswith((".", "!", "?")):
        sentence = sentence[0].upper() + sentence[1:]
    # Restes de mise en page Wikipédia : légende d'image en tête, prononciation phonétique.
    sentence = re.sub(rf"^(Signature|Logo|Drapeau|Blason) d[eu']\s*{re.escape(item.title)}\s+", "", sentence)
    sentence = re.sub(r"\s*\(/[^)]*/\)|\s*\[[^\]]*\]", "", sentence).strip()
    return sentence if len(sentence) <= max_chars else sentence[: max_chars - 1].rsplit(" ", 1)[0] + "…"


def key_passage(claim: str, item: Evidence, max_sentences: int = 2, max_chars: int = 400) -> str:
    """Titre + phrases de l'extrait les plus proches de l'affirmation (texte court = plus rapide et plus précis)."""
    sentences = split_sentences(item.excerpt)
    if item.rating or len(sentences) <= max_sentences:
        return f"{item.title}. {item.excerpt}"[:max_chars]
    ranked = sorted(range(len(sentences)), key=lambda i: overlap(claim, sentences[i]), reverse=True)
    keep = set(ranked[:max_sentences])
    if item.category == "Encyclopédie":
        keep.add(0)  # la première phrase d'un article définit son sujet
    return (f"{item.title}. " + " ".join(sentences[i] for i in sorted(keep)))[:max_chars]


def decide(result: ClaimResult) -> None:
    """Pondère les sources par catégorie et en déduit verdict, confiance et explication."""
    totals = {"confirme": 0.0, "contredit": 0.0, "trompeur": 0.0}
    for source in result.sources:
        if source.stance in totals:
            totals[source.stance] += source.strength * CATEGORY_WEIGHTS.get(source.evidence.category, 1.0)
    support, against, misleading = totals["confirme"], totals["contredit"], totals["trompeur"]
    strongest = max(support, against, misleading)

    if strongest < 0.5:
        result.verdict = "NON_VERIFIABLE"
    elif misleading >= max(support, against):
        result.verdict = "TROMPEUR"
    elif against >= 2 * support:
        result.verdict = "FAUX"
    elif support >= 2 * against:
        result.verdict = "VRAI" if support >= 1.5 else "PLUTOT_VRAI"
    else:
        result.verdict = "CONTESTE"

    others = support + against + misleading - strongest
    if strongest >= 2.5 and strongest >= 3 * max(others, 0.1):
        result.confidence = "elevee"
    elif strongest >= 1.2:
        result.confidence = "moyenne"
    else:
        result.confidence = "faible"
    result.explanation = explain(result)
    result.answer = write_answer(result)


ANSWER_INTROS = {
    "VRAI": "Oui, c'est exact.",
    "PLUTOT_VRAI": "C'est globalement exact.",
    "FAUX": "Non, c'est faux.",
    "TROMPEUR": "C'est trompeur.",
    "CONTESTE": "Les sources fiables ne sont pas d'accord sur ce point.",
    "NON_VERIFIABLE": "Les sources fiables consultées ne permettent ni de confirmer ni de démentir cette affirmation.",
}


def write_answer(result: ClaimResult) -> str:
    """Réponse rédigée : le verdict, puis les phrases exactes des sources qui l'appuient."""

    def ranked(*stances: str) -> list[SourceStance]:
        chosen = [s for s in result.sources if s.stance in stances and s.passage]
        # D'abord la phrase qui répond le plus directement à l'affirmation, puis la source la plus solide.
        return sorted(
            chosen,
            key=lambda s: s.strength * CATEGORY_WEIGHTS.get(s.evidence.category, 1.0) * (0.5 + overlap(result.claim, s.passage)),
            reverse=True,
        )

    def quote(source: SourceStance) -> str:
        return f"« {source.passage.rstrip('.')} » ({source.evidence.domain})"

    parts = [ANSWER_INTROS[result.verdict]]
    if result.verdict in {"VRAI", "PLUTOT_VRAI"}:
        parts.append("Les sources fiables le confirment : " + " ; ".join(quote(s) for s in ranked("confirme")[:2]) + ".")
    elif result.verdict in {"FAUX", "TROMPEUR"}:
        parts.append("Ce que disent les sources fiables : " + " ; ".join(quote(s) for s in ranked("contredit", "trompeur")[:2]) + ".")
    elif result.verdict == "CONTESTE":
        parts.append(f"D'un côté : {quote(ranked('confirme')[0])}. De l'autre : {quote(ranked('contredit', 'trompeur')[0])}.")
    else:
        closest = sorted((s for s in result.sources if s.passage), key=lambda s: s.relevance, reverse=True)
        if closest:
            parts.append(f"L'information la plus proche trouvée : {quote(closest[0])}.")
    if result.confidence == "faible" and result.verdict != "NON_VERIFIABLE":
        parts.append("Confiance faible : peu de sources, à recouper.")
    return " ".join(parts)


def explain(result: ClaimResult) -> str:
    groups: dict[str, list[str]] = {"confirme": [], "contredit": [], "trompeur": []}
    for source in result.sources:
        if source.stance in groups:
            groups[source.stance].append(f"{source.evidence.domain} [{source.evidence.id}]")
    parts = []
    if groups["confirme"]:
        parts.append(f"{len(groups['confirme'])} source(s) confirment : " + ", ".join(groups["confirme"]))
    if groups["contredit"]:
        parts.append(f"{len(groups['contredit'])} source(s) contredisent : " + ", ".join(groups["contredit"]))
    if groups["trompeur"]:
        parts.append(f"{len(groups['trompeur'])} fact-checkeur(s) jugent l'affirmation trompeuse : " + ", ".join(groups["trompeur"]))
    if not parts:
        if result.sources:
            return f"{len(result.sources)} source(s) fiable(s) consultée(s), aucune ne confirme ni ne contredit clairement l'affirmation."
        return "Aucune source fiable ne traite de cette affirmation."
    return ". ".join(parts) + "."
