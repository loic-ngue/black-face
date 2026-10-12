"""Liste blanche des sources jugées fiables, regroupées par catégorie.

Pour ajouter ou retirer une source, modifier simplement ce dictionnaire.
Un domaine couvre aussi ses sous-domaines (ex. « gouv.fr » couvre « sante.gouv.fr »).
"""

from __future__ import annotations

from urllib.parse import urlsplit

TRUSTED_SOURCES: dict[str, list[str]] = {
    "Fact-checking": [
        "factuel.afp.com",
        "africacheck.org",
        "observers.france24.com",
        "snopes.com",
        "politifact.com",
        "factcheck.org",
        "fullfact.org",
        "leadstories.com",
    ],
    "Agence de presse": [
        "afp.com",
        "reuters.com",
        "apnews.com",
    ],
    "Média de référence": [
        "lemonde.fr",
        "francetvinfo.fr",
        "radiofrance.fr",
        "rfi.fr",
        "france24.com",
        "liberation.fr",
        "lefigaro.fr",
        "lesechos.fr",
        "jeuneafrique.com",
        "bbc.com",
        "bbc.co.uk",
        "theguardian.com",
        "nytimes.com",
        "washingtonpost.com",
        "dw.com",
        "euronews.com",
        "radio-canada.ca",
        "rts.ch",
        "rtbf.be",
    ],
    "Source officielle": [
        "un.org",
        "who.int",
        "europa.eu",
        "worldbank.org",
        "imf.org",
        "gouv.fr",
        "insee.fr",
        "gov.uk",
    ],
    "Science et santé": [
        "nature.com",
        "science.org",
        "pasteur.fr",
        "inserm.fr",
        "nasa.gov",
        "cdc.gov",
        "nih.gov",
    ],
    "Encyclopédie": [
        "wikipedia.org",
    ],
}

# Poids de chaque catégorie dans le jugement final.
CATEGORY_WEIGHTS: dict[str, float] = {
    "Fact-checking": 1.5,
    "Agence de presse": 1.3,
    "Source officielle": 1.3,
    "Science et santé": 1.3,
    "Média de référence": 1.0,
    "Encyclopédie": 0.8,
}


def all_domains() -> list[str]:
    return [domain for domains in TRUSTED_SOURCES.values() for domain in domains]


def category_of(url: str) -> str | None:
    """Catégorie de la source correspondant à l'URL, ou None si non fiable."""
    host = (urlsplit(url).hostname or "").casefold()
    for category, domains in TRUSTED_SOURCES.items():
        for domain in domains:
            if host == domain or host.endswith("." + domain):
                return category
    return None
