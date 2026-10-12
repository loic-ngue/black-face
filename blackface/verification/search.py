"""Recherche gratuite dans des sources fiables : Tavily, Google Fact Check et Wikipédia."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from ..language import content_words, entities
from .claims import search_query
from .sources import all_domains, category_of

TAVILY_URL = "https://api.tavily.com/search"
GOOGLE_FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
USER_AGENT = "BlackFace/1.0 (outil de verification des faits)"


class SearchError(RuntimeError):
    pass


@dataclass
class Evidence:
    id: str
    title: str
    url: str
    domain: str
    category: str
    excerpt: str
    published: str | None = None
    rating: str | None = None  # verdict publié par un fact-checkeur (Google Fact Check)


def _domain(url: str) -> str:
    return (urlsplit(url).hostname or "").removeprefix("www.")


class TavilySearch:
    """Recherche web limitée à la liste blanche (offre gratuite : 1 000 recherches/mois)."""

    name = "Tavily"

    def __init__(self, api_key: str, client: httpx.Client, max_results: int = 6) -> None:
        self.api_key, self.client, self.max_results = api_key, client, max_results

    def search(self, claim: str, language: str) -> list[Evidence]:
        query = search_query(claim)
        response = self.client.post(
            TAVILY_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "query": query,
                "search_depth": "advanced",
                "max_results": self.max_results,
                "include_domains": all_domains(),
                "include_answer": False,
            },
        )
        if response.status_code == 401:
            raise SearchError("Clé Tavily invalide.")
        if response.status_code == 429:
            raise SearchError("Quota Tavily atteint.")
        response.raise_for_status()
        evidence = []
        for item in response.json().get("results", []):
            url = item.get("url") or ""
            category = category_of(url)
            if category is None:  # sécurité : on ne garde que la liste blanche
                continue
            evidence.append(Evidence(
                "", item.get("title") or url, url, _domain(url), category,
                (item.get("content") or "").strip()[:1500], item.get("published_date"),
            ))
        return evidence


class GoogleFactCheckSearch:
    """Verdicts publiés par les fact-checkeurs (AFP, Snopes...) via l'API gratuite de Google."""

    name = "Google Fact Check"

    def __init__(self, api_key: str, client: httpx.Client, max_results: int = 6) -> None:
        self.api_key, self.client, self.max_results = api_key, client, max_results

    def search(self, claim: str, language: str) -> list[Evidence]:
        params = {"query": search_query(claim), "key": self.api_key, "pageSize": self.max_results}
        if language in {"fr", "en"}:
            params["languageCode"] = language
        response = self.client.get(GOOGLE_FACTCHECK_URL, params=params)
        if response.status_code in {400, 403}:
            raise SearchError("Clé Google Fact Check invalide ou API non activée.")
        response.raise_for_status()
        evidence = []
        for claim in response.json().get("claims", []):
            for review in claim.get("claimReview", [])[:1]:
                url = review.get("url") or ""
                publisher = (review.get("publisher") or {}).get("name") or _domain(url)
                rating = review.get("textualRating") or ""
                evidence.append(Evidence(
                    "", review.get("title") or claim.get("text", ""), url, _domain(url) or publisher,
                    "Fact-checking",
                    f"Affirmation examinée : {claim.get('text', '')}. Verdict de {publisher} : {rating}.",
                    review.get("reviewDate"), rating,
                ))
        return evidence


class WikipediaSearch:
    """Résumés d'articles Wikipédia sur les sujets de l'affirmation (gratuit, sans clé)."""

    name = "Wikipédia"
    DISAMBIGUATION = ("peut désigner", "peut faire référence", "may refer to")

    def __init__(self, client: httpx.Client, max_results: int = 4) -> None:
        self.client, self.max_results = client, max_results

    def search(self, claim: str, language: str) -> list[Evidence]:
        lang = language if language in {"fr", "en"} else "fr"
        api = f"https://{lang}.wikipedia.org/w/api.php"
        common = {"action": "query", "format": "json", "prop": "extracts|info", "exintro": 1,
                  "explaintext": 1, "exsentences": 6, "inprop": "url", "redirects": 1}
        pages: list[dict] = []
        # 1. Articles portant exactement le nom des sujets (personnes, lieux, organisations).
        names = entities(claim)[:4]
        if names:
            response = self.client.get(api, params={**common, "titles": "|".join(names)})
            response.raise_for_status()
            pages += list(response.json().get("query", {}).get("pages", {}).values())
        # 2. Recherche plein texte, en ne gardant que les articles dont le titre figure dans l'affirmation.
        response = self.client.get(api, params={**common, "generator": "search", "gsrsearch": search_query(claim), "gsrlimit": 5})
        response.raise_for_status()
        found = sorted(response.json().get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
        pages += [page for page in found if _title_in_claim(page.get("title", ""), claim)]

        evidence, seen = [], set()
        for page in pages:
            extract = (page.get("extract") or "").strip()
            if not extract or page.get("title") in seen or any(m in extract[:300] for m in self.DISAMBIGUATION):
                continue
            seen.add(page["title"])
            evidence.append(Evidence("", page["title"], page.get("fullurl", ""), f"{lang}.wikipedia.org",
                                     "Encyclopédie", extract[:1500], page.get("touched")))
        return evidence[: self.max_results]


def _title_in_claim(title: str, claim: str) -> bool:
    """Vrai si tous les mots du titre (hors parenthèses) apparaissent dans l'affirmation."""
    title_words = {w.casefold() for w in content_words(title.split(" (")[0])}
    claim_words = {w.casefold() for w in content_words(claim)}
    return bool(title_words) and title_words <= claim_words


class MultiSearch:
    """Interroge chaque fournisseur disponible; une panne n'empêche pas les autres."""

    def __init__(self, providers: list) -> None:
        self.providers = providers

    @classmethod
    def from_keys(cls, tavily_key: str | None, google_key: str | None) -> MultiSearch:
        client = httpx.Client(timeout=20, headers={"User-Agent": USER_AGENT})
        providers: list = []
        if google_key:
            providers.append(GoogleFactCheckSearch(google_key, client))
        if tavily_key:
            providers.append(TavilySearch(tavily_key, client))
        providers.append(WikipediaSearch(client))
        return cls(providers)

    def search(self, claim: str, language: str) -> tuple[list[Evidence], list[str]]:
        evidence: list[Evidence] = []
        warnings: list[str] = []
        seen: set[str] = set()
        for provider in self.providers:
            try:
                results = provider.search(claim, language)
            except SearchError as error:
                warnings.append(f"{provider.name} : {error}")
                continue
            except httpx.HTTPError as error:
                warnings.append(f"{provider.name} indisponible ({type(error).__name__}).")
                continue
            for item in results:
                if item.url not in seen:
                    seen.add(item.url)
                    evidence.append(item)
        return evidence, warnings
