import json

import httpx

from blackface.language import detect_language, entities
from blackface.verification.claims import extract_claims, search_query
from blackface.verification.factcheck import FactChecker
from blackface.verification.search import (
    Evidence,
    GoogleFactCheckSearch,
    MultiSearch,
    SearchError,
    TavilySearch,
    WikipediaSearch,
)
from blackface.verification.sources import category_of


def evidence(title="Titre", excerpt="Extrait.", category="Agence de presse", domain="reuters.com", rating=None):
    return Evidence("", title, f"https://www.{domain}/{title}", domain, category, excerpt, rating=rating)


class FakeSearch:
    def __init__(self, items):
        self.items = items

    def search(self, claim, language):
        return [Evidence(**vars(item)) for item in self.items], []


class FakeNLI:
    """Renvoie (confirme, contredit) selon un mot-clé présent dans la source."""

    def __init__(self):
        self.calls = 0

    def predict(self, premises, hypothesis):
        self.calls += len(premises)
        return [(0.95, 0.01) if "CONFIRME" in p else (0.02, 0.97) if "DEMENTI" in p else (0.1, 0.1) for p in premises]


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


# --- Langue et affirmations ---------------------------------------------------------

def test_language_and_entities():
    assert detect_language("Le président a annoncé une réforme de la retraite.") == "fr"
    assert detect_language("The president announced a pension reform.") == "en"
    assert entities("Selon l'AFP, Emmanuel Macron est né à Amiens.") == ["AFP", "Emmanuel Macron", "Amiens"]


def test_claim_extraction_keeps_facts_and_skips_opinions():
    text = (
        "Je pense que ce gouvernement est scandaleux. "
        "Le 12 mars 2026, le ministère de la Santé a annoncé 340 nouveaux cas à Douala. "
        "Quelle honte ? "
        "Selon Reuters, la Banque mondiale a confirmé un prêt de 200 millions de dollars au Cameroun."
    )
    claims = extract_claims(text)
    assert claims == [
        "Le 12 mars 2026, le ministère de la Santé a annoncé 340 nouveaux cas à Douala.",
        "Selon Reuters, la Banque mondiale a confirmé un prêt de 200 millions de dollars au Cameroun.",
    ]
    assert extract_claims(text, context="Paris est en Allemagne.")[0] == "Paris est en Allemagne."
    assert search_query("Paris est la capitale de l'Allemagne.") == "Paris capitale Allemagne"


# --- Sources et recherche -----------------------------------------------------------

def test_category_of_matches_subdomains_only_from_whitelist():
    assert category_of("https://sante.gouv.fr/page") == "Source officielle"
    assert category_of("https://fr.wikipedia.org/wiki/Paris") == "Encyclopédie"
    assert category_of("https://lemonde.fr.evil.com/x") is None


def test_tavily_request_and_whitelist_filtering():
    captured = {}

    def handler(request):
        captured["auth"] = request.headers["Authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": [
            {"title": "AFP", "url": "https://factuel.afp.com/x", "content": "Faux."},
            {"title": "Blog", "url": "https://random-blog.net/x", "content": "..."},
        ]})

    results = TavilySearch("tvly-test", client(handler)).search("La tour Eiffel a brûlé.", "fr")
    assert captured["auth"] == "Bearer tvly-test"
    assert captured["body"]["query"] == "tour Eiffel brûlé"
    assert [r.domain for r in results] == ["factuel.afp.com"]


def test_google_factcheck_parses_ratings():
    def handler(request):
        assert request.url.params["key"] == "g-key"
        return httpx.Response(200, json={"claims": [{"text": "Rowan Atkinson est mort", "claimReview": [{
            "publisher": {"name": "AFP Factuel"}, "url": "https://factuel.afp.com/rowan",
            "title": "Non, Rowan Atkinson n'est pas mort", "textualRating": "Faux",
        }]}]})

    [item] = GoogleFactCheckSearch("g-key", client(handler)).search("Rowan Atkinson est mort.", "fr")
    assert item.rating == "Faux" and item.category == "Fact-checking"


def test_wikipedia_keeps_only_articles_about_the_claim_subjects():
    def handler(request):
        params = request.url.params
        if "titles" in params:
            assert params["titles"] == "Emmanuel Macron"
            return httpx.Response(200, json={"query": {"pages": {"1": {
                "title": "Emmanuel Macron", "fullurl": "https://fr.wikipedia.org/wiki/Emmanuel_Macron",
                "extract": "Emmanuel Macron, né le 21 décembre 1977 à Amiens."}}}})
        return httpx.Response(200, json={"query": {"pages": {
            "2": {"index": 1, "title": "Brigitte Macron", "fullurl": "u2", "extract": "Née en 1953."},
            "3": {"index": 2, "title": "Macron (homonymie)", "fullurl": "u3", "extract": "Macron peut désigner :"},
        }}})

    results = WikipediaSearch(client(handler)).search("Emmanuel Macron est né en 1977.", "fr")
    assert [r.title for r in results] == ["Emmanuel Macron"]


def test_multisearch_reports_provider_errors_and_dedupes():
    class Broken:
        name = "Tavily"

        def search(self, claim, language):
            raise SearchError("Clé Tavily invalide.")

    class Ok:
        name = "Wikipédia"

        def search(self, claim, language):
            return [evidence(), evidence()]

    items, warnings = MultiSearch([Broken(), Ok()]).search("x", "fr")
    assert len(items) == 1 and warnings == ["Tavily : Clé Tavily invalide."]


# --- Jugement -----------------------------------------------------------------------

def test_contradicting_sources_give_false():
    nli = FakeNLI()
    checker = FactChecker(FakeSearch([
        evidence("Rowan Atkinson", "DEMENTI Rowan Atkinson est vivant.", "Fact-checking", "factuel.afp.com"),
        evidence("Rowan Atkinson acteur", "DEMENTI Rowan Atkinson joue au théâtre.", "Média de référence", "lemonde.fr"),
    ]), lambda: nli)
    result = checker.check_claim("Rowan Atkinson est mort.")
    assert result.verdict == "FAUX" and result.label == "Faux"
    assert [s.stance for s in result.sources] == ["contredit", "contredit"]
    assert "2 source(s) contredisent" in result.explanation


def test_irrelevant_sources_are_not_scored():
    nli = FakeNLI()
    checker = FactChecker(FakeSearch([evidence("Bourse", "DEMENTI La Bourse de Paris a clôturé en hausse.")]), lambda: nli)
    result = checker.check_claim("Rowan Atkinson est mort.")
    assert nli.calls == 0
    assert result.verdict == "NON_VERIFIABLE"


def test_factchecker_rating_takes_precedence():
    checker = FactChecker(FakeSearch([evidence(
        "Non, Rowan Atkinson n'est pas mort", "Affirmation examinée : Rowan Atkinson est mort. Verdict : Faux.",
        "Fact-checking", "factuel.afp.com", rating="Faux")]), lambda: FakeNLI())
    result = checker.check_claim("Rowan Atkinson est mort.")
    assert result.sources[0].stance == "contredit"
    assert result.verdict == "FAUX"


def test_misleading_rating_and_divergent_sources():
    checker = FactChecker(FakeSearch([
        evidence("Vaccin Cameroun", "Affirmation : le vaccin est obligatoire au Cameroun.", "Fact-checking", rating="Trompeur"),
    ]), lambda: FakeNLI())
    assert checker.check_claim("Le vaccin est obligatoire au Cameroun.").verdict == "TROMPEUR"

    checker = FactChecker(FakeSearch([
        evidence("Vaccin Cameroun", "CONFIRME vaccin obligatoire Cameroun.", "Agence de presse"),
        evidence("Vaccin Cameroun bis", "DEMENTI vaccin obligatoire Cameroun.", "Agence de presse", "apnews.com"),
    ]), lambda: FakeNLI())
    assert checker.check_claim("Le vaccin est obligatoire au Cameroun.").verdict == "CONTESTE"


def test_no_claims_found():
    report = FactChecker(FakeSearch([]), lambda: FakeNLI()).check("J'adore les mangues.")
    assert report.error and not report.claims


def test_only_the_most_relevant_sources_are_scored():
    nli = FakeNLI()
    items = [evidence(f"Rowan Atkinson {i}", "DEMENTI Rowan Atkinson acteur vivant.", domain=f"site{i}.com") for i in range(7)]
    items.append(evidence("Rowan Atkinson fact", "Affirmation : Rowan Atkinson est mort.", "Fact-checking", rating="Faux"))
    result = FactChecker(FakeSearch(items), lambda: nli).check_claim("Rowan Atkinson est mort.")
    assert nli.calls == 4
    assert len(result.sources) == 5  # 4 sources examinées + le verdict du fact-checkeur


def test_written_answer_quotes_what_sources_say():
    checker = FactChecker(FakeSearch([evidence(
        "Barack Obama", "Biographie. DEMENTI Barack Obama est né le 4 août 1961 à Honolulu, à Hawaï. Il fut président.",
        "Encyclopédie", "fr.wikipedia.org")]), lambda: FakeNLI())
    result = checker.check_claim("Barack Obama est né au Kenya.")
    assert result.answer.startswith("Non, c'est faux. Ce que disent les sources fiables : « DEMENTI Barack Obama est né le 4 août 1961 à Honolulu, à Hawaï » (fr.wikipedia.org).")
    assert "Confiance faible" in result.answer


def test_unverifiable_answer_gives_closest_information():
    checker = FactChecker(FakeSearch([evidence("Rowan Atkinson", "Rowan Atkinson est un acteur britannique né en 1955.")]), lambda: FakeNLI())
    result = checker.check_claim("Rowan Atkinson est mort.")
    assert result.verdict == "NON_VERIFIABLE"
    assert "L'information la plus proche trouvée : « Rowan Atkinson est un acteur britannique né en 1955 »" in result.answer
