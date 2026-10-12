from pathlib import Path

from streamlit.testing.v1 import AppTest

from blackface import report as report_export
from blackface.detection.signals import DetectionResult, Signal
from blackface.verification.factcheck import ClaimResult, FactCheckReport, SourceStance
from blackface.verification.search import Evidence

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def sample_results():
    detection = DetectionResult("image", signals=[
        Signal("Métadonnées", None, 0, "Aucune trace IA."),
        Signal("Modèle de détection d'images", 0.9, 1, "haywoodsloan : 90 %"),
    ])
    source = Evidence("S1", "Non, Rowan Atkinson n'est pas mort", "https://factuel.afp.com/x",
                      "factuel.afp.com", "Fact-checking", "...")
    factcheck = FactCheckReport(claims=[ClaimResult(
        claim="Rowan Atkinson est mort.", query="Rowan Atkinson mort", verdict="FAUX", confidence="moyenne",
        explanation="1 source(s) contredisent : factuel.afp.com [S1].", answer="Non, c'est faux.",
        sources=[SourceStance(source, "contredit", 0.97, 0.67)],
    )])
    report = report_export.build_report(filename="a.png", data=b"abc", detection=detection, factcheck=factcheck)
    return detection, factcheck, report


def test_exports_are_valid_files():
    _, _, report = sample_results()
    assert report["detection_ia"]["verdict"] == "Très probablement généré par IA"
    assert report_export.to_pdf(report).startswith(b"%PDF")
    assert report_export.to_docx(report).startswith(b"PK")
    assert b"FAUX" in report_export.to_json(report)


def test_app_renders_input_screen():
    app = AppTest.from_file(APP).run(timeout=30)
    assert not app.exception
    assert "Analyser" in [b.label for b in app.button]


def test_app_renders_results():
    detection, factcheck, report = sample_results()
    app = AppTest.from_file(APP)
    app.session_state["result"] = {"detection": detection, "factcheck": factcheck, "report": report}
    app.run(timeout=30)
    assert not app.exception
    page = "".join(m.value for m in app.markdown)
    assert "<strong>90%</strong>" in page
    assert "« Rowan Atkinson est mort. »" in page
    assert "<b>Non, c&#x27;est faux.</b>" in page
    assert [b.label for b in app.get("download_button")] == ["PDF", "Word", "JSON"]


def test_ui_escapes_text_from_sources():
    from blackface import ui

    _, factcheck, _ = sample_results()
    claim = factcheck.claims[0]
    claim.claim = "<script>alert(1)</script>"
    claim.sources[0].evidence.title = "<img src=x onerror=alert(1)>"
    claim.sources[0].evidence.url = "javascript:alert(1)"
    markup = ui.claim_card(claim) + ui.sources_list(claim)
    assert "<script>" not in markup and "<img" not in markup
    assert 'href="#"' in markup
