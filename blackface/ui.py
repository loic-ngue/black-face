"""Composants visuels de l'interface (HTML/CSS injectés dans Streamlit).

Tout texte venant de l'utilisateur ou des sources est échappé avec `html.escape`.
"""

from __future__ import annotations

from html import escape

from .detection.signals import DetectionResult
from .verification.factcheck import ClaimResult, FactCheckReport

# Couleurs par verdict : (couleur principale, fond translucide)
VERDICT_STYLES = {
    "VRAI": ("#2FD38A", "rgba(47,211,138,.12)"),
    "PLUTOT_VRAI": ("#7BD88F", "rgba(123,216,143,.12)"),
    "TROMPEUR": ("#F5A524", "rgba(245,165,36,.12)"),
    "CONTESTE": ("#F5A524", "rgba(245,165,36,.12)"),
    "FAUX": ("#FF4D5E", "rgba(255,77,94,.12)"),
    "NON_VERIFIABLE": ("#8A93A6", "rgba(138,147,166,.12)"),
}
STANCE_STYLES = {
    "confirme": ("Confirme", "#2FD38A"),
    "contredit": ("Contredit", "#FF4D5E"),
    "trompeur": ("Trompeur", "#F5A524"),
    "neutre": ("Sans rapport direct", "#8A93A6"),
}
CONFIDENCE_LABELS = {"faible": "Confiance faible", "moyenne": "Confiance moyenne", "elevee": "Confiance élevée"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');
:root { --bf-red:#FF4D5E; --bf-card:#121826; --bf-line:#232B3B; --bf-muted:#8A93A6; --bf-text:#E6E9EF; }
.block-container { padding-top: 2rem; max-width: 1180px; }
[data-testid="stHeader"] { background: transparent; }
h1, h2, h3, .bf-title { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.02em; }

/* En-tête */
.bf-hero { position:relative; overflow:hidden; border:1px solid var(--bf-line); border-radius:20px;
  padding:34px 36px; margin-bottom:22px;
  background: radial-gradient(1200px 300px at 0% 0%, rgba(255,77,94,.18), transparent 60%),
              radial-gradient(800px 260px at 100% 100%, rgba(80,120,255,.14), transparent 60%), #0E131E; }
.bf-hero .bf-logo { display:flex; align-items:center; gap:14px; }
.bf-hero .bf-mark { width:46px; height:46px; border-radius:12px; display:grid; place-items:center;
  background:linear-gradient(135deg,#FF4D5E,#8B1E2B); box-shadow:0 8px 30px rgba(255,77,94,.35); font-size:24px; }
.bf-hero .bf-title { font-size:2.3rem !important; font-weight:700; color:var(--bf-text) !important; margin:0; padding:0 !important; line-height:1; }
.bf-hero .bf-title span { color:var(--bf-red) !important; }
.bf-hero p { color:#B8C0CF; margin:14px 0 18px; font-size:1.02rem; max-width:720px; }
.bf-pills { display:flex; flex-wrap:wrap; gap:8px; }
.bf-pill { font-size:.8rem; padding:6px 12px; border-radius:999px; border:1px solid var(--bf-line);
  background:rgba(255,255,255,.03); color:#C9D1DE; }
.bf-pill b { color:var(--bf-text); }

/* Cartes génériques */
.bf-card { border:1px solid var(--bf-line); border-radius:16px; background:var(--bf-card); padding:20px 22px; margin-bottom:14px; }
.bf-section { display:flex; align-items:center; gap:10px; margin:28px 0 12px; }
.bf-section .bf-num { width:28px; height:28px; border-radius:8px; display:grid; place-items:center; font-weight:700;
  background:rgba(255,77,94,.15); color:var(--bf-red); font-family:'Space Grotesk',sans-serif; }
.bf-section h3 { margin:0 !important; font-size:1.35rem; }

/* Panneau d'état */
.bf-status { display:flex; flex-direction:column; gap:10px; }
.bf-status-row { display:flex; align-items:center; justify-content:space-between; font-size:.9rem; color:#C9D1DE; }
.bf-dot { width:9px; height:9px; border-radius:50%; display:inline-block; margin-right:8px; }
.bf-steps { counter-reset: step; margin:4px 0 0; padding:0; list-style:none; }
.bf-steps li { counter-increment: step; display:flex; gap:12px; margin:12px 0; color:#B8C0CF; font-size:.9rem; line-height:1.4; }
.bf-steps li::before { content: counter(step); flex:0 0 24px; height:24px; border-radius:7px; display:grid; place-items:center;
  background:#1A2233; color:var(--bf-text); font-weight:600; font-size:.8rem; }
.bf-label { text-transform:uppercase; letter-spacing:.08em; font-size:.72rem; color:var(--bf-muted); margin-bottom:10px; font-weight:600; }

/* Jauge de détection IA */
.bf-detect { display:flex; gap:28px; align-items:center; flex-wrap:wrap; }
.bf-gauge { --p:0; --c:#8A93A6; width:150px; height:150px; border-radius:50%; flex:0 0 150px; display:grid; place-items:center;
  background: conic-gradient(var(--c) calc(var(--p)*1%), #1E2636 0); }
.bf-gauge > div { width:118px; height:118px; border-radius:50%; background:var(--bf-card); display:grid; place-items:center; text-align:center; }
.bf-gauge strong { font-family:'Space Grotesk',sans-serif; font-size:2rem; color:var(--bf-text); display:block; line-height:1; }
.bf-gauge span { font-size:.72rem; color:var(--bf-muted); text-transform:uppercase; letter-spacing:.08em; }
.bf-detect-body { flex:1; min-width:260px; }
.bf-verdict { font-family:'Space Grotesk',sans-serif; font-size:1.45rem; font-weight:700; margin:0 0 4px; }
.bf-signal { display:grid; grid-template-columns: 1fr auto; gap:4px 12px; padding:12px 0; border-top:1px solid var(--bf-line); }
.bf-signal:first-of-type { border-top:none; }
.bf-signal .name { color:var(--bf-text); font-weight:600; font-size:.92rem; }
.bf-signal .value { color:var(--bf-text); font-weight:600; font-variant-numeric:tabular-nums; }
.bf-signal .detail { grid-column:1 / -1; color:var(--bf-muted); font-size:.84rem; line-height:1.45; }
.bf-signal .detail.warn { color:#F5A524; }
.bf-bar { grid-column:1 / -1; height:5px; border-radius:99px; background:#1E2636; overflow:hidden; }
.bf-bar > i { display:block; height:100%; border-radius:99px; }

/* Vérification */
.bf-summary { border-radius:14px; padding:16px 20px; margin-bottom:16px; font-weight:600; font-size:1.02rem;
  border:1px solid; display:flex; gap:12px; align-items:center; }
.bf-claim { border:1px solid var(--bf-line); border-left:4px solid; border-radius:14px; background:var(--bf-card);
  padding:18px 22px; margin-bottom:12px; }
.bf-claim .quote { font-family:'Space Grotesk',sans-serif; font-size:1.12rem; color:var(--bf-text); margin:0 0 12px; line-height:1.35; }
.bf-tags { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px; }
.bf-tag { font-size:.78rem; font-weight:600; padding:4px 11px; border-radius:999px; }
.bf-answer { color:#D5DBE5; line-height:1.6; font-size:.96rem; }
.bf-answer b { color:var(--bf-text); }
.bf-explain { color:var(--bf-muted); font-size:.82rem; margin-top:10px; }
.bf-source { display:flex; gap:12px; align-items:flex-start; padding:10px 0; border-top:1px solid var(--bf-line); }
.bf-source:first-child { border-top:none; }
.bf-source .chip { flex:0 0 auto; font-size:.72rem; font-weight:700; padding:3px 9px; border-radius:6px; margin-top:2px; }
.bf-source a { color:var(--bf-text) !important; font-weight:600; text-decoration:none; }
.bf-source a:hover { color:var(--bf-red) !important; }
.bf-source .meta { color:var(--bf-muted); font-size:.8rem; margin-top:2px; }
.bf-source .passage { color:#B8C0CF; font-size:.85rem; margin-top:6px; font-style:italic; }
.bf-footer { text-align:center; color:var(--bf-muted); font-size:.8rem; margin:40px 0 10px; }
</style>
"""


def hero(sources_count: int) -> str:
    return f"""
<div class="bf-hero">
  <div class="bf-logo"><div class="bf-mark">◉</div><h1 class="bf-title">BLACK<span>FACE</span></h1></div>
  <p>Démasquez les contenus générés par intelligence artificielle et confrontez chaque information
  à des sources fiables : fact-checkeurs, agences de presse, médias de référence.</p>
  <div class="bf-pills">
    <span class="bf-pill">🖼️ <b>Images</b> · 🎬 <b>Vidéos</b> · 📄 <b>Textes, PDF, Word</b></span>
    <span class="bf-pill">🔎 <b>{sources_count}</b> sources fiables</span>
    <span class="bf-pill">🔒 <b>100 % local</b> et gratuit</span>
  </div>
</div>"""


def section(number: int, title: str) -> str:
    return f'<div class="bf-section"><div class="bf-num">{number}</div><h3>{escape(title)}</h3></div>'


def status_panel(engines: list[tuple[str, bool, str]]) -> str:
    rows = "".join(
        f'<div class="bf-status-row"><span><i class="bf-dot" style="background:{"#2FD38A" if ok else "#3A4356"}"></i>'
        f'{escape(name)}</span><span style="color:{"#2FD38A" if ok else "#8A93A6"};font-size:.8rem">{escape(note)}</span></div>'
        for name, ok, note in engines
    )
    return f"""
<div class="bf-card">
  <div class="bf-label">Moteurs de recherche</div>
  <div class="bf-status">{rows}</div>
</div>
<div class="bf-card">
  <div class="bf-label">Comment ça marche</div>
  <ol class="bf-steps">
    <li>Déposez un fichier ou collez un texte.</li>
    <li>Les métadonnées et un modèle spécialisé estiment s'il a été généré par IA.</li>
    <li>Les affirmations factuelles sont recherchées dans des sources fiables.</li>
    <li>Chaque source est lue : elle confirme, contredit ou ne dit rien.</li>
  </ol>
</div>"""


def _probability_color(probability: float | None) -> str:
    if probability is None:
        return "#8A93A6"
    if probability >= 0.6:
        return "#FF4D5E"
    if probability > 0.4:
        return "#F5A524"
    return "#2FD38A"


def detection_card(detection: DetectionResult) -> str:
    probability = detection.probability
    color = _probability_color(probability)
    percent = round(probability * 100) if probability is not None else 0
    value = f"{percent}%" if probability is not None else "—"
    signals = []
    for signal in detection.signals:
        if signal.probability is not None:
            sig_color = _probability_color(signal.probability)
            value_html = f'<span class="value">{signal.probability:.0%}</span>'
            bar = f'<div class="bf-bar"><i style="width:{signal.probability * 100:.0f}%;background:{sig_color}"></i></div>'
        else:
            value_html, bar = '<span class="value" style="color:#8A93A6">—</span>', ""
        detail = (
            f'<div class="detail warn">⚠ {escape(signal.error)}</div>' if signal.error
            else f'<div class="detail">{escape(signal.detail)}</div>' if signal.detail else ""
        )
        signals.append(f'<div class="bf-signal"><span class="name">{escape(signal.name)}</span>{value_html}{bar}{detail}</div>')
    return f"""
<div class="bf-card">
  <div class="bf-detect">
    <div class="bf-gauge" style="--p:{percent};--c:{color}"><div><div><strong>{value}</strong><span>Probabilité IA</span></div></div></div>
    <div class="bf-detect-body">
      <div class="bf-label">Verdict</div>
      <p class="bf-verdict" style="color:{color}">{escape(detection.verdict)}</p>
      {''.join(signals)}
    </div>
  </div>
</div>"""


def summary_banner(report: FactCheckReport) -> str:
    verdicts = [c.verdict for c in report.claims]
    key = next((v for v in ("FAUX", "TROMPEUR", "CONTESTE", "VRAI", "PLUTOT_VRAI") if v in verdicts), "NON_VERIFIABLE")
    color, background = VERDICT_STYLES[key]
    icon = {"FAUX": "✕", "TROMPEUR": "!", "CONTESTE": "≠", "VRAI": "✓", "PLUTOT_VRAI": "✓"}.get(key, "?")
    text = report.error or report.summary
    return (f'<div class="bf-summary" style="color:{color};background:{background};border-color:{color}55">'
            f'<span style="font-size:1.3rem">{icon}</span>{escape(text)}</div>')


def claim_card(claim: ClaimResult) -> str:
    color, background = VERDICT_STYLES[claim.verdict]
    # La réponse commence par le verdict (« Non, c'est faux. ») : on le met en gras.
    first, separator, rest = claim.answer.partition(". ")
    if separator:
        answer = f"<b>{escape(first)}.</b> {escape(rest)}"
    else:
        answer = f"<b>{escape(claim.answer)}</b>"
    warnings = "".join(f'<div class="bf-explain" style="color:#F5A524">⚠ {escape(w)}</div>' for w in claim.warnings)
    return f"""
<div class="bf-claim" style="border-left-color:{color}">
  <p class="quote">« {escape(claim.claim)} »</p>
  <div class="bf-tags">
    <span class="bf-tag" style="color:{color};background:{background}">{escape(claim.label)}</span>
    <span class="bf-tag" style="color:#B8C0CF;background:#1A2233">{escape(CONFIDENCE_LABELS.get(claim.confidence, claim.confidence))}</span>
  </div>
  <div class="bf-answer">{answer}</div>
  <div class="bf-explain">{escape(claim.explanation)}</div>{warnings}
</div>"""


def sources_list(claim: ClaimResult) -> str:
    rows = []
    for source in claim.sources:
        label, color = STANCE_STYLES[source.stance]
        item = source.evidence
        date = f" · {escape(item.published[:10])}" if item.published else ""
        url = item.url if item.url.startswith(("https://", "http://")) else "#"
        passage = f'<div class="passage">« {escape(source.passage)} »</div>' if source.passage else ""
        rows.append(
            f'<div class="bf-source"><span class="chip" style="color:{color};background:{color}22">{label}</span>'
            f'<div><a href="{escape(url)}" target="_blank" rel="noopener">{escape(item.title)}</a>'
            f'<div class="meta">{escape(item.domain)} · {escape(item.category)}{date}</div>{passage}</div></div>'
        )
    return "".join(rows)


FOOTER = ('<div class="bf-footer">Black Face · les résultats sont des estimations fondées sur des modèles et des sources '
          'publiques, pas une certification.</div>')
