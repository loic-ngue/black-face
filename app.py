"""Interface Streamlit de Black Face."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from blackface import report as report_export
from blackface import ui
from blackface.config import get_settings
from blackface.detection.analyzer import (
    DOCUMENT_TYPES,
    IMAGE_TYPES,
    SUPPORTED_TYPES,
    VIDEO_TYPES,
    Detector,
)
from blackface.detection.documents import extract_document
from blackface.detection.hf_models import ImageDetector, NLIModel, TextDetector
from blackface.detection.signals import DetectionResult
from blackface.verification.factcheck import FactChecker, FactCheckReport
from blackface.verification.search import MultiSearch
from blackface.verification.sources import TRUSTED_SOURCES

st.set_page_config(page_title="Black Face", page_icon="◉", layout="wide", initial_sidebar_state="collapsed")

settings = get_settings()


@st.cache_resource(show_spinner="Chargement du modèle de détection d'images…")
def image_model() -> ImageDetector:
    return ImageDetector()


@st.cache_resource(show_spinner="Chargement du modèle de détection de textes…")
def text_model() -> TextDetector:
    return TextDetector()


@st.cache_resource(show_spinner="Chargement du modèle de comparaison des sources…")
def nli_model() -> NLIModel:
    return NLIModel()


@st.cache_resource(show_spinner=False)
def fact_checker() -> FactChecker:
    search = MultiSearch.from_keys(settings.tavily_api_key, settings.google_factcheck_api_key)
    return FactChecker(search, nli_model)


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


html(ui.CSS)
html(ui.hero(sum(len(domains) for domains in TRUSTED_SOURCES.values())))

# --- Saisie -----------------------------------------------------------------------

left, right = st.columns([1.75, 1], gap="large")

with left:
    with st.container(border=True):
        mode = st.segmented_control(
            "Source", ["Fichier", "Texte"], default="Fichier", label_visibility="collapsed", key="mode",
        )
        filename, data, text_input = "", b"", ""
        if mode == "Texte":
            text_input = st.text_area(
                "Texte à analyser", height=200, max_chars=50_000,
                placeholder="Collez un article, une publication, un message WhatsApp…",
            )
            if text_input.strip():
                filename, data = "texte-saisi.txt", text_input.encode("utf-8")
        else:
            upload = st.file_uploader(
                "Image, vidéo ou document",
                type=SUPPORTED_TYPES,
                help="Images : JPG, PNG, WebP… · Vidéos : MP4, MOV, AVI, MKV, WebM · Documents : PDF, DOCX, TXT",
            )
            if upload is not None:
                filename, data = Path(upload.name).name, upload.getvalue()
                extension = Path(filename).suffix.casefold().lstrip(".")
                if extension in IMAGE_TYPES:
                    st.image(data, width=320)
                elif extension in VIDEO_TYPES:
                    st.video(data)

        extension = Path(filename).suffix.casefold().lstrip(".")
        context = st.text_input(
            "Affirmation à vérifier · facultatif",
            max_chars=500,
            placeholder="Ex. : Rowan Atkinson est mort.",
            help="Une phrase courte et précise donne les meilleurs résultats. Indispensable pour une image ou une vidéo.",
        )
        options = st.columns(2)
        do_detect = options[0].toggle("Détecter l'IA", value=True)
        do_verify = options[1].toggle("Vérifier les informations", value=True)

        ready = bool(data) or (do_verify and bool(context.strip()))
        analyze = st.button(
            "Analyser", type="primary", icon=":material/radar:", width="stretch",
            disabled=not ready or not (do_detect or do_verify),
        )

with right:
    html(ui.status_panel([
        ("Wikipédia", True, "actif"),
        ("Tavily · web fiable", bool(settings.tavily_api_key), "actif" if settings.tavily_api_key else "clé absente"),
        ("Google Fact Check", bool(settings.google_factcheck_api_key),
         "actif" if settings.google_factcheck_api_key else "clé absente"),
    ]))
    with st.expander(f"Les {sum(len(d) for d in TRUSTED_SOURCES.values())} sources fiables", icon=":material/verified:"):
        for category, domains in TRUSTED_SOURCES.items():
            st.markdown(f"**{category}**")
            st.caption(", ".join(domains))

# --- Analyse ----------------------------------------------------------------------

if analyze:
    detection: DetectionResult | None = None
    factcheck: FactCheckReport | None = None
    with st.status("Analyse en cours…", expanded=True) as status:
        text_to_check = text_input
        if do_detect and data:
            status.write("Recherche de traces d'intelligence artificielle…")
            try:
                detection = Detector(image_model=image_model, text_model=text_model).analyze_file(filename, data)
                text_to_check = detection.extracted_text or text_input
            except Exception as error:  # noqa: BLE001 - affiché à l'utilisateur
                st.error(f"Détection impossible : {error}")
        elif extension in DOCUMENT_TYPES:
            text_to_check = extract_document(data, extension)[0]

        if do_verify:
            status.write("Recherche dans les sources fiables et comparaison…")
            factcheck = fact_checker().check(text_to_check or "", context.strip() or None)
        status.update(label="Analyse terminée", state="complete", expanded=False)
    st.session_state["result"] = {
        "detection": detection,
        "factcheck": factcheck,
        "report": report_export.build_report(
            filename=filename or "affirmation", data=data or context.encode("utf-8"),
            detection=detection, factcheck=factcheck,
        ),
    }

# --- Résultats --------------------------------------------------------------------

result = st.session_state.get("result")
if result:
    detection, factcheck, report = result["detection"], result["factcheck"], result["report"]
    step = 1

    if detection is not None:
        html(ui.section(step, "Contenu généré par IA ?"))
        step += 1
        html(ui.detection_card(detection))
        with st.expander("Métadonnées du fichier", icon=":material/data_object:"):
            st.json(detection.metadata, expanded=False)

    if factcheck is not None:
        html(ui.section(step, "Vérification des faits"))
        step += 1
        html(ui.summary_banner(factcheck))
        for claim in factcheck.claims:
            html(ui.claim_card(claim))
            if claim.sources:
                with st.expander(f"Voir les {len(claim.sources)} sources examinées", icon=":material/library_books:"):
                    html(ui.sources_list(claim))

    html(ui.section(step, "Exporter le rapport"))
    with st.container(horizontal=True):
        st.download_button("PDF", report_export.to_pdf(report), "rapport-black-face.pdf", "application/pdf",
                           icon=":material/picture_as_pdf:")
        st.download_button(
            "Word", report_export.to_docx(report), "rapport-black-face.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document", icon=":material/description:",
        )
        st.download_button("JSON", report_export.to_json(report), "rapport-black-face.json", "application/json",
                           icon=":material/data_object:")

html(ui.FOOTER)
