"""Recherche de traces de génération IA dans les métadonnées d'un fichier."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from PIL import ExifTags, Image

from .signals import DECLARED_AI_PROBABILITY, Signal

# Valeurs IPTC « Digital Source Type » posées par les générateurs (Adobe, OpenAI, Google...).
IPTC_AI_MARKERS = ("trainedAlgorithmicMedia", "compositeWithTrainedAlgorithmicMedia")

# Noms longs et distinctifs : assez rares pour être cherchés dans les octets bruts.
RAW_TOOL_PATTERN = re.compile(
    rb"\b(Midjourney|Stable Diffusion|Adobe Firefly|DALL-E|DALL\xc2\xb7E|ComfyUI|"
    rb"Automatic1111|InvokeAI|Ideogram|NovelAI|Leonardo\.Ai)\b"
)
# Noms courts : cherchés uniquement dans les champs texte des métadonnées.
TEXT_TOOL_PATTERN = re.compile(
    r"\b(DALL[·\-\s]?E|Midjourney|Stable[\s_-]?Diffusion|Firefly|Imagen|Gemini|ChatGPT|"
    r"OpenAI|Flux|Ideogram|Leonardo|NovelAI|ComfyUI|Automatic1111|InvokeAI|Sora|"
    r"Runway|Kling|Pika|Veo|Grok|Aurora)\b",
    re.IGNORECASE,
)
# Paramètres de génération écrits par Automatic1111 / ComfyUI dans les PNG.
SD_PARAMETERS_PATTERN = re.compile(rb"Steps: \d+, Sampler: ")

MAX_SCAN_BYTES = 64 * 1024 * 1024


@dataclass
class MetadataFindings:
    ai_markers: list[str] = field(default_factory=list)
    has_c2pa: bool = False
    camera: str | None = None
    software: str | None = None
    exif: dict[str, str] = field(default_factory=dict)

    def add(self, marker: str) -> None:
        if marker not in self.ai_markers:
            self.ai_markers.append(marker)

    def as_signal(self) -> Signal:
        if self.ai_markers:
            return Signal(
                "Métadonnées",
                DECLARED_AI_PROBABILITY,
                weight=1.0,
                detail="Traces de génération IA : " + ", ".join(self.ai_markers),
            )
        details = []
        if self.has_c2pa:
            details.append("manifeste de provenance C2PA présent")
        if self.camera:
            details.append(f"appareil photo déclaré : {self.camera}")
        if not details:
            details.append("aucune trace IA ni information d'appareil (souvent effacées par les réseaux sociaux)")
        # L'absence de trace n'est pas une preuve : indice informatif, non pondéré.
        text = "; ".join(details)
        return Signal("Métadonnées", None, weight=0.0, detail=text[0].upper() + text[1:])

    def as_dict(self) -> dict[str, object]:
        return {
            "traces_ia": self.ai_markers,
            "c2pa": self.has_c2pa,
            "appareil": self.camera,
            "logiciel": self.software,
            "exif": self.exif,
        }


def scan_bytes(data: bytes, findings: MetadataFindings | None = None) -> MetadataFindings:
    """Recherche les marqueurs longs dans le contenu brut (XMP, PNG, conteneur vidéo...)."""
    findings = findings or MetadataFindings()
    data = data[:MAX_SCAN_BYTES]
    for marker in IPTC_AI_MARKERS:
        if marker.encode() in data:
            findings.add(f"IPTC {marker}")
    for match in RAW_TOOL_PATTERN.finditer(data):
        findings.add(f"outil « {match.group(0).decode('utf-8', 'replace')} »")
    if SD_PARAMETERS_PATTERN.search(data):
        findings.add("paramètres de génération Stable Diffusion")
    findings.has_c2pa = findings.has_c2pa or (b"c2pa" in data and b"jumb" in data)
    return findings


def scan_text(text: str, findings: MetadataFindings | None = None) -> MetadataFindings:
    """Recherche les noms d'outils IA dans des champs de métadonnées lisibles."""
    findings = findings or MetadataFindings()
    for match in TEXT_TOOL_PATTERN.finditer(text):
        findings.add(f"outil « {match.group(0)} »")
    return findings


def scan_image(image: Image.Image, data: bytes) -> MetadataFindings:
    findings = scan_bytes(data)
    exif = image.getexif()
    readable = {
        ExifTags.TAGS.get(tag, str(tag)): str(value)[:200]
        for tag, value in exif.items()
        if isinstance(value, (str, int, float))
    }
    findings.exif = readable
    make, model = readable.get("Make", "").strip(), readable.get("Model", "").strip()
    findings.camera = " ".join(part for part in (make, model) if part) or None
    findings.software = readable.get("Software") or None

    # Champs texte : EXIF utiles, chunks PNG (parameters, prompt...), XMP, commentaires.
    fields = [readable.get(key, "") for key in ("Software", "Artist", "ImageDescription", "Make", "Model")]
    for value in image.info.values():
        if isinstance(value, bytes):
            value = value.decode("utf-8", "ignore")
        if isinstance(value, str):
            fields.append(value[:20000])
    return scan_text("\n".join(fields), findings)
