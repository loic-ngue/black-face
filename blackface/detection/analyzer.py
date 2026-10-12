"""Point d'entrée de la détection : choisit l'analyse selon le type de fichier."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Callable

from PIL import Image

from ..language import detect_language
from .documents import extract_document
from .hf_models import ImageDetector, TextDetector
from .metadata import scan_bytes, scan_image, scan_text
from .signals import DetectionResult, Signal

IMAGE_TYPES = {"jpg", "jpeg", "png", "webp", "bmp", "gif"}
VIDEO_TYPES = {"mp4", "mov", "avi", "mkv", "webm"}
DOCUMENT_TYPES = {"pdf", "docx", "txt"}
SUPPORTED_TYPES = sorted(IMAGE_TYPES | VIDEO_TYPES | DOCUMENT_TYPES)

VIDEO_FRAMES = 12
# En dessous, le texte est analysé mais le résultat est signalé comme peu fiable.
SHORT_TEXT_CHARS = 200


class UnsupportedFileError(ValueError):
    pass


class Detector:
    """Assemble les indices disponibles : métadonnées et modèles spécialisés."""

    def __init__(
        self,
        *,
        image_model: Callable[[], ImageDetector] | None = None,
        text_model: Callable[[], TextDetector] | None = None,
    ) -> None:
        # Les modèles sont fournis sous forme de fabriques pour n'être chargés qu'au besoin.
        self.image_model = image_model
        self.text_model = text_model

    def analyze_file(self, filename: str, data: bytes) -> DetectionResult:
        extension = Path(filename).suffix.casefold().lstrip(".")
        if extension in IMAGE_TYPES:
            return self.analyze_image(data)
        if extension in VIDEO_TYPES:
            return self.analyze_video(data, extension)
        if extension in DOCUMENT_TYPES:
            text, metadata_text = extract_document(data, extension)
            return self.analyze_text(text, media_type=extension, metadata_text=metadata_text)
        raise UnsupportedFileError(f"Format non pris en charge : .{extension or '?'}")

    # --- Images -------------------------------------------------------------------

    def analyze_image(self, data: bytes) -> DetectionResult:
        image = Image.open(io.BytesIO(data))
        image.load()
        findings = scan_image(image, data)
        result = DetectionResult("image", metadata=findings.as_dict())
        result.signals.append(findings.as_signal())
        result.signals.append(
            self._run("Modèle de détection d'images", lambda: self._image_model_signal([image]))
        )
        return result

    # --- Vidéos -------------------------------------------------------------------

    def analyze_video(self, data: bytes, extension: str) -> DetectionResult:
        # Les marqueurs de conteneur (encodeur, C2PA) se trouvent au début ou à la fin du fichier.
        findings = scan_bytes(data[: 4 * 1024 * 1024] + data[-4 * 1024 * 1024 :])
        frames, info = _sample_frames(data, extension, VIDEO_FRAMES)
        result = DetectionResult("video", metadata={**findings.as_dict(), **info})
        result.signals.append(findings.as_signal())
        if not frames:
            result.signals.append(Signal("Images de la vidéo", None, 0.0, "", error="Aucune image n'a pu être lue."))
            return result
        result.signals.append(
            self._run("Modèle de détection d'images", lambda: self._image_model_signal(frames))
        )
        return result

    # --- Textes et documents ------------------------------------------------------

    def analyze_text(self, text: str, *, media_type: str = "text", metadata_text: str = "") -> DetectionResult:
        findings = scan_text(metadata_text)
        result = DetectionResult(media_type, metadata=findings.as_dict(), extracted_text=text)
        if findings.ai_markers:
            result.signals.append(findings.as_signal())
        elif media_type in {"pdf", "docx"}:
            result.signals.append(
                Signal("Métadonnées", None, 0.0, "Aucun outil IA mentionné dans les propriétés du document.")
            )
        if not text.strip():
            result.signals.append(Signal("Analyse du texte", None, 0.0, "", error="Aucun texte à analyser."))
            return result

        model_signal = self._run("Modèle de détection de textes", lambda: self._text_model_signal(text))
        if len(text.strip()) < SHORT_TEXT_CHARS and model_signal.probability is not None:
            model_signal.detail += (
                f" — texte très court (moins de {SHORT_TEXT_CHARS} caractères) : résultat peu fiable"
            )
        language = detect_language(text)
        if language != "en" and model_signal.probability is not None:
            # Modèle entraîné principalement sur l'anglais : résultat moins fiable.
            model_signal.weight = 0.5
            model_signal.detail += f" — texte en « {language} » : modèle surtout entraîné sur l'anglais, à interpréter avec prudence"
        result.signals.append(model_signal)
        return result

    # --- Outils internes ----------------------------------------------------------

    def _image_model_signal(self, images: list[Image.Image]) -> Signal:
        if self.image_model is None:
            raise RuntimeError("Modèle image non configuré.")
        scores = self.image_model().predict(images)
        mean = sum(scores) / len(scores)
        detail = f"{ImageDetector.model_id} : {mean:.0%} de probabilité IA"
        if len(scores) > 1:
            detail += f" (moyenne sur {len(scores)} images, min {min(scores):.0%}, max {max(scores):.0%})"
        return Signal("Modèle de détection d'images", mean, 1.0, detail)

    def _text_model_signal(self, text: str) -> Signal:
        if self.text_model is None:
            raise RuntimeError("Modèle texte non configuré.")
        probability = self.text_model().predict(text)
        return Signal(
            "Modèle de détection de textes",
            probability,
            1.0,
            f"{TextDetector.model_id} : {probability:.0%} de probabilité IA",
        )

    @staticmethod
    def _run(name: str, compute: Callable[[], Signal]) -> Signal:
        """Exécute un indice; une erreur n'interrompt pas les autres analyses."""
        try:
            return compute()
        except Exception as error:  # noqa: BLE001 - l'erreur est affichée à l'utilisateur
            return Signal(name, None, 0.0, "", error=f"{type(error).__name__} : {error}")


def _sample_frames(data: bytes, extension: str, count: int) -> tuple[list[Image.Image], dict[str, object]]:
    import cv2

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"video.{extension}"
        path.write_bytes(data)
        capture = cv2.VideoCapture(str(path))
        try:
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
            info = {
                "images_total": total,
                "ips": round(fps, 2),
                "duree_s": round(total / fps, 1) if fps else None,
                "resolution": f"{int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
            }
            if total <= 0:
                return [], info
            positions = sorted({int(total * (i + 0.5) / count) for i in range(count)})
            frames: list[Image.Image] = []
            for position in positions:
                capture.set(cv2.CAP_PROP_POS_FRAMES, position)
                ok, frame = capture.read()
                if ok:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            return frames, info
        finally:
            capture.release()
