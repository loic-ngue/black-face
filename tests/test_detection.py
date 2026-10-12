import io
import os

import cv2
import numpy as np
from PIL import Image, PngImagePlugin

from blackface.detection.analyzer import Detector
from blackface.detection.metadata import scan_bytes, scan_image
from blackface.detection.signals import Signal, combine, verdict_label


def png_bytes(text_chunks: dict[str, str] | None = None) -> bytes:
    info = PngImagePlugin.PngInfo()
    for key, value in (text_chunks or {}).items():
        info.add_text(key, value)
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 30, 30)).save(buffer, format="PNG", pnginfo=info)
    return buffer.getvalue()


class FakeImageModel:
    def __init__(self, score: float):
        self.score = score
        self.seen = 0

    def predict(self, images):
        self.seen += len(images)
        return [self.score] * len(images)


class FakeTextModel:
    def predict(self, text):
        return 0.9


# --- Métadonnées --------------------------------------------------------------------

def test_png_stable_diffusion_parameters_are_detected():
    data = png_bytes({"parameters": "a cat, Steps: 20, Sampler: Euler a, CFG scale: 7"})
    findings = scan_image(Image.open(io.BytesIO(data)), data)
    assert "paramètres de génération Stable Diffusion" in findings.ai_markers
    assert findings.as_signal().probability == 0.98


def test_iptc_ai_source_type_is_detected():
    findings = scan_bytes(b"...<Iptc4xmpExt:DigitalSourceType>http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia</...")
    assert findings.ai_markers == ["IPTC trainedAlgorithmicMedia"]


def test_short_tool_names_in_raw_bytes_are_not_false_positives():
    noise = os.urandom(200_000) + b"xxVeoxx Sora pika "
    assert scan_bytes(noise).ai_markers == []


def test_camera_exif_is_reported_without_weight():
    image = Image.new("RGB", (32, 32))
    exif = image.getexif()
    exif[0x010F], exif[0x0110] = "Canon", "EOS R6"
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    data = buffer.getvalue()
    findings = scan_image(Image.open(io.BytesIO(data)), data)
    signal = findings.as_signal()
    assert findings.camera == "Canon EOS R6"
    assert signal.probability is None and signal.weight == 0


# --- Combinaison des indices --------------------------------------------------------

def test_combine_weighted_mean_and_declared_ai_override():
    assert combine([Signal("A", 0.2, 1, ""), Signal("B", 0.8, 1, "")]) == 0.5
    assert combine([Signal("Métadonnées", 0.98, 1, ""), Signal("B", 0.1, 1, "")]) == 0.98
    assert combine([Signal("A", None, 0, "", error="x")]) is None
    assert verdict_label(0.9) == "Très probablement généré par IA"
    assert verdict_label(0.1) == "Très probablement authentique"


# --- Détecteur ----------------------------------------------------------------------

def test_image_analysis_uses_metadata_and_model():
    detector = Detector(image_model=lambda: FakeImageModel(0.25))
    result = detector.analyze_file("photo.png", png_bytes())
    assert [s.name for s in result.signals] == ["Métadonnées", "Modèle de détection d'images"]
    assert result.probability == 0.25


def test_declared_ai_metadata_overrides_model():
    data = png_bytes({"parameters": "Steps: 30, Sampler: DPM++"})
    result = Detector(image_model=lambda: FakeImageModel(0.1)).analyze_file("gen.png", data)
    assert result.probability == 0.98


def test_errors_in_one_signal_do_not_stop_the_analysis():
    class Broken:
        def predict(self, images):
            raise RuntimeError("panne")

    result = Detector(image_model=lambda: Broken()).analyze_file("a.png", png_bytes())
    model_signal = result.signals[1]
    assert model_signal.error == "RuntimeError : panne"
    assert result.probability is None


def test_non_english_text_is_flagged_as_less_reliable():
    result = Detector(text_model=lambda: FakeTextModel()).analyze_file("article.txt", ("Bonjour à tous, voici les nouvelles du jour. " * 10).encode())
    signal = next(s for s in result.signals if s.name == "Modèle de détection de textes")
    assert signal.weight == 0.5 and "« fr »" in signal.detail
    assert result.probability == 0.9


def test_short_text_is_scored_with_a_warning():
    result = Detector(text_model=lambda: FakeTextModel()).analyze_file("a.txt", b"A")
    assert result.probability == 0.9
    assert "peu fiable" in result.signals[-1].detail


def test_empty_text_is_not_scored():
    result = Detector(text_model=lambda: FakeTextModel()).analyze_file("a.txt", b"   ")
    assert result.probability is None


def test_video_frames_are_sampled(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for i in range(30):
        writer.write(np.full((48, 64, 3), i * 8, dtype=np.uint8))
    writer.release()
    model = FakeImageModel(0.7)
    result = Detector(image_model=lambda: model).analyze_file("clip.avi", path.read_bytes())
    assert model.seen == 12
    assert result.metadata["images_total"] == 30
    assert abs(result.probability - 0.7) < 1e-9
