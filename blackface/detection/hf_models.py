"""Modèles Hugging Face de détection IA (chargés une fois, exécutés sur CPU ou GPU)."""

from __future__ import annotations

from typing import Sequence

from PIL import Image

from ..config import (
    IMAGE_MODEL_ID,
    IMAGE_MODEL_REVISION,
    NLI_MODEL_ID,
    NLI_MODEL_REVISION,
    TEXT_MODEL_ID,
    TEXT_MODEL_REVISION,
)

AI_LABELS = {"artificial", "ai", "fake", "generated", "synthetic", "machine"}


def _device(torch):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cache_position_projections(model, torch) -> None:
    """Accélère DeBERTa en inférence : les projections des positions relatives sont constantes.

    À chaque appel, chaque couche re-projette les mêmes 512 vecteurs de position, quel que soit
    le texte. On calcule ces projections une fois puis on les réutilise (résultats identiques).
    Si la structure interne de `transformers` change, on garde simplement le modèle d'origine.
    """
    try:
        encoder = model.deberta.encoder
        with torch.inference_mode():
            rel = encoder.get_rel_embedding().clone()
        layers = [layer.attention.self for layer in encoder.layer]
        if not all(attention.share_att_key for attention in layers):
            return
    except AttributeError:
        return

    class CachedProjection(torch.nn.Module):
        def __init__(self, linear):
            super().__init__()
            self.linear, self.cache = linear, None

        def forward(self, x):
            if x.data_ptr() == rel.data_ptr():
                if self.cache is None:
                    self.cache = self.linear(x)
                return self.cache
            return self.linear(x)

    encoder.get_rel_embedding = lambda: rel
    for attention in layers:
        attention.query_proj = CachedProjection(attention.query_proj)
        attention.key_proj = CachedProjection(attention.key_proj)


def _ai_index(id2label: dict) -> int:
    for index, label in id2label.items():
        if str(label).casefold() in AI_LABELS:
            return int(index)
    raise ValueError(f"Aucun libellé « IA » trouvé parmi {id2label}")


class ImageDetector:
    """Classifieur d'images « générée par IA » vs « réelle »."""

    model_id = IMAGE_MODEL_ID
    batch_size = 8

    def __init__(self) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self._torch = torch
        self.device = _device(torch)
        self.processor = AutoImageProcessor.from_pretrained(IMAGE_MODEL_ID, revision=IMAGE_MODEL_REVISION)
        self.model = AutoModelForImageClassification.from_pretrained(
            IMAGE_MODEL_ID, revision=IMAGE_MODEL_REVISION, use_safetensors=True
        ).to(self.device).eval()
        self.ai_index = _ai_index(self.model.config.id2label)

    def predict(self, images: Sequence[Image.Image]) -> list[float]:
        """Probabilité IA de chaque image."""
        scores: list[float] = []
        for start in range(0, len(images), self.batch_size):
            batch = [img.convert("RGB") for img in images[start : start + self.batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with self._torch.inference_mode():
                probs = self._torch.softmax(self.model(**inputs).logits, dim=-1)[:, self.ai_index]
            scores.extend(float(p) for p in probs.cpu())
        return scores


class TextDetector:
    """Classifieur de textes « écrit par IA » vs « écrit par un humain » (surtout anglais)."""

    model_id = TEXT_MODEL_ID
    max_chunks = 6

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = _device(torch)
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID, revision=TEXT_MODEL_REVISION)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            TEXT_MODEL_ID, revision=TEXT_MODEL_REVISION, use_safetensors=True
        ).to(self.device).eval()
        self.ai_index = _ai_index(self.model.config.id2label)

    def predict(self, text: str) -> float:
        """Probabilité IA moyenne sur des fenêtres de 512 tokens."""
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        window = 510
        chunks = [ids[i : i + window] for i in range(0, len(ids), window)][: self.max_chunks] or [[]]
        texts = [self.tokenizer.decode(chunk) for chunk in chunks]
        inputs = self.tokenizer(
            texts, truncation=True, max_length=512, padding=True, return_tensors="pt"
        ).to(self.device)
        with self._torch.inference_mode():
            probs = self._torch.softmax(self.model(**inputs).logits, dim=-1)[:, self.ai_index]
        return float(probs.mean())


class NLIModel:
    """Inférence textuelle multilingue : un texte confirme-t-il ou contredit-il une affirmation ?"""

    model_id = NLI_MODEL_ID
    batch_size = 8

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = _device(torch)
        self.tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_ID, revision=NLI_MODEL_REVISION)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            NLI_MODEL_ID, revision=NLI_MODEL_REVISION, use_safetensors=True
        ).to(self.device).eval()
        labels = {str(label).casefold(): int(index) for index, label in self.model.config.id2label.items()}
        self.entail_index, self.contra_index = labels["entailment"], labels["contradiction"]
        _cache_position_projections(self.model, torch)

    def predict(self, premises: Sequence[str], hypothesis: str) -> list[tuple[float, float]]:
        """Pour chaque texte source : (probabilité qu'il confirme, probabilité qu'il contredise)."""
        results: list[tuple[float, float]] = []
        for start in range(0, len(premises), self.batch_size):
            batch = list(premises[start : start + self.batch_size])
            inputs = self.tokenizer(
                batch,
                [hypothesis] * len(batch),
                truncation="only_first",
                max_length=512,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with self._torch.inference_mode():
                probs = self._torch.softmax(self.model(**inputs).logits, dim=-1).cpu()
            results.extend((float(p[self.entail_index]), float(p[self.contra_index])) for p in probs)
        return results
