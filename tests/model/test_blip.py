"""Model tests for BLIP frame captioning.

Exercises the BLIP image-captioning model on a real sample frame to verify
that it produces non-empty, plausible captions from both PIL and NumPy inputs.
"""

from pathlib import Path

import pytest
from PIL import Image

pytestmark = pytest.mark.model


def test_blip_caption_real_image(sample_frame_path: Path) -> None:
    """Verify BLIP produces a non-empty caption for a PIL image."""
    from kairos.video.frame_captioning import blip_frame

    image = Image.open(sample_frame_path).convert("RGB")
    caption = blip_frame(image, prompt="a photo of")
    assert isinstance(caption, str)
    assert len(caption.strip()) > 0


def test_blip_caption_contains_plausible_words(sample_frame_path: Path) -> None:
    """Verify the BLIP caption contains at least one semantically plausible word."""
    from kairos.video.frame_captioning import blip_frame

    image = Image.open(sample_frame_path).convert("RGB")
    caption = blip_frame(image, prompt="a photo of").lower()
    plausible = ["woman", "car", "driving", "person", "vehicle", "road"]
    assert any(word in caption for word in plausible), (
        f"Caption '{caption}' lacks plausible words"
    )


def test_blip_caption_numpy_input(sample_frame_path: Path) -> None:
    """Verify BLIP handles a NumPy (cv2) array input and returns a valid caption."""
    import cv2

    from kairos.video.frame_captioning import blip_frame

    image = cv2.imread(str(sample_frame_path))
    assert image is not None
    caption = blip_frame(image)
    assert isinstance(caption, str)
    assert len(caption.strip()) > 0
