"""Model tests for BLIP frame captioning."""

import pytest
from PIL import Image

pytestmark = pytest.mark.model


def test_blip_caption_real_image(sample_frame_path):
    from kairos.video.frame_captioning import blip_frame

    image = Image.open(sample_frame_path).convert("RGB")
    caption = blip_frame(image, prompt="a photo of")
    assert isinstance(caption, str)
    assert len(caption.strip()) > 0


def test_blip_caption_contains_plausible_words(sample_frame_path):
    from kairos.video.frame_captioning import blip_frame

    image = Image.open(sample_frame_path).convert("RGB")
    caption = blip_frame(image, prompt="a photo of").lower()
    plausible = ["woman", "car", "driving", "person", "vehicle", "road"]
    assert any(word in caption for word in plausible), (
        f"Caption '{caption}' lacks plausible words"
    )


def test_blip_caption_numpy_input(sample_frame_path):
    import cv2

    from kairos.video.frame_captioning import blip_frame

    image = cv2.imread(str(sample_frame_path))
    assert image is not None
    caption = blip_frame(image)
    assert isinstance(caption, str)
    assert len(caption.strip()) > 0
