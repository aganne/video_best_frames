"""Fixtures partagées pour les tests de video_best_frames.

Toutes les images de visages proviennent de scikit-image (astronaut),
libre de droits (CC0 / domaine public).
"""
import pytest
import cv2
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def load_image(name: str):
    """Charge une image de fixture en BGR."""
    path = os.path.join(FIXTURES_DIR, name)
    img = cv2.imread(path)
    if img is None:
        pytest.skip(f"Fixture introuvable: {path}")
    return img


@pytest.fixture
def good_sharp():
    """Frame de référence : nette, sans défaut."""
    return load_image("good_sharp.jpg")


@pytest.fixture
def bad_blurry_face():
    """Frame avec visage flou (sharpness visage ~3 < 50). Image NASA (CC0)."""
    return load_image("face_real_blurry.jpg")


@pytest.fixture
def good_face():
    """Frame avec visage net (sharpness visage ~914). Image NASA (CC0)."""
    return load_image("face_real.jpg")


@pytest.fixture
def bad_obstruction():
    """Frame avec obstruction simulée (main)."""
    return load_image("bad_obstruction.jpg")


@pytest.fixture
def bad_motion_blur():
    """Frame avec flou de mouvement directionnel."""
    return load_image("bad_motion_blur.jpg")
