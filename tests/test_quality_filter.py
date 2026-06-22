"""Tests unitaires pour les filtres du pipeline video_best_frames.

Chaque test vérifie qu'une frame spécifique est correctement
acceptée ou rejetée par les filtres du QualityFilter (Pass 3).
"""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─────────────────────────────────────────────
# Helpers — duplication de la logique du Pass 3
# ─────────────────────────────────────────────

def check_sharpness(img: np.ndarray, threshold: float = 100.0) -> bool:
    """Vérifie la netteté globale via Laplacian."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() >= threshold


def check_motion_blur(img: np.ndarray, threshold: float = 0.20) -> bool:
    """Vérifie l'absence de flou de mouvement directionnel."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag_x, mag_y = np.abs(gx).mean(), np.abs(gy).mean()
    denom = max(mag_x, mag_y)
    ratio = min(mag_x, mag_y) / denom if denom > 1e-6 else 1.0
    return ratio >= threshold


def check_face_sharpness(img: np.ndarray, threshold: float = 50.0) -> bool:
    """Vérifie qu'aucun visage détecté n'est flou."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    if faces is None or len(faces) == 0:
        return True  # Pas de visage = pas de visage flou
    for (fx, fy, fw, fh) in faces:
        face_sharp = cv2.Laplacian(gray[fy:fy+fh, fx:fx+fw], cv2.CV_64F).var()
        if face_sharp < threshold:
            return False  # Au moins un visage trop flou
    return True


def check_obstruction(img: np.ndarray, max_ratio: float = 0.25) -> bool:
    """Vérifie l'absence de grande zone uniforme (obstruction)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    rows, cols = 6, 8
    cell_h, cell_w = h // rows, w // cols
    uniform = 0
    for r in range(rows):
        for c in range(cols):
            cell = gray[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            if cell.size > 0 and cv2.Laplacian(cell, cv2.CV_64F).var() < 15:
                uniform += 1
    return uniform / (rows * cols) <= max_ratio


def passes_all_filters(img: np.ndarray) -> bool:
    """Vérifie qu'une frame passe tous les filtres du Pass 3."""
    return all([
        check_sharpness(img),
        check_motion_blur(img),
        check_face_sharpness(img),
        check_obstruction(img),
    ])


# ─────────────────────────────────────────────
# Tests — Bonnes frames (doivent passer)
# ─────────────────────────────────────────────

class TestGoodFrames:
    """Frames de référence qui doivent passer tous les filtres."""

    def test_sharpness_ok(self, good_sharp):
        assert check_sharpness(good_sharp), "Frame nette rejetée par le filtre netteté"

    def test_motion_ok(self, good_sharp):
        assert check_motion_blur(good_sharp), "Frame nette rejetée par le filtre mouvement"

    def test_face_ok(self, good_sharp):
        assert check_face_sharpness(good_sharp), "Frame nette rejetée par le filtre visage"

    def test_obstruction_ok(self, good_sharp):
        assert check_obstruction(good_sharp), "Frame nette rejetée par le filtre obstruction"

    def test_all_filters_ok(self, good_sharp):
        assert passes_all_filters(good_sharp), "Frame nette devrait passer tous les filtres"


# ─────────────────────────────────────────────
# Tests — Mauvaises frames (doivent être rejetées)
# ─────────────────────────────────────────────

class TestBadFrames:
    """Frames avec défauts connus qui doivent être filtrées."""

    def test_blurry_face_rejected(self, bad_blurry_face):
        """Frame avec visage flou (sharpness ~3) doit être rejetée."""
        assert not check_face_sharpness(bad_blurry_face, threshold=50.0), \
            "Frame avec visage flou devrait être rejetée"

    def test_good_face_accepted(self, good_face):
        """Frame avec visage net (sharpness ~914) doit passer."""
        assert check_face_sharpness(good_face, threshold=50.0), \
            "Frame avec visage net devrait passer"

    def test_obstruction_rejected(self, bad_obstruction):
        """Frame avec une grande zone uniforme (main simulée)."""
        assert check_obstruction(bad_obstruction, max_ratio=0.25) == False, \
            "Frame obstruée devrait être rejetée"

    def test_motion_blur_rejected(self, bad_motion_blur):
        """Frame avec flou de mouvement directionnel."""
        assert check_motion_blur(bad_motion_blur, threshold=0.20) == False, \
            "Frame avec flou mouvement devrait être rejetée"


# ─────────────────────────────────────────────
# Tests — Cas particuliers
# ─────────────────────────────────────────────

class TestUserRejectedFrame:
    """Frame que l'utilisateur a signalée comme non désirable."""

    def test_sharpness(self, good_face):
        """Vérifie la netteté globale — OK."""
        assert check_sharpness(good_face), "Netteté globale OK"

    def test_motion(self, good_face):
        """Vérifie l'absence de flou mouvement — OK."""
        assert check_motion_blur(good_face), "Pas de flou mouvement"

    def test_face(self, good_face):
        """Vérifie la netteté du visage — OK (sharpness=914)."""
        assert check_face_sharpness(good_face), "Visage net"

    def test_obstruction(self, good_face):
        """Vérifie l'absence d'obstruction — OK."""
        assert check_obstruction(good_face), "Pas d'obstruction"

    def test_passes_current_filters(self, good_face):
        """Cette frame passe TOUS les filtres techniques actuels.
        
        Si ce test échoue, c'est qu'un nouveau filtre a été ajouté
        qui détecte le problème visuel de cette frame.
        """
        assert passes_all_filters(good_face), (
            "Cette frame passe tous les filtres — "
            "si elle ne devrait pas, ajouter un nouveau check dans Pass 3"
        )
