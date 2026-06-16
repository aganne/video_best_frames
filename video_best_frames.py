"""
video_best_frames.py
--------------------
Pipeline en 3 passes pour extraire les meilleures frames d'une vidéo.

Pass 1 — Scene Detection : repérer les moments intéressants (audio, visage, mouvement, CLIP)
Pass 2 — Frame Extraction : extraire ~30 frames max, réparties intelligemment
Pass 3 — Quality Filter  : éliminer les floues, vides, ou redondantes

Usage batch :
  python video_best_frames.py --config config.yaml --input video.mp4
  python video_best_frames.py --config config.yaml --input-dir ./videos/ --batch
"""

import argparse
import gc
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
import yaml
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class Moment:
    """Un moment intéressant détecté dans la vidéo."""
    start_sec: float
    end_sec: float
    score: float = 0.0
    sources: List[str] = field(default_factory=list)  # "scene_change", "speech", "emotion", "clip"
    description: str = ""

@dataclass
class Frame:
    """Une frame extraite avec ses métadonnées."""
    timestamp_sec: float     # Position dans la vidéo
    path: str = ""           # Chemin du fichier JPEG (rempli à l'export)
    score: float = 0.0       # Score global
    sharpness: float = 0.0   # Netteté (Laplacian)
    moment_index: int = -1   # Index du moment source
    detection_level: int = 0 # 1-5 (N1=N5)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

DEFAULT_CONFIG = {
    "mode": "segment",
    "segment_duration": 3.0,
    "sample_rate": 0.5,
    "max_frames_per_video": 30,
    "use_clip": True,
    "clip_model": "openai/clip-vit-base-patch32",
    "whisper_model": "small",
    "whisper_language": "fr",
    "dedup_threshold": 5,
    "min_sharpness": 100.0,
    "sharpness_threshold": 100.0,   # Netteté minimale pour le Pass 3 (QualityFilter) — valeurs typiques: <30 flou, 30-80 moyen, >80 net, >200 très net
    "motion_blur_threshold": 0.20,  # Ratio de flou de mouvement (0-1). Plus bas = plus de flou directionnel. 0.20 = bon équilibre, désactiver = 0
    "min_clip_quality_score": 0.2,
    "require_face": False,
    "output_root": "./best_photos",
    "enable_transcription": True,
    "scene_weight": 0.3,
    "speech_weight": 0.3,
    "emotion_weight": 0.2,
    "clip_weight": 0.2,
}


def load_config(config_path: Optional[str] = None) -> dict:
    """Charge la config depuis un fichier YAML, fusionne avec les défauts."""
    cfg = dict(DEFAULT_CONFIG)
    if config_path:
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg.update(user_cfg)
    return cfg


# ─────────────────────────────────────────────
# Pass 1 — Scene Detection
# ─────────────────────────────────────────────

class SceneDetector:
    """Détecte les moments intéressants dans une vidéo (Pass 1)."""

    CLIP_POSITIVE_PROMPTS = [
        "people smiling, happy moment",
        "family gathering, celebration",
        "beautiful landscape, scenery",
        "action, movement, activity",
        "children playing, having fun",
        "special event, memorable",
        "group of people talking",
        "sunset, sunrise, beautiful light",
        "close-up face, expression",
        "sport, game, outdoor activity",
    ]

    CLIP_NEGATIVE_PROMPTS = [
        "blurry, out of focus",
        "dark, underexposed",
        "empty scene, nothing",
        "boring, repetitive",
        "hand covering lens",
    ]

    def __init__(self, config: dict):
        self.cfg = config
        self._clip_model = None
        self._clip_processor = None
        self._whisper_model = None

    def detect(self, video_path: str) -> List[Moment]:
        """Détecte les moments intéressants.  Retourne une liste de Moments."""
        moments: List[Moment] = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir la vidéo: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        logger.info(f"📹 Vidéo: {Path(video_path).name} — {duration:.0f}s, {fps:.1f} fps, {total_frames} frames")
        cap.release()

        # 1a. Scene changes (fast - OpenCV)
        scenes = self._detect_scene_changes(video_path)
        logger.info(f"   Changements de scène: {len(scenes)}")
        for s, e in scenes:
            moments.append(Moment(start_sec=s, end_sec=e, score=0.5, sources=["scene_change"]))

        # 1b. Speech segments (Whisper)
        if self.cfg.get("enable_transcription", True):
            speech_segments = self._detect_speech(video_path)
            logger.info(f"   Segments de parole: {len(speech_segments)}")
            for s, e, txt in speech_segments:
                moments.append(Moment(start_sec=s, end_sec=e, score=0.7, sources=["speech"], description=txt[:80]))

        # 1c. CLIP scoring (on sampled frames)
        if self.cfg.get("use_clip", True):
            clip_moments = self._score_with_clip(video_path, fps, total_frames)
            logger.info(f"   Moments CLIP: {len(clip_moments)}")
            moments.extend(clip_moments)

        # Merge overlapping moments
        moments = self._merge_moments(moments)
        moments.sort(key=lambda m: m.score, reverse=True)
        logger.info(f"   → {len(moments)} moments fusionnés")
        return moments

    def _detect_scene_changes(self, video_path: str) -> List[Tuple[float, float]]:
        """Détecte les changements de plan via histogram comparison."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        scenes = []
        prev_hist = None
        scene_start = 0.0
        min_scene_len = 1.0  # 1 seconde minimum

        sample_interval = max(1, int(fps * 0.5))  # 2 samples/sec
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

                if prev_hist is not None:
                    diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
                    if diff > 30:  # Seuil de changement
                        t = frame_idx / fps
                        if t - scene_start >= min_scene_len:
                            scenes.append((scene_start, t))
                        scene_start = t
                prev_hist = hist
            frame_idx += 1

        cap.release()
        # Dernier segment
        total = frame_idx / fps
        if total - scene_start >= min_scene_len:
            scenes.append((scene_start, total))

        return scenes

    def _detect_speech(self, video_path: str) -> List[Tuple[float, float, str]]:
        """Utilise Whisper pour détecter les segments de parole."""
        try:
            import whisper
            if self._whisper_model is None:
                model_name = self.cfg.get("whisper_model", "small")
                logger.info(f"   Chargement Whisper ({model_name})…")
                self._whisper_model = whisper.load_model(model_name)

            result = self._whisper_model.transcribe(
                video_path,
                language=self.cfg.get("whisper_language", "fr") or None,
                task="transcribe",
                verbose=False,
            )
            segments = []
            for seg in result.get("segments", []):
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                text = seg.get("text", "").strip()
                if text and end - start >= 0.5:
                    segments.append((start, end, text))
            return segments
        except Exception as e:
            logger.warning(f"Whisper speech detection failed: {e}")
            return []

    def _score_with_clip(self, video_path: str, fps: float, total_frames: int) -> List[Moment]:
        """Score les frames avec CLIP pour trouver les moments intéressants."""
        try:
            from transformers import CLIPModel, CLIPProcessor
            import torch

            if self._clip_model is None:
                model_name = self.cfg.get("clip_model", "openai/clip-vit-base-patch32")
                logger.info(f"   Chargement CLIP ({model_name})…")
                self._clip_model = CLIPModel.from_pretrained(model_name)
                self._clip_processor = CLIPProcessor.from_pretrained(model_name)

            # Échantillonnage : 1 frame / (2 * segment_duration) pour la détection
            sample_rate = max(0.5, self.cfg.get("segment_duration", 3.0) / 2)
            sample_interval = max(1, int(fps * sample_rate))
            duration = total_frames / fps

            cap = cv2.VideoCapture(video_path)
            scores = []  # [(timestamp, score)]
            frame_idx = 0

            # Préparer les textes CLIP
            texts = self.CLIP_POSITIVE_PROMPTS + [f"negative: {p}" for p in self.CLIP_NEGATIVE_PROMPTS]

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % sample_interval == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)

                    inputs = self._clip_processor(
                        text=texts,
                        images=pil_img,
                        return_tensors="pt",
                        padding=True,
                    )

                    with torch.no_grad():
                        outputs = self._clip_model(**inputs)
                        logits_per_image = outputs.logits_per_image  # image-text similarity

                    # Score = similarité aux prompts positifs - similarité aux prompts négatifs
                    n_pos = len(self.CLIP_POSITIVE_PROMPTS)
                    pos_score = logits_per_image[0, :n_pos].max().item()
                    neg_score = logits_per_image[0, n_pos:].max().item() if n_pos < len(texts) else 0
                    clip_score = (pos_score - neg_score) / 100  # Normalisation approximative

                    t = frame_idx / fps
                    scores.append((t, max(0, clip_score)))

                frame_idx += 1
                # Skip frames for speed
                frame_idx += sample_interval - 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            cap.release()

            # Grouper les scores consécutifs élevés en moments
            moments = []
            seg_duration = self.cfg.get("segment_duration", 3.0)
            threshold = self.cfg.get("min_clip_quality_score", 0.2)

            i = 0
            while i < len(scores):
                if scores[i][1] >= threshold:
                    start_t = scores[i][0]
                    # Étendre le moment tant que le score reste bon
                    j = i
                    best_score = scores[i][1]
                    while j < len(scores) - 1 and scores[j + 1][0] - start_t <= seg_duration * 2:
                        j += 1
                        best_score = max(best_score, scores[j][1])
                    end_t = scores[j][0] + sample_rate
                    moments.append(Moment(start_sec=start_t, end_sec=end_t, score=best_score, sources=["clip"]))
                    i = j + 1
                else:
                    i += 1

            return moments

        except Exception as e:
            logger.warning(f"CLIP scoring failed: {e}")
            return []

    def _merge_moments(self, moments: List[Moment]) -> List[Moment]:
        """Fusionne les moments qui se chevauchent."""
        if not moments:
            return []

        # Trier par start_sec
        moments.sort(key=lambda m: m.start_sec)

        merged = [moments[0]]
        for m in moments[1:]:
            last = merged[-1]
            # Seuil de fusion : 2 secondes d'intervalle max
            if m.start_sec <= last.end_sec + 2.0:
                last.end_sec = max(last.end_sec, m.end_sec)
                last.score = max(last.score, m.score)
                last.sources.extend(s for s in m.sources if s not in last.sources)
                if m.description and m.description not in last.description:
                    last.description = (last.description + " | " + m.description)[:200]
            else:
                merged.append(m)

        return merged


# ─────────────────────────────────────────────
# Pass 2 — Frame Extraction
# ─────────────────────────────────────────────

class FrameExtractor:
    """Extrait les meilleures frames des moments détectés (Pass 2)."""

    def __init__(self, config: dict):
        self.cfg = config

    def extract(self, video_path: str, moments: List[Moment]) -> List[Frame]:
        """Extrait jusqu'à max_frames_per_video frames, réparties entre les moments."""
        if not moments:
            logger.info("   Aucun moment détecté — fallback temporel")
            return self._fallback_extraction(video_path)

        max_frames = self.cfg.get("max_frames_per_video", 30)
        seg_duration = self.cfg.get("segment_duration", 3.0)
        sample_rate = self.cfg.get("sample_rate", 0.5)

        # Distribuer les frames entre les moments proportionnellement au score
        total_score = sum(m.score for m in moments)
        if total_score <= 0:
            per_moment = max(1, max_frames // len(moments))
            frames_per_moment = {i: per_moment for i in range(len(moments))}
        else:
            raw = [max(1, int(max_frames * m.score / total_score)) for m in moments]
            frames_per_moment = {}
            for i, n in enumerate(raw):
                frames_per_moment[i] = n

        # Ajuster pour arriver à max_frames
        total = sum(frames_per_moment.values())
        while total > max_frames:
            # Enlever 1 au moment avec le plus de frames
            idx = max(frames_per_moment, key=lambda i: frames_per_moment[i])
            if frames_per_moment[idx] > 1:
                frames_per_moment[idx] -= 1
                total -= 1
            else:
                break

        all_frames: List[Frame] = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir la vidéo: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        for i, moment in enumerate(moments):
            n_frames = frames_per_moment.get(i, 1)
            seg_len = moment.end_sec - moment.start_sec
            if seg_len <= 0:
                continue

            # Échantillonner dans le moment
            step = max(0.2, seg_len / max(n_frames * 2, 1))
            candidates: List[Frame] = []

            t = moment.start_sec
            while t <= moment.end_sec:
                frame_idx = int(t * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    t += sample_rate
                    continue

                score = self._score_frame(frame)
                candidates.append(Frame(
                    timestamp_sec=t,
                    score=score,
                    sharpness=self._sharpness(frame),
                    moment_index=i,
                    detection_level=3 if (t % 1 < 0.5) else 2,
                ))
                t += step

            # Garder les N meilleures frames candidates
            candidates.sort(key=lambda f: f.score, reverse=True)
            for f in candidates[:n_frames]:
                all_frames.append(f)

        cap.release()

        # Déduplication globale (pHash)
        all_frames = self._deduplicate(video_path, all_frames)

        # Classement final
        all_frames.sort(key=lambda f: f.score, reverse=True)
        all_frames = all_frames[:max_frames]
        logger.info(f"   → {len(all_frames)} frames extraites")
        return all_frames

    def _score_frame(self, frame: np.ndarray) -> float:
        """Score composite d'une frame : netteté + exposition + composition."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        # Netteté (Laplacian variance)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharp_score = min(1.0, sharpness / 500.0)

        # Exposition (éviter trop sombre ou trop clair)
        mean_brightness = np.mean(gray)
        exp_score = 1.0 - abs(mean_brightness - 128) / 128

        # Composition : éviter les bords trop vides
        center = frame[h//4:3*h//4, w//4:3*w//4]
        center_var = np.var(cv2.cvtColor(center, cv2.COLOR_BGR2GRAY))
        comp_score = min(1.0, center_var / 3000.0)

        return 0.5 * sharp_score + 0.3 * exp_score + 0.2 * comp_score

    def _sharpness(self, frame: np.ndarray) -> float:
        """Score de netteté via Laplacian variance."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _deduplicate(self, video_path: str, frames: List[Frame]) -> List[Frame]:
        """Élimine les frames trop similaires via pHash."""
        try:
            import imagehash
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            threshold = self.cfg.get("dedup_threshold", 5)

            kept: List[Frame] = []
            kept_hashes = []

            for f in frames:
                frame_idx = int(f.timestamp_sec * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                h = imagehash.phash(pil_img)

                if all(abs(h - other) > threshold for other in kept_hashes):
                    kept.append(f)
                    kept_hashes.append(h)

            cap.release()
            logger.info(f"   Déduplication: {len(frames)} → {len(kept)} frames")
            return kept
        except Exception as e:
            logger.warning(f"Deduplication failed: {e}")
            return frames

    def _fallback_extraction(self, video_path: str) -> List[Frame]:
        """Fallback : extraction temporelle régulière si aucun moment détecté."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        max_frames = self.cfg.get("max_frames_per_video", 30)
        interval = max(1, int(duration / max_frames))
        frames = []
        for i in range(max_frames):
            t = i * interval
            if t < duration:
                frames.append(Frame(timestamp_sec=t, score=0.5, detection_level=5))
        return frames


# ─────────────────────────────────────────────
# Pass 3 — Quality Filter
# ─────────────────────────────────────────────

class QualityFilter:
    """Filtre les frames de mauvaise qualité (Pass 3)."""

    def __init__(self, config: dict):
        self.cfg = config
        self._clip_model = None
        self._clip_processor = None

    def filter(self, video_path: str, frames: List[Frame]) -> List[Frame]:
        """Filtre les frames : flou, vide, ou sans intérêt."""
        if not frames:
            return frames

        # Utiliser sharpness_threshold si présent, sinon min_sharpness (backward compat)
        raw = self.cfg.get("sharpness_threshold")
        if not isinstance(raw, (int, float)) or raw is None:
            raw = self.cfg.get("min_sharpness", 100.0)
        min_sharpness = float(raw)
        # Seuil de flou de mouvement (0 = désactivé)
        raw_mb = self.cfg.get("motion_blur_threshold", 0.20)
        motion_blur_threshold = float(raw_mb) if isinstance(raw_mb, (int, float)) else 0.20
        require_face = self.cfg.get("require_face", False)
        min_quality = self.cfg.get("min_clip_quality_score", 0.2)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        kept: List[Frame] = []

        for f in tqdm(frames, desc="   Filtrage qualité", leave=False):
            frame_idx = int(f.timestamp_sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 1. Net ?
            sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
            if sharp < min_sharpness:
                continue

            # 1b. Flou de mouvement ? (détection directionnelle via Sobel)
            if motion_blur_threshold > 0:
                gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                mag_x = np.abs(gx).mean()
                mag_y = np.abs(gy).mean()
                # Ratio petit/grand gradient directionnel
                motion_ratio = min(mag_x, mag_y) / max(mag_x, mag_y) if max(mag_x, mag_y) > 1e-6 else 1.0
                if motion_ratio < motion_blur_threshold:
                    continue

            # 2. Pas trop sombre ?
            mean = np.mean(gray)
            if mean < 15 or mean > 240:
                continue

            # 3. Assez de variance (pas vide)
            var = np.var(gray)
            if var < 100:
                continue

            # 4. Visage présent ? (optionnel)
            if require_face:
                try:
                    import mediapipe as mp
                    mp_face = mp.solutions.face_detection
                    with mp_face.FaceDetection(min_detection_confidence=0.5) as fd:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = fd.process(rgb)
                        if not results.detections:
                            continue
                except Exception:
                    pass  # Skip face check on error

            kept.append(f)

        cap.release()

        # Vérification CLIP (si disponible)
        if self.cfg.get("use_clip", True) and len(kept) > 5:
            kept = self._clip_quality_check(video_path, kept, min_quality)

        n_removed = len(frames) - len(kept)
        if n_removed > 0:
            logger.info(f"   Filtrage qualité: -{n_removed} frames éliminées ({len(kept)} restantes)")
        return kept

    def _clip_quality_check(self, video_path: str, frames: List[Frame], min_score: float) -> List[Frame]:
        """Vérifie la qualité via CLIP avec des prompts négatifs."""
        try:
            from transformers import CLIPModel, CLIPProcessor
            import torch

            if self._clip_model is None:
                model_name = self.cfg.get("clip_model", "openai/clip-vit-base-patch32")
                self._clip_model = CLIPModel.from_pretrained(model_name)
                self._clip_processor = CLIPProcessor.from_pretrained(model_name)

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            kept = []

            negative_prompts = [
                "blurry photo", "dark image", "empty scene",
                "nothing interesting", "boring", "out of focus"
            ]

            for f in frames:
                frame_idx = int(f.timestamp_sec * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                inputs = self._clip_processor(
                    text=negative_prompts,
                    images=pil_img,
                    return_tensors="pt",
                    padding=True,
                )
                with torch.no_grad():
                    outputs = self._clip_model(**inputs)

                neg_scores = outputs.logits_per_image[0] / 100
                max_neg = neg_scores.max().item()

                if max_neg < (1.0 - min_score):
                    kept.append(f)

            cap.release()
            return kept
        except Exception as e:
            logger.warning(f"CLIP quality check failed: {e}")
            return frames


# ─────────────────────────────────────────────
# Transcription Engine
# ─────────────────────────────────────────────

class TranscriptionEngine:
    """Transcription audio Whisper."""

    def __init__(self, model_name: str = "small", language: Optional[str] = "fr"):
        self.model_name = model_name
        self.language = language
        self._model = None

    def _load(self):
        if self._model is None:
            import whisper
            logger.info(f"Chargement Whisper ({self.model_name})…")
            self._model = whisper.load_model(self.model_name)

    def transcribe(self, video_path: str, output_dir: str) -> Path:
        """Transcrit la vidéo et sauvegarde le texte."""
        self._load()
        out_path = Path(output_dir) / f"{Path(video_path).stem}_transcript.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        result = self._model.transcribe(
            video_path,
            language=self.language or None,
            task="transcribe",
            verbose=False,
        )

        # Sauvegarder le texte complet
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result.get("text", "").strip())

        # Sauvegarder les segments horodatés
        seg_path = out_path.with_suffix(".segments.json")
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": seg.get("text", "").strip(),
            })
        with open(seg_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)

        logger.info(f"   Transcription: {len(segments)} segments → {out_path.name}")
        return out_path


# ─────────────────────────────────────────────
# VideoBestFrames — Pipeline complet
# ─────────────────────────────────────────────

class VideoBestFrames:
    """Pipeline complet d'extraction des meilleures frames."""

    def __init__(self, config: Optional[dict] = None):
        self.cfg = config or dict(DEFAULT_CONFIG)
        self._scene_detector: Optional[SceneDetector] = None
        self._frame_extractor: Optional[FrameExtractor] = None
        self._quality_filter: Optional[QualityFilter] = None

    def configure(self, **kwargs):
        """Met à jour la configuration."""
        self.cfg.update(kwargs)

    def process(self, video_path: str, output_dir: Optional[str] = None) -> dict:
        """Exécute le pipeline complet sur une vidéo."""
        video_name = Path(video_path).stem
        out_dir = output_dir or str(Path(self.cfg["output_root"]) / video_name)
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'='*50}")
        logger.info(f"🎬 Traitement: {Path(video_path).name}")
        logger.info(f"{'='*50}")

        # Pass 1 — Scene Detection
        logger.info("🔍 Pass 1 — Détection des moments…")
        self._scene_detector = self._scene_detector or SceneDetector(self.cfg)
        moments = self._scene_detector.detect(video_path)
        if not moments:
            logger.info("   ⚠️ Aucun moment détecté, utilisation du fallback temporel")

        # Pass 2 — Frame Extraction
        logger.info("📸 Pass 2 — Extraction des frames…")
        self._frame_extractor = self._frame_extractor or FrameExtractor(self.cfg)
        frames = self._frame_extractor.extract(video_path, moments)

        # Pass 3 — Quality Filter
        logger.info("🔍 Pass 3 — Filtrage qualité…")
        self._quality_filter = self._quality_filter or QualityFilter(self.cfg)
        frames = self._quality_filter.filter(video_path, frames)

        # Export JPEG
        logger.info(f"💾 Export de {len(frames)} frames…")
        export_results = self._export_frames(video_path, frames, out_dir)

        # Transcription
        transcript_path = None
        segments_path = None
        if self.cfg.get("enable_transcription", True):
            try:
                transcriber = TranscriptionEngine(
                    model_name=self.cfg.get("whisper_model", "small"),
                    language=self.cfg.get("whisper_language", "fr"),
                )
                transcript_path = transcriber.transcribe(video_path, out_dir)
                segments_path = transcript_path.with_suffix(".segments.json")
            except Exception as e:
                logger.warning(f"Transcription failed: {e}")

        # Résumé
        summary = {
            "video": Path(video_path).name,
            "output_dir": out_dir,
            "moments_detected": len(moments),
            "frames_total": len(frames),
            "frames_exported": export_results["exported"],
            "transcription": transcript_path.name if transcript_path else None,
        }
        logger.info(f"✅ Terminé: {summary['frames_exported']} frames exportées")

        # Sauvegarder le résumé JSON
        with open(Path(out_dir) / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return summary

    def _export_frames(self, video_path: str, frames: List[Frame], out_dir: str) -> dict:
        """Exporte les frames en JPEG avec métadonnées EXIF."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        # Lire les métadonnées de la vidéo source (date, appareil, GPS…)
        meta = self._get_video_metadata(video_path)
        creation_date = meta.get("creation_date")
        video_name = Path(video_path).stem

        exported = 0
        for i, f in enumerate(frames):
            frame_idx = int(f.timestamp_sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Nommer le fichier
            ts_str = f"{int(f.timestamp_sec // 60):02d}m{int(f.timestamp_sec % 60):02d}s"
            fname = f"{video_name}_{ts_str}_{i+1:02d}.jpg"
            fpath = Path(out_dir) / fname

            # Sauvegarder
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.save(fpath, "JPEG", quality=95)

            # Injecter EXIF (date, appareil, GPS)
            self._inject_exif(str(fpath), creation_date, f.timestamp_sec, video_name, meta)

            f.path = str(fpath)
            exported += 1

        cap.release()
        return {"exported": exported}

    def _get_video_metadata(self, video_path: str) -> dict:
        """Extrait les métadonnées de la vidéo source via ffprobe.

        Retourne un dict avec:
          - creation_date: str (format EXIF \"YYYY:MM:DD HH:MM:SS\") ou None
          - make: str (ex: \"Apple\") ou None
          - model: str (ex: \"iPhone 14\") ou None
          - software: str (ex: \"16.6\") ou None
          - gps_lat: float ou None
          - gps_lon: float ou None
        """
        meta: dict = {}

        # Lire les tags du format via ffprobe
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_entries", "format_tags", video_path],
                capture_output=True, text=True, timeout=10,
            )
            data = json.loads(result.stdout) if result.stdout.strip() else {}
            tags = data.get("format", {}).get("tags", {})

            # Date de création — plusieurs sources possibles
            date_str = (
                tags.get("com.apple.quicktime.creationdate")
                or tags.get("creation_time")
            )
            if date_str:
                try:
                    # Formater en EXIF : "2025:01:15 14:30:00"
                    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    meta["creation_date"] = dt.strftime("%Y:%m:%d %H:%M:%S")
                except Exception:
                    pass

            # Appareil
            make = tags.get("com.apple.quicktime.make")
            if make:
                meta["make"] = make
            model = tags.get("com.apple.quicktime.model")
            if model:
                meta["model"] = model
            software = tags.get("com.apple.quicktime.software")
            if software:
                meta["software"] = software

            # GPS : chercher dans les tags
            gps_lat = tags.get("com.apple.quicktime.location.ISO6709")
            if gps_lat:
                # Format possible: "+48.8566+002.3522/" ou "+48.8566-002.3522/"
                m = re.match(r"([+-]\d+\.?\d*)([+-]\d+\.?\d*)/?", gps_lat)
                if m:
                    meta["gps_lat"] = float(m.group(1))
                    meta["gps_lon"] = float(m.group(2))

        except Exception:
            pass

        # Fallback date : date du fichier
        if not meta.get("creation_date"):
            try:
                mtime = os.path.getmtime(video_path)
                dt = datetime.fromtimestamp(mtime)
                meta["creation_date"] = dt.strftime("%Y:%m:%d %H:%M:%S")
            except Exception:
                pass

        return meta

    def _inject_exif(self, jpeg_path: str, creation_date: str, timestamp_sec: float, video_name: str, video_meta: dict = None):
        """Injecte les métadonnées EXIF dans le JPEG (date, appareil, GPS)."""
        try:
            import piexif
            from PIL import Image as PILImage

            img = PILImage.open(jpeg_path)

            # Ajuster la date en fonction du timestamp dans la vidéo
            if creation_date:
                try:
                    base_dt = datetime.strptime(creation_date, "%Y:%m:%d %H:%M:%S")
                    frame_dt = base_dt + timedelta(seconds=timestamp_sec)
                    exif_date = frame_dt.strftime("%Y:%m:%d %H:%M:%S")
                except Exception:
                    exif_date = creation_date
            else:
                exif_date = None

            # Valeurs par défaut
            make = "Apple"
            model = "iPhone"
            software = "video_best_frames"
            if video_meta:
                make = video_meta.get("make", make)
                model = video_meta.get("model", model)
                software = video_meta.get("software", software) or software

            exif_dict = {
                "Exif": {
                    piexif.ExifIFD.UserComment: f"Video: {video_name} @ {timestamp_sec:.1f}s".encode("utf-8"),
                },
                "0th": {
                    piexif.ImageIFD.Make: make,
                    piexif.ImageIFD.Model: model,
                    piexif.ImageIFD.Software: software,
                },
            }

            # Date
            if exif_date:
                exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = exif_date
                exif_dict["0th"][piexif.ImageIFD.DateTime] = exif_date

            # GPS
            if video_meta and video_meta.get("gps_lat") is not None and video_meta.get("gps_lon") is not None:
                lat = video_meta["gps_lat"]
                lon = video_meta["gps_lon"]

                def _to_dms(dec: float) -> tuple:
                    """Convertit degrés décimaux → degrés, minutes, secondes pour EXIF."""
                    dec = abs(dec)
                    d = int(dec)
                    m = int((dec - d) * 60)
                    s = (dec - d - m / 60) * 3600
                    return (d, 1), (m, 1), (int(s * 1000), 1000)

                lat_ref = "N" if lat >= 0 else "S"
                lon_ref = "E" if lon >= 0 else "W"
                exif_dict["GPS"] = {
                    piexif.GPSIFD.GPSLatitudeRef: lat_ref,
                    piexif.GPSIFD.GPSLatitude: _to_dms(lat),
                    piexif.GPSIFD.GPSLongitudeRef: lon_ref,
                    piexif.GPSIFD.GPSLongitude: _to_dms(lon),
                }

            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, jpeg_path)
        except Exception as e:
            logger.warning(f"EXIF injection failed for {Path(jpeg_path).name}: {e}")


# ─────────────────────────────────────────────
# Batch processing
# ─────────────────────────────────────────────

def batch_process(config: dict, input_dir: str, output_dir: Optional[str] = None):
    """Traite toutes les vidéos d'un dossier."""
    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mts", ".ts", ".3gp"}
    videos = sorted(
        p for p in Path(input_dir).iterdir()
        if p.is_file() and p.suffix.lower() in video_extensions
    )

    if not videos:
        logger.warning(f"Aucune vidéo trouvée dans {input_dir}")
        return

    logger.info(f"🎬 Traitement batch: {len(videos)} vidéos dans {input_dir}")
    pipeline = VideoBestFrames(config)
    results = []

    for v in videos:
        out = output_dir or config.get("output_root", "./best_photos")
        try:
            summary = pipeline.process(str(v), str(Path(out) / v.stem))
            results.append(summary)
        except Exception as e:
            logger.error(f"❌ Erreur sur {v.name}: {e}")
            results.append({"video": v.name, "error": str(e)})

    # Rapport global
    report_path = Path(output_dir or config["output_root"]) / "batch_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"total": len(videos), "results": results}, f, indent=2, ensure_ascii=False)

    success = sum(1 for r in results if "error" not in r)
    logger.info(f"\n{'='*50}")
    logger.info(f"✅ Batch terminé: {success}/{len(videos)} réussites")
    logger.info(f"📄 Rapport: {report_path}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main_cli():
    parser = argparse.ArgumentParser(description="Video Best Frames — Extraction intelligente")
    parser.add_argument("--config", "-c", help="Fichier de configuration YAML")
    parser.add_argument("--input", "-i", help="Fichier vidéo unique")
    parser.add_argument("--input-dir", "-d", help="Dossier contenant les vidéos")
    parser.add_argument("--batch", "-b", action="store_true", help="Mode batch (traite tout le dossier)")
    parser.add_argument("--output", "-o", help="Dossier de sortie")
    parser.add_argument("--max-frames", type=int, help="Nombre max de frames par vidéo")
    parser.add_argument("--verbose", "-v", action="store_true", help="Logs détaillés")

    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    # Config
    config = load_config(args.config)
    if args.max_frames:
        config["max_frames_per_video"] = args.max_frames
    if args.output:
        config["output_root"] = args.output

    if args.input:
        pipeline = VideoBestFrames(config)
        pipeline.process(args.input)
    elif args.input_dir and args.batch:
        batch_process(config, args.input_dir, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main_cli()
