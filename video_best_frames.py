"""
video_best_frames.py
--------------------
Extrait les meilleures photos d'une vidéo courte (téléphone).

Hiérarchie de détection (5 niveaux) :
  N1 - Émotions positives   (DeepFace / FER)
  N2 - Engagement humain    (MediaPipe Holistic)
  N3 - Scène significative  (CLIP via HuggingFace)
  N4 - Qualité technique    (OpenCV : netteté, exposition)
  N5 - Fallback temporel    (1 frame / intervalle + déduplication phash)

Installation des dépendances :
  pip install opencv-python deepface mediapipe transformers torch \\
              Pillow imagehash numpy tqdm fer piexif

  Optionnel (lecture EXIF vidéo enrichie) :
    - ffprobe  : inclus dans ffmpeg  →  https://ffmpeg.org/download.html
    - exiftool : outil CLI            →  https://exiftool.org

Métadonnées injéctées dans chaque JPEG :
  - Provenance vidéo  : nom du fichier source, date de création
  - Horodatage précis : date/heure de la frame dans la vidéo (DateTimeOriginal)
  - GPS               : coordonnées extraites de la vidéo si disponibles
  - Appareil          : marque, modèle, logiciel (Make, Model, Software)
  - Infos script      : scores IA, rang, niveau de détection (champ UserComment)
  - Dimensions        : résolution réelle de la frame exportée

Usage :
  python video_best_frames.py --input ma_video.mp4 --output ./photos --top 10
  python video_best_frames.py --input ma_video.mp4 --top 5 --no-clip
  python video_best_frames.py --input ma_video.mp4 --sample-rate 0.5 --min-score 0.2
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
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Chargement optionnel des librairies lourdes
# ─────────────────────────────────────────────

pass
