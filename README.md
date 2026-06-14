# Video Best Frames 🎬

Interface Streamlit pour extraire automatiquement les meilleures photos d'une vidéo courte (vidéo téléphone).

## Fonctionnalités

- **Intelligence IA multi-niveaux** : Détection d'émotions, engagement humain, scènes significatives
- **Extraction intelligente** : CLIP (vision-langage) + qualité technique (netteté, exposition)
- **Mode représentatif** : Découper la vidéo en segments et garder la meilleure frame par segment
- **Transcription audio** : Intégration Whisper pour transcrire automatiquement les vidéos
- **Métadonnées enrichies** : Injection EXIF (GPS, date, appareil, scores IA)
- **Traitement par lot** : Traiter plusieurs vidéos en une seule opération
- **Interface web intuitive** : Application Streamlit avec paramètres ajustables

## Installation

### Prérequis
- Python 3.9+
- FFmpeg (pour l'extraction audio)

### Dépendances Python

```bash
pip install -r requirements.txt
```

## Utilisation

### Lancer l'application web

```bash
streamlit run app_streamlit.py
```

L'application sera accessible sur `http://localhost:8501`

### Utilisation en ligne de commande

```bash
python video_best_frames.py --input ma_video.mp4 --output ./photos --top 10
python video_best_frames.py --input ma_video.mp4 --top 5 --no-clip
python video_best_frames.py --input ma_video.mp4 --sample-rate 0.5 --min-score 0.2
```

## Hiérarchie de détection

Le script analyse la vidéo selon 5 niveaux de détection :

1. **N1 - Émotions positives** : DeepFace / FER
2. **N2 - Engagement humain** : MediaPipe Holistic
3. **N3 - Scène significative** : CLIP (vision-langage)
4. **N4 - Qualité technique** : Netteté, exposition
5. **N5 - Fallback temporel** : Sélection par intervalle + déduplication

## Modes de sélection

### Mode classique
- Sélectionne les **N meilleures frames** selon les scores IA
- Paramètres : nombre de photos, score minimum, intervalle d'échantillonnage

### Mode représentatif
- Découpe la vidéo en **segments temporels égaux**
- Extrait la meilleure frame de chaque segment
- Garantit une couverture homogène de toute la vidéo

## Configuration Streamlit

Les paramètres de l'application se configurent dans la barre latérale :

- **Dossier des vidéos** : Chemin absolu vers vos fichiers vidéo
- **Mode** : Classique ou représentatif (segments)
- **Modèle CLIP** : Activer/désactiver pour la qualité vs vitesse
- **Modèle Whisper** : tiny, base, small, medium, large
- **Langue** : FR, EN, ES, DE, IT, PT, auto

## Tests

```bash
python test_streamlit.py
```

## Licence

MIT
