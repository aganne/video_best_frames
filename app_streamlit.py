"""
app_streamlit.py
----------------
Interface web Streamlit pour video_best_frames.py.

Permet de traiter un dossier de vidéos locales séquentiellement,
de visualiser les résultats et de télécharger les photos extraites.

Lancement :
  pip install streamlit
  streamlit run app_streamlit.py
"""

import json
import logging
import time
import traceback
from pathlib import Path

import streamlit as st

# ── Logging vers Streamlit ─────────────────────────────────────────────────────

class StreamlitLogHandler(logging.Handler):
    """Redirige les logs vers la liste de messages stockée en session."""

    def emit(self, record):
        msg = self.format(record)
        level = record.levelname
        if "log_messages" in st.session_state:
            st.session_state.log_messages.append((level, msg))


def setup_logging():
    """Configure le logging pour rediriger vers Streamlit + console."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, StreamlitLogHandler) for h in root.handlers):
        handler = StreamlitLogHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                                datefmt="%H:%M:%S"))
        root.addHandler(handler)


# ── Utilitaires ────────────────────────────────────────────────────────────────

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mts", ".ts"}


def list_videos(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


# ── Import du pipeline ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Chargement des modèles IA…")
def load_pipeline(use_clip, enable_transcription, whisper_model):
    """Charge le pipeline (modèles IA) une seule fois.

    Mis en cache uniquement sur les paramètres qui influencent les modèles
    chargés. Les réglages par lancement (sample_rate, top_n, min_score, mode
    segments, langue Whisper) sont appliqués via `configure()` — ils ne
    déclenchent donc PAS de rechargement des modèles.
    """
    from video_best_frames import VideoBestFrames
    return VideoBestFrames(
        use_clip=use_clip,
        dedup_threshold=5,
        enable_transcription=enable_transcription,
        whisper_model=whisper_model,
    )

@st.cache_resource(show_spinner="Chargement du modèle Whisper…")
def load_transcriber(whisper_model: str, whisper_language: str):
    """Charge TranscriptionEngine une seule fois (mis en cache)."""
    from video_best_frames import TranscriptionEngine
    return TranscriptionEngine(model_name=whisper_model, language=whisper_language)


def main():
    st.set_page_config(
        page_title="Video Best Frames",
        page_icon="🎬",
        layout="wide",
    )
    setup_logging()
    logger = logging.getLogger(__name__)

    # ── État de session ────────────────────────────────────────────────────────
    if "log_messages" not in st.session_state:
        st.session_state.log_messages = []
    if "results_by_video" not in st.session_state:
        st.session_state.results_by_video = {}   # {video_name: [exported_frames]}
    if "processing_done" not in st.session_state:
        st.session_state.processing_done = False

    # ── Sidebar — paramètres ───────────────────────────────────────────────────
    with st.sidebar:
        st.title("🎬 Video Best Frames")
        st.divider()

        folder_input = st.text_input(
            "📂 Dossier des vidéos",
            placeholder="C:/mes_videos",
            help="Chemin absolu du dossier contenant vos fichiers vidéo.",
        )

        st.subheader("⚙️ Mode de sélection")
        segment_mode = st.toggle(
            "Mode représentatif (segments)",
            value=False,
            help="Découpe la vidéo en tranches temporelles égales et garde la meilleure photo "
                 "de chaque tranche → couverture homogène de toute la vidéo.",
        )

        if segment_mode:
            segment_duration = st.slider(
                "Durée d'un segment (s)", 1.0, 30.0, 6.0, step=0.5,
                help="Une photo sera extraite par tranche de cette durée. "
                     "Ex : vidéo de 2 min ÷ 6 s = ~20 photos.",
            )
            sample_rate = st.slider(
                "Finesse d'analyse interne (s)", 0.1, 2.0, 0.2, step=0.1,
                help="Échantillonnage à l'intérieur de chaque segment pour trouver la meilleure frame.",
            )
            # Valeurs neutres — non utilisées en mode segments
            top_n = 9999
            min_score = 0.0
        else:
            top_n = st.slider("Nombre de photos par vidéo", 1, 30, 10)
            sample_rate = st.slider(
                "Intervalle d'échantillonnage (s)", 0.1, 5.0, 0.5, step=0.1,
                help="1 frame analysée toutes les N secondes.",
            )
            min_score = st.slider(
                "Score minimum", 0.0, 1.0, 0.50, step=0.01,
                help="Frames en dessous de ce seuil sont ignorées.",
            )
            segment_duration = 6.0

        st.subheader("⚙️ Paramètres avancés")
        use_clip = st.toggle("Activer CLIP (N3)", value=True,
                             help="CLIP améliore la qualité mais est plus lent.")
        dedup_threshold = st.slider(
            "Seuil déduplication (pHash)", 1, 20, 5,
            help="Plus bas = moins de doublons tolérés. (Ignoré en mode segments.)",
        )

        st.subheader("🎙️ Transcription audio")
        whisper_model = st.selectbox(
            "Modèle Whisper",
            ["tiny", "base", "small", "medium", "large"],
            index=4,
            help="Plus le modèle est grand, plus la transcription est précise mais lente.",
        )
        whisper_language = st.selectbox(
            "Langue",
            ["fr", "en", "es", "de", "it", "pt", "auto"],
            index=0,
            help="Forcer la langue évite les erreurs de détection automatique. 'auto' = détection automatique.",
        )
        if whisper_language == "auto":
            whisper_language = None
        enable_transcription = st.toggle(
            "Inclure la transcription lors de l'extraction",
            value=False,
            help="Lance Whisper après l'extraction des frames.",
        )

        output_root = st.text_input(
            "📁 Dossier de sortie",
            value="./best_photos",
            help="Les sous-dossiers par vidéo y seront créés.",
        )

        st.divider()
        launch_btn      = st.button("🚀 Lancer le traitement", type="primary", use_container_width=True)
        transcribe_btn  = st.button("🎙️ Transcrire uniquement", use_container_width=True,
                                    help="Lance uniquement la transcription Whisper sans extraire de frames.")

    # ── Zone principale ────────────────────────────────────────────────────────
    st.header("🎬 Video Best Frames")

    if not folder_input:
        st.info("👈 Saisissez le chemin du dossier vidéo dans la barre latérale pour commencer.")
        return

    folder = Path(folder_input)
    if not folder.exists() or not folder.is_dir():
        st.error(f"Dossier introuvable : `{folder}`")
        return

    videos = list_videos(folder)
    if not videos:
        st.warning(f"Aucune vidéo trouvée dans `{folder}` (extensions : {', '.join(VIDEO_EXTENSIONS)})")
        return

    # ── Liste des vidéos avec cases à cocher ──────────────────────────────────
    st.subheader(f"📋 Vidéos détectées ({len(videos)})")

    col_all, col_none = st.columns([1, 1])
    with col_all:
        if st.button("✅ Tout sélectionner"):
            for v in videos:
                st.session_state[f"sel_{v.name}"] = True
    with col_none:
        if st.button("☐ Tout désélectionner"):
            for v in videos:
                st.session_state[f"sel_{v.name}"] = False

    selected_videos = []
    with st.expander("Liste des vidéos", expanded=True):
        for v in videos:
            key = f"sel_{v.name}"
            if key not in st.session_state:
                st.session_state[key] = True
            checked = st.checkbox(v.name, key=key)
            if checked:
                selected_videos.append(v)

    if not selected_videos:
        st.warning("Aucune vidéo sélectionnée.")
        return

    st.caption(f"{len(selected_videos)} vidéo(s) sélectionnée(s) sur {len(videos)}")

    # ── Traitement ─────────────────────────────────────────────────────────────
    if launch_btn:
        st.session_state.log_messages = []
        st.session_state.results_by_video = {}
        st.session_state.processing_done = False

        pipeline = load_pipeline(use_clip, enable_transcription, whisper_model)
        pipeline.configure(
            sample_rate=sample_rate,
            top_n=top_n,
            min_score=min_score,
            segment_mode=segment_mode,
            segment_duration=segment_duration,
            dedup_threshold=dedup_threshold,
            whisper_language=whisper_language,
        )

        progress_bar  = st.progress(0, text="Initialisation…")
        status_text   = st.empty()
        log_container = st.expander("📜 Logs en temps réel", expanded=True)

        total = len(selected_videos)
        global_start = time.time()

        for i, video_path in enumerate(selected_videos):
            video_name  = video_path.stem
            out_dir     = Path(output_root) / video_name
            pct         = i / total
            progress_bar.progress(pct, text=f"Traitement {i+1}/{total} : {video_path.name}")
            status_text.info(f"⏳ Analyse de **{video_path.name}**…")

            t0 = time.time()
            try:
                results = pipeline.process(str(video_path), str(out_dir))
                elapsed = time.time() - t0
                st.session_state.results_by_video[video_path.name] = {
                    "results":  results,
                    "out_dir":  out_dir,
                    "elapsed":  elapsed,
                    "error":    None,
                }
                # Affichage des logs accumulés
                with log_container:
                    for level, msg in st.session_state.log_messages[-20:]:
                        if level == "WARNING":
                            st.warning(msg)
                        elif level == "ERROR":
                            st.error(msg)
                        else:
                            st.text(msg)

            except Exception as e:
                elapsed = time.time() - t0
                tb = traceback.format_exc()
                st.session_state.results_by_video[video_path.name] = {
                    "results":  [],
                    "out_dir":  out_dir,
                    "elapsed":  elapsed,
                    "error":    str(e),
                }
                st.error(f"❌ Erreur sur {video_path.name} : {e}")
                with st.expander("🔍 Traceback complet", expanded=True):
                    st.code(tb, language="python")
                logger.error("Crash sur %s :\n%s", video_path.name, tb)

        total_elapsed = time.time() - global_start
        progress_bar.progress(1.0, text=f"✅ Terminé en {total_elapsed:.1f}s")
        status_text.success(f"Traitement terminé : {len(selected_videos)} vidéos en {total_elapsed:.1f}s")
        st.session_state.processing_done = True

    # ── Transcription uniquement ────────────────────────────────────────────────
    if transcribe_btn:
        st.session_state.log_messages = []
        transcriber = load_transcriber(whisper_model, whisper_language)

        st.divider()
        st.header("🎙️ Transcriptions")
        progress_bar = st.progress(0, text="Initialisation…")
        total = len(selected_videos)
        global_start = time.time()

        for i, video_path in enumerate(selected_videos):
            video_name = video_path.stem
            out_dir    = Path(output_root) / video_name
            out_dir.mkdir(parents=True, exist_ok=True)
            progress_bar.progress(i / total, text=f"Transcription {i+1}/{total} : {video_path.name}")

            try:
                transcript_path = transcriber.transcribe(video_path, out_dir)
                with st.expander(f"✅ {video_path.name}", expanded=False):
                    st.text(transcript_path.read_text(encoding="utf-8"))
                    with open(transcript_path, encoding="utf-8") as f:
                        st.download_button(
                            "⬇️ Télécharger le transcript",
                            data=f.read(),
                            file_name=f"{video_name}_transcript.txt",
                            mime="text/plain",
                            key=f"tr_only_{video_name}",
                        )
            except Exception as e:
                tb = traceback.format_exc()
                st.error(f"❌ Erreur sur {video_path.name} : {e}")
                with st.expander("🔍 Traceback complet", expanded=True):
                    st.code(tb, language="python")
                logger.error("Crash transcription %s :\n%s", video_path.name, tb)

        total_elapsed = time.time() - global_start
        progress_bar.progress(1.0, text=f"✅ Terminé en {total_elapsed:.1f}s")




if __name__ == "__main__":
    main()
