#!/usr/bin/env bash
# GazeCLoD-GS viewer launcher for Linux (needs an NVIDIA GPU) and macOS.
#     bash gazeclod/run_viewer.sh                    (starts in truck)
#     bash gazeclod/run_viewer.sh --scene playroom
# Inside the viewer press T to switch scenes, ESC to quit.
ENV_NAME="${GAZECLOD_ENV:-clodgs}"
cd "$(dirname "$0")/.." || exit 1

if [ "$(uname)" = "Darwin" ]; then
  echo "Macs have no NVIDIA GPU and the splat renderer is CUDA-only, so the live viewer can't run here."
  vid=$(ls output/*_clod/gaze_lod/video_*_h264.mp4 2>/dev/null | head -n 1)
  if [ -n "$vid" ]; then
    echo "Opening the side-by-side demo video instead: $vid"
    open "$vid"
  else
    echo "Copy a demo video (output/<scene>_clod/gaze_lod/video_*_h264.mp4) over from the Windows PC to watch it."
  fi
  exit 0
fi

if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
  if ! command -v conda >/dev/null 2>&1; then
    for d in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
      [ -f "$d/etc/profile.d/conda.sh" ] && . "$d/etc/profile.d/conda.sh" && break
    done
  fi
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME" || { echo "Could not activate conda env '$ENV_NAME' (see: conda env list)"; exit 1; }
  else
    echo "conda not found, using whatever python is on PATH"
  fi
fi
exec python gazeclod/viewer.py "$@"
