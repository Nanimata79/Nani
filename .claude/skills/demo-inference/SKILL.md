---
name: demo-inference
description: Reference for launching LongCat-Video's run_demo_*.py / run_streamlit.py scripts with the right torchrun flags, checkpoint paths, and avatar-specific options.
---

# Demo Inference

Quick reference for running LongCat-Video's inference entry points. Use this
instead of re-deriving flags from scratch each time.

## Common pattern

All non-avatar demos share the same shape:

```shell
# Single-GPU
torchrun run_demo_<task>.py --checkpoint_dir=./weights/LongCat-Video --enable_compile

# Multi-GPU (context-parallel)
torchrun --nproc_per_node=N run_demo_<task>.py --context_parallel_size=N --checkpoint_dir=./weights/LongCat-Video --enable_compile
```

`<task>` is one of: `text_to_video`, `image_to_video`, `video_continuation`,
`long_video`, `interactive_video`.

- `--checkpoint_dir` must point at a `download_weights.sh`-produced folder.
- `--context_parallel_size` must equal `--nproc_per_node`.
- `--enable_compile` triggers a `torch.compile` of the DiT — slower first
  call, faster subsequent ones; drop it while iterating on correctness.

## Avatar demos

`run_demo_avatar_single_audio_to_video.py` (single speaker) and
`run_demo_avatar_multi_audio_to_video.py` (multi speaker) additionally need:

- `--checkpoint_dir=./weights/LongCat-Video-Avatar` (v1.0) or
  `./weights/LongCat-Video-Avatar-1.5` (v1.5)
- `--input_json=assets/avatar/<single|multi>_example_*.json`
- `--model_type avatar-v1.0` (default, wav2vec2) or `avatar-v1.5` (Whisper,
  **requires** `--use_distill`)
- optional: `--use_int8` (v1.5 only, lower VRAM), `--num_segments` +
  `--ref_img_index` + `--mask_frame_range` for video-continuation-style
  multi-segment generation, `--resolution` (480p/720p), `--stage_1`
  (`at2v` audio-text-to-video or `ai2v` audio-image-to-video, single-audio
  script only)

Tuning notes (see README for the full list):
- Audio CFG 3–5 for best lip-sync.
- `--ref_img_index` 0–24 for consistency, ~30 to reduce repeated actions.
- `--mask_frame_range` up from default 3 also reduces repeated actions but
  can introduce artifacts at high values.
- Dual-audio `--audio_type para` needs equal-length clips (summed); `add`
  concatenates with silence padding, person1 first.

## Streamlit UI

```shell
streamlit run ./run_streamlit.py --server.fileWatcherType none --server.headless=false
```

Wraps the same pipelines interactively — useful for eyeballing output
quality without writing a one-off script.
