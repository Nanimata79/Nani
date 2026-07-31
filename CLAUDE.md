# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this repo is

LongCat-Video: a 13.6B-parameter video generation model (Text-to-Video,
Image-to-Video, Video-Continuation) plus LongCat-Video-Avatar (audio-driven
character animation). Pure Python/PyTorch inference code — there is no
training code, test suite, or CI in this repo.

## Layout

- `longcat_video/` — model code
  - `pipeline_longcat_video.py`, `pipeline_longcat_video_avatar.py` — diffusers-style pipelines
  - `modules/` — DiT transformer, VAE (`autoencoder_kl_wan.py`), scheduler, LoRA, quantization, avatar modules
  - `block_sparse_attention/` — Block Sparse Attention kernels/interface
  - `context_parallel/` — multi-GPU context-parallel (Ulysses) support
  - `audio_process/` — wav2vec2 / audio front-end for avatar models
  - `utils/` — prompt enhancer, bucket config
- `run_demo_*.py` — one entry point per task (text2video, image2video, video-continuation, long-video, interactive-video, avatar single/multi audio)
- `run_streamlit.py` — Streamlit UI wrapping the same pipelines
- `setup_env.sh` — conda env + dependency install (CUDA 12.4 by default)
- `download_weights.sh` — pulls the three checkpoints from Hugging Face into `./weights/`
- `assets/` — example images/audio/JSON inputs for demos, logos, tech reports

## Running things

All demo scripts are launched with `torchrun` (they use `torch.distributed`
and read `RANK` from the environment) and take `--checkpoint_dir` pointing at
a downloaded weights folder, plus `--context_parallel_size` for multi-GPU and
`--enable_compile` to `torch.compile` the DiT:

```shell
torchrun run_demo_text_to_video.py --checkpoint_dir=./weights/LongCat-Video --enable_compile
torchrun --nproc_per_node=2 run_demo_text_to_video.py --context_parallel_size=2 --checkpoint_dir=./weights/LongCat-Video --enable_compile
```

Avatar demos take an `--input_json` (see `assets/avatar/*.json`) and, for the
1.5 model, `--model_type avatar-v1.5 --use_distill` (required for 1.5) and
optionally `--use_int8`. See the README's "Run LongCat-Video-Avatar" section
for the full flag set and tuning tips (audio CFG, `--ref_img_index`,
`--mask_frame_range`, `--resolution`).

There is no test suite — validating a change means actually running the
relevant `run_demo_*.py` (or `run_streamlit.py`) against downloaded weights
and inspecting the output `.mp4`. This requires an NVIDIA GPU; assume it's
unavailable unless the environment proves otherwise, and say so rather than
claiming a change "works" without having run it.

## Conventions

- Match the existing style: plain PyTorch/HF `diffusers`-flavored code, no
  type-checking config, no linter config, no test framework — don't introduce
  any of those unless asked.
- Distributed/context-parallel code (`context_parallel/`, `modules/attention.py`,
  `block_sparse_attention/`) is easy to break subtly (rank/shape mismatches
  that only surface at multi-GPU runtime). Double-check tensor shapes, dtypes
  (models run in `bfloat16`), and rank-dependent branches when touching it.
- Keep dependency versions pinned as in `requirements.txt` /
  `requirements_avatar.txt` unless a change specifically requires a bump.
