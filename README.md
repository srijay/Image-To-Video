# Image + Text to Video on Apple Silicon

A local Gradio app that turns an **input image and a motion prompt into an MP4
video**, using Wan 2.2 TI2V-5B with GGUF quantization and the Apple GPU through
Metal.

Upload a starting image, describe what should move, adjust the generation
settings, and download the resulting clip. You can also extract a frame from
a generated video and use it as the next input image.

This project adapts the original
[kulkas2pintu/wan777 Hugging Face Space](https://huggingface.co/spaces/kulkas2pintu/wan777)
for local Apple Silicon use. The original app used a WAMU Wan 2.2 A14B Lightning
pipeline with CUDA, Spaces, FP8 quantization, and compiled transformer artifacts.
The current app uses a smaller, standard Wan 2.2 5B model through
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp).

## How it works

1. The Gradio interface receives an image and a text prompt.
2. Python applies image orientation, converts it to RGB, and resizes it while
   preserving its approximate aspect ratio. Dimensions are rounded to multiples of 32.
3. A native `sd-cli` process loads the local GGUF transformer, quantized text
   encoder, and Wan 2.2 VAE. The uploaded image conditions the first frame;
   the prompt describes the desired motion and scene.
4. The runtime generates frames using Euler sampling at 24 FPS. Frame counts
   are rounded to the VAE-compatible form `4n + 1`.
5. FFmpeg converts the intermediate AVI into an H.264 MP4 for browser playback
   and download.

Device placement is explicit in `gguf_runtime.py`:

```text
--backend diffusion=metal,vae=metal,te=cpu
```

The video transformer and VAE use the Apple GPU through Metal. Text encoding,
image preparation, and MP4 export use the CPU. Weight memory mapping, component
offloading, and VAE tiling help manage memory; CPU and GPU still share the same
physical memory on a Mac.

## Requirements

- Apple Silicon Mac with Metal support. Development and tiny generation tests
  were performed on an **M4 with 16 GB unified memory**; M3 uses the same Metal
  backend but has not been benchmarked here.
- Python 3.11; the environment used for development is named `video_gen`.
- Git, Xcode Command Line Tools, and CMake to build the native runtime.
- About **8.2 GB for the model files**, plus space for dependencies, runtime
  source/build files, and generated videos.
- Internet access for initial package installation, model downloads, and runtime setup.

A 16 GB Mac can run the small tested workload, but longer clips and higher
resolutions can exceed available memory. Quantized weight size is not the total
memory needed during generation.

## Setup

```bash
git clone https://github.com/srijay/Image-To-Video.git
cd Image-To-Video

conda create -n video_gen python=3.11 pip -y
conda activate video_gen
python -m pip install -r requirements-mac.lock.txt
```

If Xcode Command Line Tools are not installed, run:

```bash
xcode-select --install
```

Download the model components, build the runtime, and launch the app:

```bash
python download_model.py
python setup_runtime.py
python app.py
```

Open the local URL printed by Gradio, usually `http://127.0.0.1:7860`.
Rerun `download_model.py` to resume an interrupted download. After downloading
files or building a missing runtime, restart the app to refresh its setup check.

`setup_runtime.py` clones stable-diffusion.cpp into `vendor/`, checks out a pinned
revision on a fresh clone, initializes GGML, and builds `sd-cli` with Metal enabled.
An existing source checkout is reused. To supply another compatible binary:

```bash
export SD_CLI=/absolute/path/to/sd-cli
python app.py
```

Use `requirements-mac.lock.txt` for the current environment. The original
`requirements.txt`, `packages.txt`, and `aoti.py` remain from the hosted app and
are not used by the current local setup.

## Model components

The download script fetches only these three files:

| Component | Source | File | Approximate size |
| --- | --- | --- | --- |
| Video transformer | [QuantStack/Wan2.2-TI2V-5B-GGUF](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF) | `Wan2.2-TI2V-5B-Q4_K_S.gguf` | 3.12 GB |
| Text encoder | [city96/umt5-xxl-encoder-gguf](https://huggingface.co/city96/umt5-xxl-encoder-gguf) | `umt5-xxl-encoder-Q4_K_M.gguf` | 3.66 GB |
| VAE | [Comfy-Org/Wan-2.2](https://huggingface.co/Comfy-Org/Wan-2.2) | `split_files/vae/wan2.2_vae.safetensors` | 1.4 GB |

Files are stored in `models/Wan2.2-TI2V-5B-GGUF/`. This is a single-transformer
5B pipeline; it does not use the older WAMU A14B download or its two transformer
folders. Existing WAMU files are left untouched.

## Usage and performance

Upload an image and enter a prompt describing motion, for example:

> The subject slowly turns toward the camera, hair moving gently in the breeze,
> with a smooth cinematic camera push-in.

The UI defaults to a one-second clip, 512 pixels on the longest side, 20 steps,
and guidance 5. These are adjustable under **Advanced settings**. Output duration
may differ slightly from the requested duration because frame counts are rounded.

**Local video generation is slow on the tested base M4.** Q4 quantization reduces
weight memory; it does not make this a real-time video generator. Standard Wan
5B also does not inherit the original WAMU Lightning model's four-step behavior.

Recorded local logs include:

- A tiny 128 × 64, five-frame, one-step test with guidance 1 completed native
  generation in about 71 seconds using explicit Metal placement and produced
  a valid 24 FPS MP4. This verifies the integration, not production quality.
- A larger 20-step run spent about **31 minutes in denoising alone**, plus image
  encoding, text encoding, and decoding. This is an observed run, not a controlled
  benchmark or a prediction for every image.

For an initial preview, try 384 pixels, 0.5 seconds, and 8–10 steps. Fewer steps
can reduce quality, and lower resolutions can perform worse visually than the
model's native 720p use. Increasing duration adds frames and can substantially
increase memory use and rendering time. Only one generation runs at a time.

Each request starts a new native process and loads the weights. First-use Metal
shader compilation can add startup time. Logs contain per-stage timing and
sampling ETA; the UI currently shows coarse generation and encoding progress.

## Outputs and troubleshooting

Each request creates a directory under `outputs/<run-id>/` containing:

- `input.png`: resized conditioning image.
- `generation.log`: native runtime output, timings, and errors.
- `video.mp4`: final video after successful generation and conversion.

The intermediate `generated.avi` is deleted after successful MP4 conversion.
Failed requests retain their logs and any intermediate files. Output directories
are not automatically deleted.

If **Generate Video** is disabled, read the setup message above the controls.
The model files or native runtime may be missing. Run the download/setup commands
and restart the app. If a generation fails, inspect its `generation.log` and try
a shorter clip or lower resolution. In Metal initialization logs, the GPU should
be identified as an Apple device such as `MTL0 (Apple M4)`.

The app binds to `127.0.0.1`. Generation uses local model files and the local native
runtime. Models, runtime builds, generated media, and the Python environment are
excluded from Git.

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Gradio interface, generation callback, and frame extraction |
| `gguf_runtime.py` | Model paths, Metal CLI arguments, native process execution, MP4 conversion |
| `download_model.py` | Downloads the Q4 transformer, Q4 text encoder, and VAE |
| `setup_runtime.py` | Downloads and builds the Metal runtime |
| `requirements-mac.lock.txt` | Python dependency versions from the development environment |
| `LOCAL_SETUP.md` | Additional local setup and runtime notes |

## Attribution

Model credit belongs to the Wan team and the authors of the linked quantized
conversions. Native inference is provided by stable-diffusion.cpp and GGML;
Gradio provides the interface. Consult each upstream project and model repository
for its license and usage terms. This repository does not distribute model weights.
