# Local Wan 2.2 TI2V-5B GGUF

```bash
conda activate video_gen
python download_model.py
python setup_runtime.py
python app.py
```

Open the local Gradio URL printed in the terminal. Upload an image, describe
its motion, and select Generate Video. The app uses the Q4_K_S transformer,
Q4_K_M UMT5 encoder, and Wan 2.2 VAE in `models/Wan2.2-TI2V-5B-GGUF`.
It does not use the older WAMU weights; those files are left untouched.

The native stable-diffusion.cpp runtime is built with Metal for Apple Silicon.
Building requires Xcode Command Line Tools (`xcode-select --install`) and
CMake (`python -m pip install cmake`). `setup_runtime.py` downloads the source
and GGML submodule and builds `sd-cli`. To use another compatible binary, set
`SD_CLI` to its absolute path before starting the app.

Generation uses both the uploaded first image and the text prompt. Defaults
are 512 pixels on the longest side, roughly one second, 20 Euler steps,
and guidance 5. Frame counts are rounded to 4n+1 at 24 FPS. This is standard
Wan 5B, not the old four-step Lightning model. Low-resolution previews may
have lower quality than native 720p, but reduce memory and time. No fixed
speed or 16 GB peak-memory guarantee has been measured for full-size clips.

The runtime memory-maps weights, offloads components between stages, runs
text encoding on CPU, and tiles VAE decoding. The transformer and VAE explicitly select Metal with
`--backend diffusion=metal,vae=metal,te=cpu`; this works across M3/M4 chips.
The CPU also handles image preparation and MP4 encoding. An unavailable Metal
backend causes an error instead of selecting CPU for the transformer.
CPU and GPU share physical RAM on Mac; long clips or large resolutions can still exceed memory. Only one
generation runs at a time. Each generation starts a native process and loads
weights; it releases that process's memory after completion.

Outputs are saved under `outputs/<run-id>/`: the resized input, a native
runtime log, and an H.264 MP4 for playback and download. The intermediate AVI
is deleted after successful MP4 conversion. Logs are retained on failure.
FFmpeg comes from the installed `imageio-ffmpeg` package. Output folders are
not automatically removed; delete unwanted folders to reclaim disk space.

The old A14B-specific controls (dual guidance, last frame, Spaces reservation,
RIFE interpolation, Diffusers schedulers) are omitted from this 5B interface.
Frame extraction remains available using a timestamp in the generated clip.
