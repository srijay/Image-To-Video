"""Wan TI2V GGUF inference through the local stable-diffusion.cpp CLI."""
import os
from pathlib import Path
import shutil
import subprocess
import uuid

import imageio_ffmpeg
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'models' / 'Wan2.2-TI2V-5B-GGUF'
TRANSFORMER = MODEL_DIR / 'Wan2.2-TI2V-5B-Q4_K_S.gguf'
TEXT_ENCODER = MODEL_DIR / 'umt5-xxl-encoder-Q4_K_M.gguf'
VAE = MODEL_DIR / 'split_files' / 'vae' / 'wan2.2_vae.safetensors'
FPS = 24


def cli_path():
    override = os.environ.get('SD_CLI')
    return Path(override).expanduser().resolve() if override else Path(
        shutil.which('sd-cli') or ROOT / 'vendor/stable-diffusion.cpp/build/bin/sd-cli')


def preflight():
    missing = [str(p.relative_to(ROOT)) for p in (TRANSFORMER, TEXT_ENCODER, VAE)
               if not p.is_file() or p.stat().st_size == 0]
    if missing:
        return 'Run python download_model.py. Missing files: ' + ', '.join(missing)
    if not cli_path().is_file() or not os.access(cli_path(), os.X_OK):
        return 'Native runtime missing. Run python setup_runtime.py, then restart the app.'
    return None


def frame_count(duration):
    # Wan's temporal VAE accepts 4n+1 frames.
    return max(5, min(121, 4 * round(float(duration) * FPS / 4) + 1))


def build_command(image, output, prompt, negative, width, height, frames, steps, guidance, seed):
    return [str(cli_path()), '-M', 'vid_gen', '--diffusion-model', str(TRANSFORMER),
            '--t5xxl', str(TEXT_ENCODER), '--vae', str(VAE), '-i', str(image),
            '-p', prompt, '-n', negative, '-W', str(width), '-H', str(height),
            '--video-frames', str(frames), '--fps', str(FPS), '--steps', str(steps),
            '--cfg-scale', str(guidance), '--sampling-method', 'euler',
            '--flow-shift', '3.0', '--seed', str(seed), '--offload-to-cpu',
            '--mmap', '--backend', 'diffusion=metal,vae=metal,te=cpu',
            '--vae-tiling', '--vae-tile-size', '256x256',
            '--conv-direct', 'auto,vae=on', '--flash-attn', 'on', '-o', str(output)]


def generate(image, prompt, negative, duration, steps, guidance, seed, max_dimension=512, progress=None):
    issue = preflight()
    if issue:
        raise RuntimeError(issue)
    if image is None:
        raise ValueError('Upload an input image.')
    if not prompt.strip():
        raise ValueError('Enter a motion prompt.')
    run_dir = ROOT / 'outputs' / uuid.uuid4().hex[:12]
    run_dir.mkdir(parents=True)
    image = ImageOps.exif_transpose(image).convert('RGB')
    ratio = max_dimension / max(image.size)
    width, height = [max(32, round(n * ratio / 32) * 32) for n in image.size]
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    input_path = run_dir / 'input.png'
    resized.save(input_path)
    raw_video = run_dir / 'generated.avi'
    frames = frame_count(duration)
    command = build_command(input_path, raw_video, prompt, negative, width, height,
                            frames, int(steps), float(guidance), int(seed))
    if progress:
        progress(0.05, desc=f'Generating {frames} frames at {width}×{height}; this can take several minutes')
    log_path = run_dir / 'generation.log'
    # Direct log output to disk so verbose native logs cannot fill a pipe.
    with log_path.open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if code != 0 or not raw_video.is_file():
        tail = log_path.read_text(errors='replace')[-2500:]
        raise RuntimeError(f'Generation failed (exit {code}). Log: {log_path}\n{tail}')
    if progress:
        progress(0.95, desc='Encoding MP4')
    output = run_dir / 'video.mp4'
    converted = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-i', str(raw_video),
                               '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                               '-crf', '20', '-movflags', '+faststart', str(output)],
                              capture_output=True, text=True)
    if converted.returncode or not output.is_file():
        raise RuntimeError(f'MP4 encoding failed: {converted.stderr[-2000:]}')
    raw_video.unlink()
    if progress:
        progress(1, desc='Complete')
    return str(output)
