"""Download Wan 2.2 TI2V-5B Q4 and its companions for a GGUF runtime.

Run with: conda activate video_gen && python download_model.py
Rerun the same command to resume interrupted downloads.
These files are used by app.py through stable-diffusion.cpp.
"""
from pathlib import Path

from huggingface_hub import hf_hub_download

MODEL_DIR = Path(__file__).resolve().parent / "models" / "Wan2.2-TI2V-5B-GGUF"
MODEL_FILES = (
    ("QuantStack/Wan2.2-TI2V-5B-GGUF", "Wan2.2-TI2V-5B-Q4_K_S.gguf"),
    ("city96/umt5-xxl-encoder-gguf", "umt5-xxl-encoder-Q4_K_M.gguf"),
    ("Comfy-Org/Wan-2.2", "split_files/vae/wan2.2_vae.safetensors"),
)


def main():
    print("Downloading three files (~8.2 GB total); existing downloads are reused.", flush=True)
    print(f"Destination: {MODEL_DIR}", flush=True)
    for index, (repo_id, filename) in enumerate(MODEL_FILES, start=1):
        print(f"[{index}/{len(MODEL_FILES)}] {filename}", flush=True)
        path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=MODEL_DIR)
        print(f"Ready: {path}", flush=True)
    print("Download complete. Run python setup_runtime.py if needed, then python app.py.")


if __name__ == "__main__":
    main()
