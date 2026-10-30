"""Local image + text to video UI for Wan 2.2 TI2V-5B GGUF."""
import random

import cv2
import gradio as gr

from gguf_runtime import generate, preflight


def generate_video(image, prompt, negative, duration, steps, guidance, seed, randomize, resolution,
                   progress=gr.Progress()):
    seed = random.randint(0, 2**31 - 1) if randomize else int(seed)
    try:
        output = generate(image, prompt, negative, duration, steps, guidance, seed,
                          max_dimension=int(resolution), progress=progress)
    except (RuntimeError, ValueError, OSError) as error:
        raise gr.Error(str(error)) from error
    return output, output, seed


def extract_frame(video_path, timestamp):
    if not video_path:
        raise gr.Error('Generate a video first.')
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0, float(timestamp)) * 1000)
        ok, frame = cap.read()
        if not ok:
            raise gr.Error('No frame at this timestamp. Choose an earlier time.')
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown('## Wan 2.2 — Image + Text to Video')
    gr.Markdown('Upload an image and describe its motion. Start with a short clip at 512 pixels. '
                'Rendering on a 16 GB Mac can take several minutes or longer.')
    gr.Markdown('Video generation uses the Apple GPU through Metal. Text encoding uses the CPU.')
    issue = preflight()
    if issue:
        gr.Markdown(f'**Setup required:** {issue}')
    with gr.Row():
        with gr.Column():
            image = gr.Image(type='pil', label='Input image', sources=['upload', 'clipboard'])
            prompt = gr.Textbox(label='Motion prompt', value='Cinematic motion, the subject moves gently, smooth camera movement')
            duration = gr.Slider(0.2, 5, value=1, step=0.1, label='Duration (seconds)',
                                 info='24 FPS; frame count is rounded to 4n+1.')
            with gr.Accordion('Advanced settings', open=False):
                resolution = gr.Dropdown([384, 512, 640, 832], value=512, label='Longest side (pixels)',
                                         info='Smaller previews reduce memory and time; detail may be lower than native 720p.')
                steps = gr.Slider(1, 50, value=20, step=1, label='Inference steps',
                                  info='This is standard Wan 5B, not the four-step Lightning model.')
                guidance = gr.Slider(1, 10, value=5, step=0.5, label='Prompt guidance')
                negative = gr.Textbox(label='Negative prompt', value='blurry, distorted, static, subtitles, watermark')
                seed = gr.Number(value=42, precision=0, minimum=0, maximum=2**31-1, label='Seed')
                randomize = gr.Checkbox(value=True, label='Randomize seed')
            button = gr.Button('Generate Video', variant='primary', interactive=issue is None)
        with gr.Column():
            video = gr.Video(label='Generated video', format='mp4', interactive=False)
            download = gr.File(label='Download MP4')
            timestamp = gr.Number(value=0, minimum=0, label='Frame timestamp (seconds)')
            grab = gr.Button('Use frame as input')
    button.click(generate_video, [image, prompt, negative, duration, steps, guidance, seed, randomize, resolution],
                 [video, download, seed], concurrency_limit=1)
    grab.click(extract_frame, [video, timestamp], image)

if __name__ == '__main__':
    demo.queue(default_concurrency_limit=1).launch(server_name='127.0.0.1', show_error=True)
