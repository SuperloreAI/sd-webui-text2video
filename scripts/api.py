from fastapi import FastAPI, Body

import gradio as gr

from modules.api.models import *
from modules.api import api
from scripts.video_audio_utils import find_ffmpeg_binary

from scripts.text2vid import process

def text2video_api(_: gr.Blocks, app: FastAPI):
    @app.get("/text2video/version")
    async def version():
        return {"version": 1}

    @app.post("/text2video/inference")
    async def inference(
        prompt: str = Body(title='Prompt for inference'),
        negative_prompt: str = Body("text, watermark, copyright, blurry", title='Negative prompt for inference'),
        steps: int = Body(30, title='Steps for inference'),
        cfg_scale: int = Body(7, title='CFG Scale for inference'),
        width: int = Body(256, title='Width for inference'),
        height: int = Body(256, title='Height for inference'),
        seed: int = Body(-1, title='Seed for inference'),
        frames: int = Body(24, title='Frames for inference'),
        fps: int = Body(24, title='FPS for inference'),
    ):
        print(f"text2video called: \
                prompt: {prompt} \n \
                negative_prompt: {negative_prompt} \n \
                steps: {steps} \n \
                cfg_scale: {cfg_scale} \n \
                width: {width} \n \
                height: {height} \n \
                seed: {seed} \n \
                frames: {frames}")
            
        try:
            ffmpeg_location = find_ffmpeg_binary()

            result = process(prompt=prompt,
                    n_prompt=negative_prompt,
                    steps=steps,
                    frames=frames,
                    seed=seed,
                    cfg_scale=cfg_scale,
                    width=width,
                    height=height,
                    fps=fps,
                    skip_video_creation=False, 
                    ffmpeg_location=ffmpeg_location,
                    ffmpeg_crf='17',
                    ffmpeg_preset='slow',
                    add_soundtrack = 'None',  # ["File","Init Video"]
                    soundtrack_path = "https://deforum.github.io/a1/A1.mp3",
                    eta=0,
                    prompt_v=prompt,
                    n_prompt_v=negative_prompt,
                    steps_v=steps, 
                    frames_v=frames, 
                    seed_v=seed, 
                    cfg_scale_v=cfg_scale, 
                    width_v=width, 
                    height_v=height,
                    eta_v=0)
        
            print('results', result)
            
            return {"info": "Success", "data": {"video_files": result}}
        except Exception as e:
            print(e)
            return {"info": "Error", "data": {"reason": str(e)}}

try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(text2video_api)
except:
    pass