# --------------------------------------------------------
# Semantic-SAM: Segment and Recognize Anything at Any Granularity
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Hao Zhang (hzhangcx@connect.ust.hk)
# --------------------------------------------------------
# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Feng Li (fliay@connect.ust.hk)
# --------------------------------------------------------


import gradio as gr
import torch
import argparse

from dinov.BaseModel import BaseModel
from dinov import build_model
from utils.arguments import load_opt_from_config_file

from demo import task_openset

def parse_option():
    parser = argparse.ArgumentParser('DINOv Demo', add_help=False)
    parser.add_argument('--conf_files', default="configs/dinov_sam_coco_swinl_train.yaml", metavar="FILE", help='path to config file', )
    parser.add_argument('--ckpt', default="", metavar="FILE", help='path to ckpt', required=True)
    parser.add_argument('--port', default=6099, type=int, help='path to ckpt', )
    args = parser.parse_args()

    return args


class ImageMask(gr.components.Image):
    """
    Sets: source="canvas", tool="sketch"
    """

    is_template = True

    def __init__(self, **kwargs):
        super().__init__(source="upload", tool="sketch", interactive=True, **kwargs)

    def preprocess(self, x):
        return super().preprocess(x)


'''
build args
'''
args = parse_option()

'''
build model
'''

sam_cfg=args.conf_files

opt = load_opt_from_config_file(sam_cfg)

model_sam = BaseModel(opt, build_model(opt)).from_pretrained(args.ckpt).eval().cuda()

@torch.no_grad()
def inference(generic_vp1, generic_vp2, generic_vp3, generic_vp4,
                   generic_vp5, generic_vp6, generic_vp7, generic_vp8, image2,*args, **kwargs):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        model=model_sam
        a= task_openset(model, generic_vp1, generic_vp2, generic_vp3, generic_vp4,
                   generic_vp5, generic_vp6, generic_vp7, generic_vp8, image2, *args, **kwargs)
        return a


'''
launch app
'''
title = "DINOv: Visual In-Context Prompting"

article = "The Demo is Run on DINOv."

demo = gr.Blocks()
image_tgt=gr.components.Image(label="Target Image ",type="pil",brush_radius=15.0)
gallery_output=gr.components.Image(label="Results Image ",type="pil",brush_radius=15.0)

generic_vp1 = ImageMask(label="scribble on refer Image 1",type="pil",brush_radius=15.0)
generic_vp2 = ImageMask(label="scribble on refer Image 2",type="pil",brush_radius=15.0)
generic_vp3 = ImageMask(label="scribble on refer Image 3",type="pil",brush_radius=15.0)
generic_vp4 = ImageMask(label="scribble on refer Image 5",type="pil",brush_radius=15.0)
generic_vp5 = ImageMask(label="scribble on refer Image 6",type="pil",brush_radius=15.0)
generic_vp6 = ImageMask(label="scribble on refer Image 7",type="pil",brush_radius=15.0)
generic_vp7 = ImageMask(label="scribble on refer Image 8",type="pil",brush_radius=15.0)
generic_vp8 = ImageMask(label="scribble on refer Image 9",type="pil",brush_radius=15.0)
generic = gr.TabbedInterface([
                        generic_vp1, generic_vp2, generic_vp3, generic_vp4,
                        generic_vp5, generic_vp6, generic_vp7, generic_vp8
                    ], ["1", "2", "3", "4", "5", "6", "7", "8"])

title='''
# DINOv: Visual In-Context Prompting

# [[Read our arXiv Paper](https://arxiv.org/pdf/2311.13601.pdf)\] &nbsp; \[[Github page](https://github.com/UX-Decoder/DINOv)\] 
'''

with demo:
    with gr.Row():
        with gr.Column(scale=3.0):
            generation_tittle = gr.Markdown(title)
            image_tgt.render()
            generic.render()
            with gr.Row(scale=2.0):
                clearBtn = gr.ClearButton(
                    components=[image_tgt])
                runBtn = gr.Button("Run")
        with gr.Column(scale=5.0):

            gallery_tittle = gr.Markdown("# Open-set results.")
            with gr.Row(scale=9.0):
                gallery_output.render()

            example = gr.Examples(
                examples=[
                    ["demo/examples/bags.jpg"],
                    ["demo/examples/img.png"],
                    ["demo/examples/corgi2.jpg"],
                    ["demo/examples/ref_cat.jpeg"],
                ],
                inputs=image_tgt,
                cache_examples=False,
            )

    title = title,
    article = article,
    allow_flagging = 'never',

    runBtn.click(inference, inputs=[generic_vp1, generic_vp2, generic_vp3, generic_vp4,
                   generic_vp5, generic_vp6, generic_vp7, generic_vp8, image_tgt],
              outputs = [gallery_output])



demo.queue().launch(share=True,server_port=args.port)

