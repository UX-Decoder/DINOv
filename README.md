# Visual In-Context Prompting
:grapes: \[[Read our arXiv Paper](https://arxiv.org/pdf/2311.13601.pdf)\] &nbsp; :apple: \[[Try our Demo](http://semantic-sam.xyzou.net:6099/)\] 

In this work, we introduce [DINOv](https://arxiv.org/pdf/2311.13601.pdf), a Visual In-Context Prompting framework for referring and generic segmentation tasks.

For visualization and demos, we also recommend trying [T-Rex demo link](https://deepdataspace.com/playground/ivp), which is another visual prompting tool in our team with similar properties as DINOv.

![teaser](https://github.com/UX-Decoder/DINOv/assets/34880758/f686dd20-a5aa-40fa-ad57-c4c69575853b)

### :hammer_and_wrench: Installation
```shell
pip3 install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
pip install git+https://github.com/cocodataset/panopticapi.git
git clone https://github.com/UX-Decoder/DINOv
cd DINOv
python -m pip install -r requirements.txt
```
#### :point_right: Launch a demo for visual in-context prompting
```shell
python demo_openset.py --ckpt /path/to/swinL/ckpt
```

# Openset segmentation
![generic_seg_vis](https://github.com/UX-Decoder/DINOv/assets/34880758/bfbe4d90-5be9-4fa5-a4e7-83f5c25f7f23)

# Panoptic segmentation
![panoptic_vis](https://github.com/UX-Decoder/DINOv/assets/34880758/c958f7b7-98c6-49cc-9c51-73bc6ad01808)

👉: **Related projects:**

* [Semantic-SAM](https://github.com/UX-Decoder/Semantic-SAM): We base on the mutli-granularity interactive segmentation to extract proposals.
* [Mask DINO](https://github.com/IDEA-Research/MaskDINO): We build upon Mask DINO which is a unified detection and segmentation model to implement our model.
* [SEEM](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once): Segment using a wide range of user prompts.

## :unicorn: Getting Started

### :mosque: Data preparation
We jointly train on COCO and SA-1B data. Please refer to [prepare SA-1B data](https://github.com/UX-Decoder/Semantic-SAM/blob/main/DATASET.md) and [prepare coco data](https://github.com/IDEA-Research/MaskDINO/blob/main/README.md).

For evaluation, you need to prepare 
* [ADE20K](https://github.com/IDEA-Research/MaskDINO/blob/main/datasets/README.md) for open-set segmentation evaluation.
* [DAVIS2017](https://davischallenge.org/davis2017/code.html) for refering segmentation (video object segmentation).

### :volcano: Model Zoo
The currently released checkpoints are trained with SA-1B and COCO data. 
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Training Dataset</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">PQ (COCO)</th>
<th valign="bottom">PQ (ADE)</th>
<th valign="bottom">download</th>

 <tr><td align="left">DINOv | <a href="configs/dinov_sam_coco_train.yaml">config</a></td>
<td align="center">SA-1B, COCO</td>
<td align="center">SwinT</td>
<td align="center">49.0</td>
<td align="center">19.4</td>
<td align="center"><a href="https://github.com/UX-Decoder/DINOv/releases/download/checkpoint/model_swinT.pth">model</a></td>
   
 <tr><td align="left">DINOv | <a href="configs/dinov_sam_coco_swinl_train.yaml">config</a></td>
<td align="center">SA-1B, COCO</td>
<td align="center">SwinL</td>
<td align="center">57.7</td>
<td align="center">23.2</td>
<td align="center"><a href="https://github.com/UX-Decoder/DINOv/releases/download/checkpoint/model_swinL.pth">model</a></td>

</tbody></table>

### :sunflower: Evaluation
We do detection evaluation on COCO val2017.
`$n` is the number of gpus you use

Process visual prompt embeddings for inference. We calculate the all the instance prompt embeddings of the validate set (you can also use the training set, but the processing time is much longer) and store them. Then we infrence by randomly selecting some visual prompts as in-context examples.
#### Evaluate Open-set detection and segmentation
* Infenrence script to get and store visual prompts
```shell
python train_net.py --eval_only --resume --eval_get_content_features --num-gpus 8 --config-file /path/to/configs COCO.TEST.BATCH_SIZE_TOTAL=8 MODEL.WEIGHTS=/path/to/weights OUTPUT_DIR=/path/to/outputs
```
* Inference script for open-set detection on COCO with visual prompts
```shell
python train_net.py --eval_only --resume --eval_visual_openset --num-gpus 8 --config-file /path/to/configs COCO.TEST.BATCH_SIZE_TOTAL=8 MODEL.WEIGHTS=/path/to/weights MODEL.DECODER.INFERENCE_EXAMPLE=16 OUTPUT_DIR=/path/to/outputs
```
* **configs** to use are `configs/dinov_sam_coco_train.yaml` for swinT and `configs/dinov_sam_coco_swinl_train.yaml` for swinL.
* For ADE20K data, use `configs/dinov_sam_ade_eval.yaml` and adjust the batchsize of ADE evaluation to the correct number.
* `OUTPUT_DIR` is the dir to store the visual prompt embeddings
* `INFERENCE_EXAMPLE` number of in-context examples to represent a category. Default set to 16.
#### Evaluate Refering segmentation on VOS
We evaluate under the `DAVIS 2017 Semi-supervised` setting, please refer to [davis2017-evaluation](https://github.com/davisvideochallenge/davis2017-evaluation) for more details.

The first step is to compute and store the results of DAVIS2017. We implement a navie memory-aware approach with our in-context visual prompting.
```shell
python train_net.py --eval_track_prev --eval_only --resume --num-gpus 8 --config-file configs/dinov_sam_coco_train.yaml DAVIS.TEST.BATCH_SIZE_TOTAL=8 OUTPUT_DIR=$outdir MODEL.WEIGHTS=/path/to/weights MODEL.DECODER.NMS_THRESHOLD=0.9 MODEL.DECODER.MAX_MEMORY_SIZE=9 OUTPUT_DIR=/path/to/outputs
```
The second step is to evaluate the semi-supervised results.
```shell
python evaluation_method.py --task semi-supervised --results_path /path/to/results --davis_path /path/to/davis/data
```
* We use MAX_MEMORY_SIZE = 9 by default (1 current frame token and 8 previous memory tokens)
### :star: Training 
We currently release the code of training on SA-1B and COCO. It can also support Objects365 and other datasets with minimal modifications. 
`$n` is the number of gpus you use
before running the training code, you need to specify your training data of SA-1B.
```shell
export DETECTRON2_DATASETS=/pth/to/cdataset  # path to coco, ade
export SAM_DATASET=/pth/to/sam_dataset  # patch to sa-1b data
export SAM_DATASET_START=$start
export SAM_DATASET_END=$end
```
We convert SA-1B data into 100 tsv files. `start`(int, 0-99) is the start of your SA-1B data index and `end`(int, 0-99) is the end of your data index.
You can refer to Semantic-SAM [json registration for SAM](datasets/registration/register_sam_json.py) for a reference on the data preparation. 

We recommend using total batchsize `64` for training, which provides enough postive and negative samples for contrastive learning.

For SwinT backbone
```shell
python train_net.py --resume --num-gpus 8 --config-file configs/dinov_sam_coco_train.yaml SAM.TRAIN.BATCH_SIZE_TOTAL=8 COCO.TRAIN.BATCH_SIZE_TOTAL=8
```
For SwinL backbone
```shell
python train_net.py --resume --num-gpus 8 --config-file configs/dinov_sam_coco_swinl_train.yaml SAM.TRAIN.BATCH_SIZE_TOTAL=8 COCO.TRAIN.BATCH_SIZE_TOTAL=8
```
* Please use multi-node training, i.e, 64 gpu for batchsize 64, where each gpu handle one SA-1B image and one coco image.
* By default, we do not use COCO data for referring segmentation training. You can set `MODEL.DECODER.COCO_TRACK=True` to enable this task, which can improve the referring segmentation performance on DAVIS. 
* We did not implement multi-image training for this task, which mean you can only put **one image on a gpu** for each datatype (i.e., one SA-1b and one COCO image).
 
# Model framework
![framework](https://github.com/UX-Decoder/DINOv/assets/34880758/8c756028-a7bd-42dc-8aa7-e6773fd60711)
![query_formulation](https://github.com/UX-Decoder/DINOv/assets/34880758/5ca36a9e-06ff-452c-b102-c05bebd5b5cf)

# Results
## Open-set detection and segmentation
<img width="826" alt="image" src="https://github.com/UX-Decoder/DINOv/assets/34880758/5d464654-bdb6-4c18-addb-9eb45d3968db">

## Video object segmentation
<img width="828" alt="image" src="https://github.com/UX-Decoder/DINOv/assets/34880758/9b1d5a06-af26-40b0-a9ac-72ab900e7382">

## :black_nib: Citation

If you find our work helpful for your research, please consider citing the following BibTeX entry.   

```bibtex
@article{li2023visual,
  title={Visual In-Context Prompting},
  author={Li, Feng and Jiang, Qing and Zhang, Hao and Ren, Tianhe and Liu, Shilong and Zou, Xueyan and Xu, Huaizhe and Li, Hongyang and Li, Chunyuan and Yang, Jianwei and others},
  journal={arXiv preprint arXiv:2311.13601},
  year={2023}
}



