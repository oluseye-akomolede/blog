---
layout: post
title: "How HCM Turns a Camera Frame into a Mask"
date: 2026-11-11
---

A segmentation model looks at an image and decides, for every pixel, what it belongs to. HCM does that for street scenes, on a camera-sized image, fast enough to run on a small embedded GPU. This post is about what the model is, what each block in it computes, and what this implementation changed from the paper it comes from.

## What HCM is

HCM is an image-segmentation model. For each pixel of a camera frame it decides whether the pixel belongs to an object or to the background. The training target is the union of every object in the image, and the model has two output classes ([model_utils.py:1860](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/ecs_hcm_model/core/model_utils.py#L1860), [trainer.py:3858](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/ecs_hcm_model/core/trainer.py#L3858)). It is trained on COCO first and then on a mix of BDD100K street-driving scenes and COCO, and it runs in real time on an NVIDIA Jetson Orin Nano ([README.md:3](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/README.md#L3), [README.md:9](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/README.md#L9)).

HCM is not my invention. It stands for Hierarchical Coherence Modeling, from Fengyang Xiao, Pan Zhang, Chunming He, Runze Hu and Yutao Liu, "Concealed Object Segmentation with Hierarchical Coherence Modeling", CICAI 2023, arXiv:2401.11767 (22 January 2024). The paper designed HCM for concealed-object segmentation: camouflaged objects, polyps and transparent objects. The code I describe here is an independent implementation of that design, adapted to street-scene segmentation on a Jetson. The architecture and the three modules in the middle of it are the paper's work. The adaptations are the subject of the section near the end.

Every claim about the code here is a claim about commit `261a630` of the public repository, where the paths and line numbers I cite can be checked.

## What it is for

The output is a mask at half the input resolution. A 352 by 352 image becomes two logit maps at 176 by 176, one for object and one for background, and the mask is the per-pixel comparison between them. The model is not multi-class and it is not an instance segmenter: it does not say "car" or "person", and it does not separate one object from the next. It answers a narrower question, which pixels belong to any object at all, and it answers it on a live video feed, on a device that fits in a hand.

It answers that question on a live video feed: the whole model runs on a Jetson Orin Nano at 14 frames per second.

## What it is good at

The numbers are on this repository's own validation metric: the overlap between predicted and true object pixels, averaged per image. It is not the mean intersection over union used on public leaderboards, and it is not comparable with the paper's numbers, because the task, the datasets and the metric all differ.

On that metric the live model scores 83.12 on BDD100K, 79.46 on COCO and 81.29 on the combined set. The previous live model, an earlier version of this design (ResNet-50, the original wide neck and v1 decoder, trained on COCO only), scored 72.29 and 78.28 on the same sets, so the current model is 10.83 points higher on BDD100K and 1.18 on COCO. The gain mixes the new backbone, the decoder changes and the BDD100K training data; no experiment separates them. On the Orin Nano it runs a TensorRT fp32 engine at 70.84 milliseconds per frame, which is 14.12 frames per second, against 181.67 milliseconds and 5.50 frames per second for the previous model. The TensorRT engine was checked against PyTorch on seeded inputs before use: the largest logit difference was 0.000403, against an allowed 0.25. A ConvNeXt V2-Base variant scores higher, 84.59 and 80.95, but it does not fit the Jetson and is used for offline renders only.

## The backbone, which is not ours either

The live model uses ConvNeXt V2-Tiny, loaded through timm with the tag `convnextv2_tiny.fcmae_ft_in22k_in1k` ([backbones.py:38](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/ecs_hcm_model/core/backbones.py#L38)). It was pretrained with FCMAE, ConvNeXt V2's masked-autoencoder pretraining, on ImageNet-22k, then fine-tuned on ImageNet-1k. ConvNeXt V2 is from Woo, Debnath, Hu, Chen, Liu, Kweon and Xie, "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders", CVPR 2023.

The backbone turns the image into four feature maps, at 1/4, 1/8, 1/16 and 1/32 of the input size. For the Tiny model they have 96, 192, 384 and 768 channels. HCM keeps those four maps and throws away the backbone's own classifier.

The backbone is not frozen. Training starts from the ImageNet weights and keeps updating them along with the decoder. There is an optional warm-up that freezes the backbone for the first epochs and then unfreezes it stage by stage, but it is written with ResNet layer names and is not part of how the live model was trained.

Five backbones are supported behind one four-level interface: ResNet-50, EfficientViT-B2, ConvNeXt V2-Nano, ConvNeXt V2-Tiny and ConvNeXt V2-Base. ResNet-50 is still the code's default; the live model is the Tiny one.

## What the decoder adds

A backbone on its own is not a segmenter. It produces coarse feature maps at four resolutions and has no mask output. Everything between those maps and the mask is the decoder, and in this code it is made of seven parts: a neck, four ISC blocks, three CSC blocks, an ASPP module, four RRD blocks, a refinement step and a 1 by 1 head.

The paper contributes ISC, CSC, RRD and the way they are wired together. This implementation contributes the neck and the refinement step, and changes a handful of details inside the paper's blocks. The next sections take them in order.

![The full model, from the backbone's four feature maps through the neck, the ISC and CSC blocks and the ASPP seed, to the four RRD blocks, the refinement step and the segmentation head]({{ site.baseurl }}/assets/hcm-architecture/full-model.svg)

## The blocks, one at a time

### The neck

Each of the four backbone maps gets a 1 by 1 convolution and a GroupNorm that bring it to 128 channels, and then all four are resized to the 1/4 grid, 88 by 88 ([hcm_nn.py:856-861](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/ecs_hcm_model/core/hcm_nn.py#L856-L861), [hcm_nn.py:970-976](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/ecs_hcm_model/core/hcm_nn.py#L970-L976)). The outputs are four maps of shape 128 by 88 by 88, one per level, and every block after this reads from that common grid.

The neck is also where most of the width is cut. The earlier version of this code widened every level to 2048 channels; this one narrows to 128.

### The ISC blocks

There are four, one per level. Each takes a 128 by 88 by 88 map and returns 32 by 88 by 88. Inside, the input feeds three branches. All three branches start with a 1 by 1 convolution. The second and third then apply a 3 by 3 and a 5 by 5 convolution, and those two results are added. The sum feeds a second 3 by 3 and 5 by 5 pair, and those two results are multiplied rather than added, which lets one scale gate the other. A GroupNorm and a 3 by 3 convolution follow, and the result is added to the first branch, a single 1 by 1 convolution of the input that serves as the shortcut ([hcm_nn.py:473-503](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/ecs_hcm_model/core/hcm_nn.py#L473-L503)).

![One ISC block: three 1 by 1 branches, a summed 3 by 3 and 5 by 5 pair, a multiplied second pair, then group normalisation, a 3 by 3 convolution and a shortcut]({{ site.baseurl }}/assets/hcm-architecture/isc.svg)

In the paper this block is called intra-stage coherence. The structure above is the paper's. The GroupNorm and the extra 3 by 3 after the multiplication are this implementation's, and the fixed widths of 32 out and 64 inside are choices it had to make, because the paper does not give widths.

### The CSC blocks

There are three, and they merge levels from coarse to fine. A CSC block takes two 32-channel maps at 88 by 88. It normalises both with GroupNorm, upsamples the coarser one and adds them, calling the sum S. S is then average-pooled to 16 by 16, normalised, and passed through the attention module. The attention is multi-head self-attention with 8 heads of 4 channels each over the 256 positions of the 16 by 16 grid, with query, key, value and output as 1 by 1 convolutions. The result is upsampled back to 88 by 88 and passed through PONO, a per-pixel normalisation across channels with a learned scale and shift. The v2 decoder then adds S back to the output, so detail finer than the 16 by 16 attention grid still reaches the decoder ([hcm_nn.py:586-633](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/ecs_hcm_model/core/hcm_nn.py#L586-L633)).

![One CSC block: both inputs normalised, the coarse one upsampled and added to form S, pooled to 16 by 16, attended, upsampled, positionally normalised, and added back to S]({{ site.baseurl }}/assets/hcm-architecture/csc.svg)

This block is called cross-stage coherence in the paper. Two differences are worth stating plainly. The paper describes its joint attention as spatial plus channel attention; this implementation uses spatial self-attention only, with no separate channel step. And the paper's equation has no residual around the attention; the v2 decoder here adds the pre-attention sum back. PONO is the same idea as the paper's position normalization, credited to Li, Wu, Weinberger and Belongie, "Positional Normalization", NeurIPS 2019.

### The ASPP seed

ASPP is the standard DeepLab multi-rate context module, from Chen, Papandreou, Schroff and Adam, "Rethinking Atrous Convolution for Semantic Image Segmentation" (2017), and it is not this project's invention. It reads the deepest level, already resized to 128 by 88 by 88, and runs five branches in parallel: a 1 by 1 convolution, three 3 by 3 convolutions dilated by 6, 12 and 18, and a branch that pools the map down to a 4 by 4 grid and upsamples it back. Each has GroupNorm and ReLU. The five are joined and reduced by a 1 by 1 convolution with stride 2, followed by GroupNorm and ReLU, giving a 32 by 44 by 44 map ([hcm_nn.py:716-751](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/ecs_hcm_model/core/hcm_nn.py#L716-L751)).

That map is the first prediction, at stride 8. Each RRD block below it upsamples the prediction it is given back to 88 by 88 before using it, so the priors do not have to share a resolution.

### The RRD blocks

There are four, and they decode from coarse to fine. Each takes a 32 by 88 by 88 feature map and a prior prediction of any size, and returns 32 by 88 by 88. The prior is upsampled and passed through a sigmoid to give a soft mask, called sigma. The features are then viewed twice: once multiplied by sigma, which keeps what the current prediction calls object, and once multiplied by one minus sigma, which keeps what it calls background. Each view gets a GroupNorm. The v2 decoder joins them on the channel axis, giving 64 channels, and fuses them with a 1 by 1 convolution back to 32, then applies a residual channel attention block and a 3 by 3 convolution ([hcm_nn.py:671-713](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/ecs_hcm_model/core/hcm_nn.py#L671-L713)).

![One RRD block: the prior is upsampled and passed through a sigmoid, the features are weighted by sigma and by one minus sigma, group normalised, fused by a 1 by 1 convolution, then passed through a residual channel attention block and a 3 by 3 convolution]({{ site.baseurl }}/assets/hcm-architecture/rrd.svg)

The paper calls this the reversible re-calibration decoder, and the two-view idea is the paper's. The GroupNorm on each branch is this implementation's. So is the fusion: the v1 decoder joined the two views side by side along the image width and halved the width with a strided convolution, while v2 joins on channels and fuses with a 1 by 1 convolution. The residual channel attention block is a 3 by 3 convolution, a ReLU, a second 3 by 3 convolution and squeeze-and-excitation weights, with the input added back; squeeze-and-excitation is from Hu, Shen and Sun, "Squeeze-and-Excitation Networks", CVPR 2018.

### Refinement and the head

The decoder's last output is at stride 4. The refinement step doubles it to 176 by 176. ConvNeXt V2-Tiny has no stride-2 feature map, so the earliest backbone map, c2 at 96 by 88 by 88, is upsampled to 176 by 176 and projected from 96 to 32 channels with a 1 by 1 convolution. It is joined with the upsampled decoder output, giving 64 channels, and two rounds of convolution, GroupNorm and ReLU bring it back to 32 ([hcm_nn.py:1063-1098](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/ecs_hcm_model/core/hcm_nn.py#L1063-L1098)). The head is a single 1 by 1 convolution to the two class logits.

The refinement step is this implementation's addition. Its point is that the finer mask is built from the backbone's own fine detail, not by stretching the coarse mask alone.

## The data flow, exactly

For the live configuration, a 352 by 352 input, ConvNeXt V2-Tiny, a neck of 128 and the v2 decoder ([hcm_nn.py:936-1050](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/ecs_hcm_model/core/hcm_nn.py#L936-L1050)):

1. The backbone produces c2 at 96 by 88 by 88, c3 at 192 by 44 by 44, c4 at 384 by 22 by 22 and c5 at 768 by 11 by 11. c2 is kept aside for the refinement step.
2. The neck turns each into 128 channels and resizes all four to 88 by 88, giving r2, r3, r4 and r5.
3. Four ISC blocks turn r2 through r5 into four 32-channel maps, ISC0 through ISC3.
4. The CSC loop runs from coarse to fine. CSC2 takes ISC2 and ISC3. CSC1 takes ISC1 and CSC2. CSC0 takes ISC0 and CSC1.
5. ASPP reads r5 and produces a 32 by 44 by 44 map.
6. The RRD loop runs from coarse to fine as well. RRD3 takes ISC3 and the ASPP map and produces p3. RRD2 takes CSC2 and p3 and produces p2. RRD1 takes CSC1 and p2 and produces p1. RRD0 takes CSC0 and p1 and produces p0, the decoder's 32 by 88 by 88 output.
7. The refinement step joins an upsampled p0 with the projected c2 and produces 32 by 176 by 176.
8. The 1 by 1 head produces two logit maps at 176 by 176.

Two details of the wiring are easy to get wrong, so they are worth naming. The CSC blocks see ISC outputs, not the neck outputs, and each CSC takes the previous CSC's result rather than the next ISC. The RRD blocks see CSC outputs and the previous RRD's prediction, while the first of them sees an ISC output and the ASPP map instead.

## What this implementation does differently

The paper is a method for concealed-object segmentation, evaluated on camouflaged-object, polyp and transparent-object datasets. This implementation is trained for object against background on street scenes and everyday photographs, first on COCO and then on BDD100K mixed with COCO, with two output classes. The task, the data and the metric all differ, and no number here should be read against the paper's.

The backbone is different. The paper uses an ImageNet-pretrained ResNet-50; the live model here uses ConvNeXt V2-Tiny, with ResNet-50 still the default in the code. The input size is the same, 352 by 352.

The neck and the refinement step are additions. The paper describes no neck and no refinement stage, so this implementation had to build both. The neck narrows where the earlier version of this code widened, and the refinement step recovers a half-resolution mask from the backbone's finest features.

Inside the paper's blocks, several details are this implementation's choices. ISC gets GroupNorm and an extra 3 by 3 convolution after the gating multiply. CSC's attention is spatial only, where the paper describes spatial plus channel attention, and v2 adds the pre-attention sum back. RRD gets a GroupNorm on each view and fuses on channels rather than along the width. Where the paper is silent, the channel widths, the attention pooling size, the upsampling method and the RRD joining axis are choices this implementation had to make, not changes from a stated design.

The training loss differs too. The paper uses a weighted binary cross-entropy and a weighted intersection over union at five scales, with deep supervision, plus an extra term. This implementation uses weighted cross-entropy and Dice on the final output only; the decoder's intermediate predictions are not supervised.

Two engineering decisions are visible in the code. The ISC to CSC chain runs in 32-bit floats on purpose, because its activations are large enough to overflow 16-bit, and the attention scores are computed in fp32 as well. And deployment is a TensorRT engine for the Orin Nano, verified against PyTorch before use.

## The numbers

| Claim | Number |
|---|---|
| Live model, BDD100K validation IoU, this metric | 83.12 |
| Live model, COCO validation IoU | 79.46 |
| Live model, combined validation IoU | 81.29 |
| Previous model (ResNet-50, earlier design, COCO only), same sets | 72.29 and 78.28 |
| Difference, current vs previous model | +10.83 BDD100K, +1.18 COCO |
| Orin Nano speed, live model | 70.84 ms per frame, 14.12 fps |
| Orin Nano speed, previous model | 181.67 ms per frame, 5.50 fps |
| TensorRT against PyTorch, seeded inputs | 0.000403 maximum logit difference, allowed 0.25 |
| ConvNeXt V2-Base variant, offline only | 84.59 and 80.95 |

The accuracy figures come from a run on 22 October 2026, best epoch 10. The speed figures are from a benchmark loop with the TensorRT fp32 engine on 22 October 2026. The previous model's speed is from 18 October 2026.

## What is not known

Several things are not measured. The parameter count and the floating-point operations per image are not measured. The accuracy on the Jetson itself was not measured; only the engine against PyTorch was checked. There is no comparison against other segmentation models on the same data. Latency was measured only on the Orin Nano. And there is no per-block ablation: no experiment shows how much each of ISC, CSC, RRD or PONO contributes to the final score, so none of them is claimed to be the reason for it.

The Base variant does not run on the Jetson. The model does not do multi-class or instance segmentation. The backbone does not stay frozen. And the numbers above come from this codebase, on this metric, not leaderboard numbers and not the paper's.

## References

- Fengyang Xiao, Pan Zhang, Chunming He, Runze Hu, Yutao Liu. "Concealed Object Segmentation with Hierarchical Coherence Modeling." CICAI 2023. arXiv:2401.11767, 22 January 2024.
- Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, Saining Xie. "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders." CVPR 2023. arXiv:2301.00808.
- Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam. "Rethinking Atrous Convolution for Semantic Image Segmentation." arXiv:1706.05587, 2017.
- Jie Hu, Li Shen, Gang Sun. "Squeeze-and-Excitation Networks." CVPR 2018. arXiv:1709.01507.
- Boyi Li, Felix Wu, Kilian Q. Weinberger, Serge Belongie. "Positional Normalization." NeurIPS 2019. arXiv:1907.04312.
