---
layout: post
title: "Specialise without forgetting"
date: 2026-11-25
---

A model trained on everyday photographs is not at its best on a dashcam. Fine-tuning it on street scenes fixes that and costs something: the accuracy it already had. This post is about how HCM is trained in two stages so that a street-scene result can be judged on what it gained and on what it kept, and about the rule that decides whether a result is allowed to replace the model in use.

## What HCM is

HCM is a segmentation model for street scenes. For every pixel of a camera frame it decides whether the pixel is part of an object or part of the background, and it was built to run in real time on a small edge GPU ([README.md:3](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/README.md#L3)). Trained in the two stages described here, a version of it scores 82.56 on BDD100K by this repository's own validation metric, measured at 352 by 352, against 72.29 for the model in use, and that result is what this post explains. What the model is and how it is built are in the post of 11 November, "How HCM Turns a Camera Frame into a Mask" ([oluseye-akomolede.github.io/blog/2026/11/11/how-hcm-works](https://oluseye-akomolede.github.io/blog/2026/11/11/how-hcm-works/)); this one is about training it.

## Why fine-tuning on street scenes by itself does not work

A model trained on COCO does well on everyday objects and less well on street scenes, so fine-tuning it on street scenes is the obvious next step. Done alone, it overwrites what the COCO training put there: an earlier refine of this code on a different dataset took COCO validation IoU from 69.8 down to 44.5, and the comment recording that is still in the trainer as the reason the mix exists ([trainer.py:152-156](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/ecs_hcm_model/core/trainer.py#L152-L156)). The phenomenon is catastrophic forgetting, and the usual remedy is rehearsal: keep part of the old data in the mix, so the weights that served it go on being exercised.

## Stage 1: general training on COCO

Stage 1 trains on COCO train from an ImageNet-pretrained backbone, and validates on COCO val every epoch. The epoch that counts as best is the one with the highest `val_iou`, the segmentation overlap, not the one with the lowest validation loss ([trainer.py:145-150](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/ecs_hcm_model/core/trainer.py#L145-L150)). Early stopping watches the same series in the same direction, so it ends a run whose overlap has stopped improving ([trainer.py:564-575](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/ecs_hcm_model/core/trainer.py#L564-L575)). What comes out of stage 1 is one checkpoint, chosen on COCO accuracy alone.

## Stage 2: street scenes with COCO mixed in

Stage 2 starts from that checkpoint rather than from ImageNet. The `--init-from` flag takes its path ([main.py:134](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/main.py#L134), [services/trainer.py:21](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/ecs_hcm_model/core/services/trainer.py#L21)), and a warm start that could not mean anything for the run it is starting is rejected rather than trained from ([trainer.py:174](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/ecs_hcm_model/core/trainer.py#L174)).

Its training data is BDD100K train with COCO train batches interleaved back in. The batches are interleaved, not concatenated, so the two datasets alternate through the epoch instead of each taking a long turn at it ([trainer.py:482](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/ecs_hcm_model/core/trainer.py#L482), [trainer.py:2288-2296](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/ecs_hcm_model/core/trainer.py#L2288-L2296)).

How much COCO goes in is a share, `--bdd-coco-mix-share`, defaulting to 0.5 ([main.py:121-123](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/main.py#L121-L123)). The share is a share of each epoch's samples, while the trainer's own knob counts extra COCO batches per BDD batch, so the code converts a share s into s/(1-s) ([trainer.py:333-344](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/ecs_hcm_model/core/trainer.py#L333-L344)). A share of 0.5 is one COCO batch for each BDD batch; a share of 0.75 is three.

Every epoch is scored on both validation sets and each figure is logged under its own suffix, `val_iou_bdd` and `val_iou_coco`, with the unsuffixed `val_iou` kept as the mean of the two ([spec.md:41-65](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/openspec/specs/bdd100k-refine/spec.md#L41-L65)).

The best epoch, and the point at which early stopping ends the run, are decided on BDD val only. The spec gives the reason in one sentence: "a COCO gain MUST NOT be able to pay for a BDD loss" ([spec.md:71](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/openspec/specs/bdd100k-refine/spec.md#L71)). Read on the mean, a model that had become better at COCO and worse at street scenes could win the comparison, which is the opposite of what the fine-tune is for. The spec tests exactly that case: an epoch whose mean falls because COCO fell further than BDD rose still counts as the best so far ([spec.md:76-79](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/openspec/specs/bdd100k-refine/spec.md#L76-L79)). Early stopping reads the same key, so a falling COCO figure cannot end a run whose BDD figure is still climbing ([trainer.py:345-360](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/ecs_hcm_model/core/trainer.py#L345-L360)).

The checkpoint records what produced it: the mix ratio, both per-dataset validation IoUs, and a dataset label naming the two sources, `BDD100K+MSCOCO` ([trainer.py:1627-1628](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/ecs_hcm_model/core/trainer.py#L1627-L1628), [trainer.py:374](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/ecs_hcm_model/core/trainer.py#L374), [spec.md:81-84](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/openspec/specs/bdd100k-refine/spec.md#L81-L84)).

![The training pipeline: ImageNet weights start stage 1 on COCO train, which keeps its best epoch on COCO validation IoU and hands its checkpoint to stage 2; stage 2 fine-tunes on BDD100K with COCO batches mixed in at a share of 0.75, validates on both sets every epoch and keeps its best epoch on BDD validation only; the gate then asks whether COCO validation is no more than one point below 78.28, the model in use, and the candidate ships only if it is]({{ site.baseurl }}/assets/hcm-two-stage/two-stage.svg)

## The street data

BDD100K labels a scene with many classes. This model treats nine of them as object: person, rider, car, truck, bus, train, motorcycle, bicycle and traffic light. Every other class, and the pixels the annotator left unlabelled, become background ([bdd.py:55-65](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/ecs_hcm_model/core/bdd.py#L55-L65), [spec.md:11-18](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/openspec/specs/bdd100k-refine/spec.md#L11-L18)).

The conversion happens once, outside training, in a prepare step that writes a binary tree of images and masks, refuses an image with no matching label, and drops any pair whose image and label are not the same frame, recording it in the manifest instead of writing it ([spec.md:91-106](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/openspec/specs/bdd100k-refine/spec.md#L91-L106), [spec.md:123-129](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/openspec/specs/bdd100k-refine/spec.md#L123-L129)). Training reads that tree.

The numbers below come from a split of 6,996 BDD100K training images and 998 validation images, with 4,952 COCO validation images. Those counts live in the prepared data's manifest; there is no public list to point at, so they are stated here as the split these results were measured on.

## How the IoU is measured

Every IoU in this post is the mean over images of each image's overlap for the binary foreground mask. For one image it is intersection plus a small epsilon, over union plus the same epsilon, and the per-image values are then averaged ([trainer.py:1304-1335](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/ecs_hcm_model/core/trainer.py#L1304-L1335)). The average is over images and not over pixels, so a small object counts as much as a large one in the score. The predicted mask is the foreground probability, a softmax over the background and foreground logits, thresholded at 0.5 ([trainer.py:1338-1355](https://github.com/oluseye-akomolede/hcm-segmentation/blob/261a630/ecs_hcm_model/core/trainer.py#L1338-L1355)).

Every model below is scored at 352 by 352, on the ground-truth masks at that size, so the figures are comparable with each other. The input is 352 by 352 for every run. The ConvNeXt V2-Tiny model builds its mask at a stride of 2, so its 176 by 176 output is upsampled to 352 before it is scored.

## The gate

A stage 2 run produces a candidate, not a replacement. The candidate is scored on COCO val and compared with the model in use, which scores 78.28 there. A candidate more than one point below that does not ship. One that holds the line is then compared with other candidates on BDD val, and that comparison is what picks the next model. The candidate in the table scores 79.39 on COCO val, above the model in use rather than below it, so it passes the rule.

It is a release rule I apply by hand when results come in, and nothing in the trainer enforces it: there is no flag, no assertion and no automatic check. What the code supplies is the evidence, because COCO val and BDD val are computed every epoch and by the same function as the numbers below, so applying the rule means reading two figures that are already there.

## The results

All figures are IoU times 100, on the split above.

| Model | Backbone, mask stride | Stage 2 COCO share | BDD val | COCO val | Change vs model in use (BDD / COCO) | Gate |
|---|---|---|---|---|---|---|
| Model in use, COCO only | ResNet-50, 4 | none | 72.29 | 78.28 | baseline | not applicable |
| Street refine, one COCO batch per BDD batch | ResNet-50, 4 | 0.5 | 81.82 | 76.13 | +9.53 / -2.15 | fails |
| Street refine, three COCO batches per BDD batch | ResNet-50, 4 | 0.75 | 81.16 | 76.90 | +8.87 / -1.38 | fails |
| Two stages, ConvNeXt V2-Tiny | ConvNeXt V2-Tiny, 2 | 0.75 | 82.56 | 79.39 | +10.27 / +1.11 | passes |

![Scatter plot of BDD validation IoU against COCO validation IoU for the four models, all scored at 352 by 352: the model in use at 72.29 and 78.28, the 1:1 mix at 81.82 and 76.13, the 3:1 mix at 81.16 and 76.90, and the two-stage ConvNeXt V2-Tiny model at 82.56 and 79.39; a dashed line marks the gate at COCO 77.28, one point below the model in use, and only the two-stage model sits above it]({{ site.baseurl }}/assets/hcm-two-stage/results.svg)

Read the three ResNet-50 rows together, where the backbone is the same, and the share does one thing. More COCO in the mix costs a little BDD accuracy and buys COCO back: going from one COCO batch per BDD batch to three raised COCO val from 76.13 to 76.90 and lowered BDD val from 81.82 to 81.16. Neither setting held the gate. Choosing that run's epoch by COCO rather than by BDD does not rescue it either; its best COCO epoch reached 77.21, 1.07 below the model in use and still under the gate line of 77.28.

Both mixes lost COCO against the untouched model, one by 2.15 points and one by 1.38. Rehearsal reduced the forgetting and did not remove it, and at these settings the dial could not be turned far enough to pass.

The fourth row is a different model rather than a further turn of the dial. It changed the backbone from ResNet-50 to ConvNeXt V2-Tiny, whose mask stride is 2 rather than 4, and it trained in two stages with the same three to one mix. Its COCO val, 79.39, is 1.11 above the model in use. Against its own stage 1 it gained 1.01, from 78.45 to 79.46; those two are scored on the model's own 176 by 176 output, the one comparison in this post not made at 352. Its BDD val, 82.56, is 10.27 above the model in use. The gain belongs to the whole recipe, and this table cannot divide it: the mix, the backbone and the finer mask all changed at once, so nothing here shows that the COCO gain comes from the mix alone.

What the table does show is what rehearsal did and did not do. On one backbone it cut the COCO loss from 2.15 to 1.38, a little more than a third, and cost 0.66 of a point of BDD val. It did not, by itself, keep COCO inside the gate. The recipe that passed combined the mix with a newer backbone and finer masks, and its COCO figure is a gain rather than a smaller loss.

## What these numbers are not

They are not BDD100K benchmark results. BDD100K's segmentation task is 19-class semantic or instance segmentation, while the target here is binary, object against background, so none of these figures is comparable with published numbers on that task. They come from runs of this code on the split above; the repository holds the code that produces them, not the results.

There are no per-class BDD scores, because the prepared data is binary and the model has no class output. There is no on-device figure and no speed figure for these models; the previous post has the speed measurement on the target device, and nothing here repeats or extends it. And there is no ablation: this is four models, and the one that passed changed three things at once.

## Prior work

The design is the standard rehearsal answer to catastrophic forgetting. The code cites none of these papers and the mix does not come from a particular one of them; they are the standard references for the idea.

McCloskey and Cohen described catastrophic interference in 1989, Ratcliff analysed it in models of recognition memory in 1990, and French reviewed the literature in 1999. Rehearsal, feeding old data back in alongside the new, is Robins's term from 1995, and Chaudhry and coauthors showed in 2019 that even a small replay of old examples works well. Mixing in-domain data with the original data during fine-tuning is the mixed fine-tuning that Chu, Dabre and Kurohashi compared against other domain-adaptation methods in 2017.

The alternatives that keep old accuracy by protecting weights rather than by replaying data are not used here: elastic weight consolidation (Kirkpatrick and coauthors, 2017) and learning without forgetting (Li and Hoiem, 2016).

The data and the backbone are other people's work: COCO (Lin and coauthors, 2014), BDD100K (Yu and coauthors, 2020), ConvNeXt V2 (Woo and coauthors, 2023), and HCM itself (Xiao and coauthors, 2023).

## References

- McCloskey, M. and Cohen, N. J. "Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem." Psychology of Learning and Motivation 24, 1989, 109-165. https://doi.org/10.1016/S0079-7421(08)60536-8
- Ratcliff, R. "Connectionist models of recognition memory: Constraints imposed by learning and forgetting functions." Psychological Review 97(2), 1990, 285-308. https://doi.org/10.1037/0033-295X.97.2.285
- French, R. M. "Catastrophic forgetting in connectionist networks." Trends in Cognitive Sciences 3(4), 1999, 128-135. https://doi.org/10.1016/S1364-6613(99)01294-2
- Robins, A. "Catastrophic Forgetting, Rehearsal and Pseudorehearsal." Connection Science 7(2), 1995, 123-146. https://doi.org/10.1080/09540099550039318
- Chaudhry, A., Rohrbach, M., Elhoseiny, M., Ajanthan, T., Dokania, P. K., Torr, P. H. S. and Ranzato, M. "On Tiny Episodic Memories in Continual Learning." arXiv:1902.10486, 2019. https://arxiv.org/abs/1902.10486
- Chu, C., Dabre, R. and Kurohashi, S. "An Empirical Comparison of Domain Adaptation Methods for Neural Machine Translation." ACL 2017, 385-391. https://doi.org/10.18653/v1/P17-2061
- Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D. and Hadsell, R. "Overcoming catastrophic forgetting in neural networks." PNAS 114(13), 2017, 3521-3526. https://doi.org/10.1073/pnas.1611835114
- Li, Z. and Hoiem, D. "Learning without Forgetting." ECCV 2016; IEEE TPAMI 40(12), 2018, 2935-2947. https://doi.org/10.1109/TPAMI.2017.2773081
- Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollar, P. and Zitnick, C. L. "Microsoft COCO: Common Objects in Context." ECCV 2014, 740-755. https://arxiv.org/abs/1405.0312
- Yu, F., Chen, H., Wang, X., Xian, W., Chen, Y., Liu, F., Madhavan, V. and Darrell, T. "BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning." CVPR 2020. https://arxiv.org/abs/1805.04687
- Woo, S., Debnath, S., Hu, R., Chen, X., Liu, Z., Kweon, I. S. and Xie, S. "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders." CVPR 2023. https://arxiv.org/abs/2301.00808
- Xiao, F., Zhang, P., He, C., Hu, R. and Liu, Y. "Concealed Object Segmentation with Hierarchical Coherence Modeling." CICAI 2023. arXiv:2401.11767. https://arxiv.org/abs/2401.11767
