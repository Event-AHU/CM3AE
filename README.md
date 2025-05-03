# CM3AE 

**CM3AE: A Unified RGB Frame and Event-Voxel/-Frame Pre-training Framework**, 
Wentao Wu, Xiao Wang, Chenglong Li, Bo Jiang, Jin Tang, Bin Luo, Qi Liu [[arXiv](https://arxiv.org/abs/2504.12576)] 

## Abstract 
Event cameras have attracted increasing attention in recent years due to their advantages in high dynamic range, high temporal resolution, low power consumption, and low latency. Some researchers have begun exploring pre-training directly on event data. Nevertheless, these efforts often fail to establish strong connections with RGB frames, limiting their applicability in multi-modal fusion scenarios. To address these issues, we propose a novel CM3AE pre-training framework for the RGB-Event perception. This framework accepts multi-modalities/views of data as input, including RGB images, event images, and event voxels, providing robust support for both event-based and RGB-event fusion based downstream tasks. Specifically, we design a multi-modal fusion reconstruction module that reconstructs the original image from fused multi-modal features, explicitly enhancing the model’s ability to aggregate cross-modal complementary information. Additionally, we employ a multi-modal contrastive learning strategy to align cross-modal feature representations in a shared latent space, which effectively enhances the model’s capability for multi-modal understanding and capturing global dependencies. We construct a large-scale dataset containing 2,535,759 RGB-Event data pairs for the pre-training. Extensive experiments on five downstream tasks fully demonstrated the effectiveness of CM3AE.

<p align="center">
  <img width="100%" src="https://github.com/Event-AHU/CM3AE/blob/main/figures/firstIMG.jpg" alt="firstIMG"/>
</p> 

## Environment Setting 

Configure the environment according to the content of the requirements.txt file.

## Pre-trained Model Download

Baidu Netdisk Link ：[download](https://pan.baidu.com/s/1rR9H6fmrbMBL8jf5LwC9Hw?pwd=ytje)

Extracted code ：ytje

## Training

```bibtex
#If you pre-training CM3AE using a single GPU, please run.
CUDA_VISIBLE_DEVICES=0 python main.py
#If you pre-training CM3AE using multiple GPUs, please run.
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py
```


## Experimental Results 

We used full fine-tuning to test the pre-trained model on five downstream tasks. The results are shown in the table below. 

<p align="center">
  <img width="80%" src="https://github.com/Event-AHU/CM3AE/blob/main/figures/result.png" alt="result"/>
</p> 


## Visual Results 

<p align="center">
  <img width="80%" src="https://github.com/Event-AHU/CM3AE/blob/main/figures/reconst_vis.jpg" alt="reconst_vis"/>
</p> 

<p align="center">
  <img width="80%" src="https://github.com/Event-AHU/CM3AE/blob/main/figures/attentionmaps.jpg" alt="attentionmaps"/>
</p> 

<p align="center">
  <img width="80%" src="https://github.com/Event-AHU/CM3AE/blob/main/figures/downstream_tasks_visualization.jpg" alt="downstream_tasks_visualization"/>
</p> 

## Acknowledgement 
[[MAE](https://github.com/facebookresearch/mae)] 

## Citation 

If you find this work helps your research, please cite the following paper and give us a star. 
```bibtex
@misc{wu2025cm3ae,
      title={CM3AE: A Unified RGB Frame and Event-Voxel/-Frame Pre-training Framework}, 
      author={Wu, Wentao and Wang, Xiao and Li, Chenglong and Jiang, Bo and Tang, Jin and Luo, Bin and Liu, Qi},
      year={2025},
      eprint={2504.12576},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


if you have any problems with this work, please leave an issue. 





