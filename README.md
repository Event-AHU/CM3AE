# CM3AE 

**CM3AE: A Unified RGB Frame and Event-Voxel/-Frame Pre-training Framework**, 
Wentao Wu, Xiao Wang, Chenglong Li, Bo Jiang, Jin Tang, Bin Luo, Qi Liu [[arXiv](https://arxiv.org/abs/2504.12576)] 

## Abstract 
Event cameras have attracted increasing attention in recent years due to their advantages in high dynamic range, high temporal resolution, low power consumption, and low latency. Some researchers have begun exploring pre-training directly on event data. Nevertheless, these efforts often fail to establish strong connections with RGB frames, limiting their applicability in multi-modal fusion scenarios. To address these issues, we propose a novel CM3AE pre-training framework for the RGB-Event perception. This framework accepts multi-modalities/views of data as input, including RGB images, event images, and event voxels, providing robust support for both event-based and RGB-event fusion based downstream tasks. Specifically, we design a multi-modal fusion reconstruction module that reconstructs the original image from fused multi-modal features, explicitly enhancing the model’s ability to aggregate cross-modal complementary information. Additionally, we employ a multi-modal contrastive learning strategy to align cross-modal feature representations in a shared latent space, which effectively enhances the model’s capability for multi-modal understanding and capturing global dependencies. We construct a large-scale dataset containing 2,535,759 RGB-Event data pairs for the pre-training. Extensive experiments on five downstream tasks fully demonstrated the effectiveness of CM3AE.




