# SEE-VCN

This is the official codebase for 

Viewer-Centred Surface Completion for Unsupervised Domain Adaptation in 3D Object Detection

### Abstract

Every autonomous driving dataset has a different configuration of sensors, originating from distinct geographic regions and covering various scenarios. As a result, 3D detectors tend to overfit the datasets they are trained on. This causes a drastic decrease in accuracy when the detectors are trained on one dataset and tested on another. We observe that lidar scan pattern differences form a large component of this reduction in performance. We address this in our approach, SEE-VCN, by designing a novel viewer-centred surface completion network (VCN) to complete the surfaces of objects of interest within an unsupervised domain adaptation framework, SEE [1]. With SEE-VCN, we obtain a unified representation of objects across datasets, allowing the network to focus on learning geometry, rather than overfitting on scan patterns. By adopting a domain- invariant representation, SEE-VCN can be classed as a multi- target domain adaptation approach where no annotations or re-training is required to obtain 3D detections for new scan patterns. Through extensive experiments, we show that our approach outperforms previous domain adaptation methods in multiple domain adaptation settings.

```
# Install vcn
cd see/surface_completion && pip install -e . --user

# Install OpenPCDet
cd detector3d && python setup.py develop
```
