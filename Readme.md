# UniGeo: Taming Video Diffusion for Unified Consistent Geometry Estimation

[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2505.24521)
[![Website](assets/badge-website.svg)](https://sunyangtian.github.io/UniGeo-web/)

<p align="center">
  <img src="assets/video_frame.png" alt="Result Overview" width="100%"/>
</p>


In this repo, we provide a unified framework for geometry estimation and evaluation.

Our framework provides a convenient interface for **various dataset** and **various methods**, which supports a fair comparison by aligning the output and evaluation scripts.
<p align="center">
  <img src="assets/framework.jpg" alt="Framework Overview" width="50%"/>
</p>

### Evaluation (in progressing ... )
The dataset, model and evaluation metric configuration can be set in the yaml file in [configs](./configs/). E.g.,

- Dataset Config:
  ```
  dataset: "ScannetPPDataset"
  root: /path/to/dataset
  h: 384
  w: 512
  clip_length: 25
  clip_overlap: 5
  split: "test"
  ```
- Model Config:
  ```
  model_name: "DepthCrafter"
  model_params:
    model_dir: /path/to/model
    unet_path: /path/to/DepthCrafter_unet
    pre_train_path: /path/to/stable_video_diffusion
  ```
- Metric Config
  ```
  eval_depth:
    metric_names: 
      - 'Abs Rel'
      - 'delta < 1.25'
      - 'delta < 1.25^2'
      - 'delta < 1.25^3'
    depth_alignment: "lstsq"

  eval_normal:
    metric_names: 
      - 'normal mean'
      - 'normal median'
      - 'angle < 7.5'
      - 'angle < 11.25'
  ```

Finally, the evaluation process can be performed by
  ```
    python eval.py
  ```

### Supported Datasets
Please refer to [dataset](./dataset/Readme.md) for more details.


### Supported Methods
Please refer to [model](./model/Readme.md) for more details.

### Acknowledgements
This code borrows heavily from [Spann3R](https://github.com/HengyiWang/spann3r), [Monst3R](https://github.com/Junyi42/monst3r), [CUT3R](https://github.com/CUT3R/CUT3R) and [robustmvd](https://github.com/lmb-freiburg/robustmvd). Thanks for these awesome works.


---
If you find this work helpful, please consider citing
```
@article{sun2025unigeo,
  title={UniGeo: Taming Video Diffusion for Unified Consistent Geometry Estimation},
  author={Sun, Yang-Tian and Yu, Xin and Huang, Zehuan and Huang, Yi-Hua and Guo, Yuan-Chen and Yang, Ziyi and Cao, Yan-Pei and Qi, Xiaojuan},
  journal={arXiv preprint arXiv:2505.24521},
  year={2025}
}
```
