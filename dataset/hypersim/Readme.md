### Hypersim Dataset Structure

```
├── ai_001_001
├── ai_055_009
├── ...
├── ai_055_010
├── metadata_camera_parameters.csv
└── metadata_images_split_scene_v1.csv
```

Unzip 'hypersim_scores.zip', and you will get 
```
hypersim_proc
├── ai_001_001
│   ├── ai_001_001_cam_00_mask_score.csv
│   └── cam_00_label.csv
├── ...
```

Merge these files according to the folder structure.
```
ai_001_001
├── _detail
├── ai_001_001_cam_00_mask_score.csv
├── cam_00_label.csv
└── images
```