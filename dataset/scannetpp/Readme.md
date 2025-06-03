1. Download ScannetPP

2. Render with the following script for depth and normal

- Install pyrender if you don't have

    ```
    pip install -i https://mirrors.aliyun.com/pypi/simple/ pyrender

    ```

- Render depth, normal and save camera pose

    ```
    python preprocess_scannetpp_imu.py --target_resolution 512 --pyopengl-platform egl --precomputed_pairs ./splits/  --output_dir /mnt/pfs/data/
    ```


3. You should obtain data like this
    ```
    scannetpp_processed_iphone
    └── 1ada7a0617
        ├── depth
        ├── depth_vis
        ├── images
        └── normal
    └── ...
    ```