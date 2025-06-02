### Supported Methods
- [x] CUT3R 
- [x] Spann3R 
- [x] Dust3R 
- [x] AETHER 
- [x] UniGeo 
- [x] VideoDepthAnything 
- [x] ChronoDepth 
- [x] DepthCrafter 
- [x] DepthAnyVideo 

In order to integrate into our interface, we need to implement the prepare_input / prepare_output function.

The input of `prepare_input` is listed in [dataset part](../dataset/Readme.md).

The output of `prepare_output` should be a dict, containing 
- pred_world_pts : [Nf,H,W,3]
- pred_depths : [Nf,H,W]
- pred_normals : [Nf,H,W,3]
- pred_poses : [Nf,4,4]