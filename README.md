### ClothingMigration-NCNN
##### 工作内容
将CVPR2021的服装迁移模型[CT-Net](https://github.com/yf1019/CT-Net)搬运至[NCNN](https://github.com/Tencent/ncnn)框架上
##### CT-Net依赖
1. 依赖[OpenPose](https://github.com/Hzzone/pytorch-openpose)的人体姿势——pytorch实现
2. 依赖[LIP_JPPNet](https://github.com/Engineering-Course/LIP_JPPNet)的的人体解析分割图——Tensorflow实现
3. 依赖[Densepose](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose)的人体IUV图——[Detectron2](https://github.com/facebookresearch/detectron2)实现
##### 待完成工作
- [x] OpenPose的ncnn&c++实现
- [x] LIP_JPPNet的ncnn&c++实现
- [x] Densepose的ncnn&c++实现
- [x] CT-Net的ncnn&c++实现
- [x] 组合四个模型的ncnn&c++实现
- [x] QT GUI实现