# LiteAdapt-DETR：轻量化鲁棒性交通标志检测模型
🚗 LiteAdapt-DETR 是基于 RT-DETR 框架改进的实时目标检测模型，专为解决车载边缘设备在复杂交通环境中，对极致轻量化和高鲁棒精度的严苛要求而设计。  
✨ 核心创新点本模型通过结构级优化和特征增强，实现了检测精度、效率和资源消耗之间的最佳平衡。  
* HSPNet异尺度部分卷积网络替换原有 ResNet 主干网络，通过部分卷积（Partial Convolution）削减计算冗余，实现结构级轻量化，大幅降低参数量。
* DL-DHSA动态轻量化分桶自注意力替换原始 AIFI 模块，通过动态门控机制增强特征鲁棒性，有效提升模型在阴影、雨雪等复杂环境下的检测精度。
* CSUB通道位移上采样块引入零参数上采样策略，增强解码器对高分辨率特征的感知与融合，提升对长距离、极小尺寸交通标志的定位精度。

🚀 性能优势 (Performance Highlights)我们在 CCTSDB 2021 和 TT100K 两个数据集上对 LiteAdapt-DETR 进行了综合评估，结果显示模型在轻量化和精度上均超越基准。
  * 结构和效率优化相较于原始 RT-DETR 模型，LiteAdapt-DETR 显著降低了计算资源需求：指标LiteAdapt-DETRRT-DETR 基准优化幅度参数量 (Params)降低约 32%运算量 (FLOPs)降低约 30%推理延迟 (Latency)降低约 12%  
  * 泛化精度提升LiteAdapt-DETR 在两个不同数据域上的 $\text{mAP@0.5}$ 均获得提升，证明了其结构的稳定性和跨域泛化能力。数据集LiteAdapt-DETR mAP@0.5相比 RT-DETR 提升CCTSDB 2021)+0.5% TT100K+0.6%  


⚙️ 快速开始 (Quick Start)重要说明： 本仓库仅包含 LiteAdapt-DETR 的三个核心改进模块（HSPNet, DL-DHSA, CSUB）的实现代码。用户需克隆原始 RT-DETR 仓库，并将本仓库提供的模块代码集成到相应位置，以构建完整的 LiteAdapt-DETR 模型。


1. 安装 PyTorch 
```pip install torch torchvision torchaudio```

2. 获取 RT-DETR 基准框架请克隆原始 RT-DETR 或其官方实现仓库：
```git clone [https://github.com/lyuwenyu/RT-DETR.git](https://github.com/xxxxx/RT-DETR.git)```     
```cd RT-DETR```


3. 集成 LiteAdapt-DETR 模块将本仓库提供的 hspnet.py, dl_dhsa.py, 和 csub.py 等模块代码文件复制到 RT-DETR 框架的对应目录中（例如 ultralytics/nn/modules 或 models/backbones）。

4. 数据集准备本项目使用 CCTSDB 2021 和 TT100K 数据集。请按照 RT-DETR 框架要求准备数据路径。
```/liteadapt_detr/data
    /CCTSDB2021
        /images
        /labels
    /TT100K
        /images
        /labels
```

CCTSDB2021的百度网盘下载地址为https://pan.baidu.com/s/13ZjIKXoTIBD_0fJ1rdWSRA 提取码pu2p  
TT100K下载地址为https://cg.cs.tsinghua.edu.cn/traffic-sign/
