# HAVPRLab DeepAction Learning 
Human Activity & Visual Perception Research Lab (HAVPRLab) DeepAction Learning Resources.

欢迎提交PR, 持续更新中✨

![License](https://img.shields.io/badge/license-MIT-yellow)

## 🏷️ Learning Resources
### Books
* [深度学习上手指南](https://github.com/nndl/nndl.github.io/blob/master/md/DeepGuide.md)
### Videos
* **[如何读论文-李沐](https://www.bilibili.com/video/BV1H44y1t75x)** [BiliBili](https://www.bilibili.com/video/BV1H44y1t75x)
* 推荐李沐大神团队出品的精读论文系列 [[BiliBili]](https://space.bilibili.com/1567748478/channel/collectiondetail?sid=32744) [[GitHub]](https://github.com/mli/paper-reading) 
    * [视频理解论文串讲（上）](https://www.bilibili.com/video/BV1fL4y157yA)🔥
    * [视频理解论文串讲（下）](https://www.bilibili.com/video/BV11Y411P7ep)🔥
    * [双流网络论文逐段精读](https://www.bilibili.com/video/BV1mq4y1x7RU)🔥
    * [I3D 论文精读](https://www.bilibili.com/video/BV1tY4y1p7hq)🔥

    * [ResNet论文逐段精读](https://www.bilibili.com/video/BV1P3411y7nn)(基础)
    * [Transformer论文逐段精读](https://www.bilibili.com/video/BV1pu411o7BE)（基础）
    * [ViT论文逐段精读](https://www.bilibili.com/video/BV15P4y137jb)（基础)

## 🏷️ Paper Lists
* [awesome-action-recognition](https://github.com/jinwchoi/awesome-action-recognition)(Action Recognition 论文合集)🔥
* [Video Swin Transformer](https://arxiv.org/abs/2106.13230) [[PDF]](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Video_Swin_Transformer_CVPR_2022_paper.pdf) [[Code]](https://github.com/SwinTransformer/Video-Swin-Transformer)


## 🏷️ Training Skills
* [PyTorch 技巧](https://github.com/lartpang/PyTorchTricks)🔥
* [PyTorch 炼丹过程常用小代码](pytorch_snippets.md)
* [SWA](https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/) (🔥无痛涨点训练方法)
* [调参-如何确定学习率 lr](https://www.yuque.com/explorer/blog/sv37zs)
* [Label Smoothing](https://github.com/pytorch/pytorch/issues/7455)
    * [torch.nn.CrossEntropyLoss 已支持](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
    * 一个 PyTorch 多分类简单实现
    ```
    class LabelSmoothingLoss(nn.Module):
        def __init__(self, classes, smoothing=0.0, dim=-1):
            super(LabelSmoothingLoss, self).__init__()
            self.confidence = 1.0 - smoothing
            self.smoothing = smoothing
            self.cls = classes
            self.dim = dim
            
            def forward(self, pred, target):
                pred = pred.log_softmax(dim=self.dim)
                with torch.no_grad():
                    # true_dist = pred.data.clone()
                    true_dist = torch.zeros_like(pred)
                    true_dist.fill_(self.smoothing / (self.cls - 1))
                    true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
    ```


##  🏷️ Github Repos
### Action-Related
* [mmaction2](https://github.com/open-mmlab/mmaction2)(知名框架，包含动作识别算法多)🔥
* [TSM](https://github.com/mit-han-lab/temporal-shift-module)(动作识别 2DCNN 经典算法)
* [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch)(动作识别 3DCNN 经典算法)🔥
* [Video-Swin-Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer) (当红辣子鸡Transformer )🔥
* [MARS](https://github.com/craston/MARS) (知识蒸馏算法)

### Optical Flow
* [denseflow](https://github.com/open-mmlab/denseflow)(TVL1等光流提取)
* [RAFT](https://github.com/princeton-vl/RAFT)(ECCV2020 Best Paper 深度学习高质量光流提取)
### 其他代码库
* [External-Attention-pytorch](https://github.com/xmu-xiaoma666/External-Attention-pytorch)(各种 Attention 机制的核心实现，简单易懂)🔥

## 🏷️  Useful Tools

* [decord](https://github.com/dmlc/decord) (高性能视频读取库)
* [profile](https://github.com/shibing624/python-tutorial/blob/master/06_tool/profiler%E5%B7%A5%E5%85%B7.md) (Python 代码性能分析)
### Linux 使用
* [Linux 就该这么学](https://www.linuxprobe.com/) (免费PDF教材)
* [Oh-my-zsh](https://zhuanlan.zhihu.com/p/35283688) 🚀 (配置好用的命令行)
* [Tmux](https://zhuanlan.zhihu.com/p/98384704) (远程连接服务器后台运行代码) [使用手册](http://louiszhai.github.io/2017/09/30/tmux/)

## 🏷️ Others
* [Cuda 安装和问题解决](./nvidia_gpu.md)

* [Anaconda 安装使用](https://blog.csdn.net/a745233700/article/details/109376667)(方便 Python 环境管理)

