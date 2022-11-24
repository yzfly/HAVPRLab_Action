# HAVPRLab DeepAction Learning 
Human Activity & Visual Perception Research Lab (HAVPRLab) DeepAction Learning Resources.

✨持续更新中, 欢迎提交 PR ✨

![License](https://img.shields.io/badge/license-MIT-yellow)

## 🏷️ Learning Resources


#### 基础
* [知名的吴恩达深度学习教程](https://mooc.study.163.com/university/deeplearning_ai#/c)
* [斯坦福 cs231n 课程资源](https://cs231n.github.io/)
* [Fast.ai 出品的深度学习基础教程](https://www.fast.ai/)
* [深度学习上手指南](https://github.com/nndl/nndl.github.io/blob/master/md/DeepGuide.md)
* **[如何读论文-李沐](https://www.bilibili.com/video/BV1H44y1t75x)** [BiliBili](https://www.bilibili.com/video/BV1H44y1t75x)
* 李沐大神团队出品的精读论文系列 [[BiliBili]](https://space.bilibili.com/1567748478/channel/collectiondetail?sid=32744) [[GitHub]](https://github.com/mli/paper-reading) 
    * [ResNet论文逐段精读](https://www.bilibili.com/video/BV1P3411y7nn)(基础)
    * [Transformer论文逐段精读](https://www.bilibili.com/video/BV1pu411o7BE)（基础）
    * [ViT论文逐段精读](https://www.bilibili.com/video/BV15P4y137jb)（基础)
    * more ...

#### Action Recognition
* [视频理解论文串讲（上）](https://www.bilibili.com/video/BV1fL4y157yA)🔥
* [视频理解论文串讲（下）](https://www.bilibili.com/video/BV11Y411P7ep)🔥
* [双流网络论文逐段精读](https://www.bilibili.com/video/BV1mq4y1x7RU)🔥
* [I3D 论文精读](https://www.bilibili.com/video/BV1tY4y1p7hq)🔥
* [管中窥”视频“，”理解“一斑 —— 视频理解概览](https://techbeat.net/article-info?id=2200) (2D, 3D方法的概览)


## 🏷️ Paper Lists
### Action Recognition
* [awesome-action-recognition](https://github.com/jinwchoi/awesome-action-recognition)(Action Recognition 论文合集)🔥
* [TSN (ECCV 2016)](https://arxiv.org/abs/1608.00859) [[Code](https://github.com/yjxiong/temporal-segment-networks)] ⭐
* [I3D (CVPR 2017)](https://arxiv.org/abs/1705.07750) [[Code: kinetics-i3d](https://github.com/deepmind/kinetics-i3d)][[Code:pytorch-i3d](https://github.com/piergiaj/pytorch-i3d)] ⭐
* [3D-ResNets (CVPR 2018)](https://openaccess.thecvf.com/content_cvpr_2018/html/Hara_Can_Spatiotemporal_3D_CVPR_2018_paper.html) [[Code](https://github.com/kenshohara/3D-ResNets-PyTorch)] ⭐
* [TSM (ICCV 2019) ](http://arxiv.org/abs/1811.08383) [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Lin_TSM_Temporal_Shift_Module_for_Efficient_Video_Understanding_ICCV_2019_paper.pdf)][[Code](https://github.com/mit-han-lab/temporal-shift-module)] ⭐
* [TEA (CVPR 2020)](https://arxiv.org/abs/2004.01398) [[Code](https://github.com/Phoenix1327/tea-action-recognition)]
* [SlowFast (ICCV 2019)](https://arxiv.org/abs/1812.03982) [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Feichtenhofer_SlowFast_Networks_for_Video_Recognition_ICCV_2019_paper.pdf)] [[Code: official](https://github.com/facebookresearch/SlowFast)] [Code: mmaction2](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/slowfast/README.md) ⭐
* [X3D (CVPR 2020)](https://arxiv.org/abs/2004.04730) [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Feichtenhofer_X3D_Expanding_Architectures_for_Efficient_Video_Recognition_CVPR_2020_paper.html)] [[Code: official](https://github.com/facebookresearch/SlowFast)] [[Code: mmaction2](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/x3d/README.md)]
* [TDN (CVPR 2021)](https://arxiv.org/abs/2012.10071) [[Paper](https://arxiv.org/abs/2012.10071)] [[Code](https://github.com/MCG-NJU/TDN)] ⭐
* [TimeSformer (ICML 2021)](https://arxiv.org/pdf/2102.05095.pdf) [[Code](https://github.com/facebookresearch/TimeSformer)] 
* [Motionformer (NeurIPS 2021)](https://facebookresearch.github.io/Motionformer/) [[Paper](https://arxiv.org/abs/2106.05392)] [[Code](https://github.com/facebookresearch/Motionformer)]
* [Video Swin Transformer (CVPR 2022)](https://arxiv.org/abs/2106.13230) [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Video_Swin_Transformer_CVPR_2022_paper.Paper)] [[Code](https://github.com/SwinTransformer/Video-Swin-Transformer)]⭐

* [FineDiving: A Fine-grained Dataset for Procedure-aware Action Quality Assessment](https://finediving.ivg-research.xyz/) [[Paper](https://arxiv.org/pdf/2204.03646.pdf)] [[Code & Dataset](https://github.com/xujinglin/FineDiving)] (CVPR 2022 Oral | 清华开源FineDiving：细粒度动作质量评估数据集)
* [Expanding Language-Image Pretrained Models for General Video Recognition](https://github.com/microsoft/VideoX/tree/master/X-CLIP) [[Paper](https://arxiv.org/abs/2208.02816)] [[Code](https://github.com/microsoft/videox)] (ECCV 2022 Oral | 微软开源 X-Clip,动作识别，小样本学习)
* [UniFormer: Unified Transformer for Efficient Spatiotemporal Representation Learning
](https://github.com/microsoft/VideoX/tree/master/X-CLIP) [[Paper](https://arxiv.org/abs/2201.04676)] [[Code](https://github.com/Sense-X/UniFormer)] (Uniformer ICLR2022 (评分 8868, Top 3%), 比较有意思的 CNN 与 Transformer 相互启发的工作，作者也使用 Uniformer 打了 CVPR dark action recognition 的比赛)

### Others
* [ConvGRU (ICLR 2016)](https://arxiv.org/abs/1511.06432) [[Paper](https://arxiv.org/abs/1511.06432)]
* [NeRF (ECCV 2020 oral)](https://www.matthewtancik.com/nerf) [[Paper](https://arxiv.org/abs/2003.08934)][[Code](https://github.com/yenchenlin/nerf-pytorch)] (Neural Radiance Fields 用于 AI 内容生成)
* [RepVGG (CVPR 2021)](https://arxiv.org/abs/2101.03697) [[Paper](https://arxiv.org/abs/2101.03697)][[Code](https://github.com/DingXiaoH/RepVGG)] [[知乎](https://zhuanlan.zhihu.com/p/344324470)] (清华大学 VGG 网络的复兴)
* [MLP-Mixer (NIPS 2021)](https://papers.nips.cc/paper/2021/hash/cba0a4ee5ccd02fda0fe3f9a3e7b89fe-Abstract.html) [[Paper](https://papers.nips.cc/paper/2021/file/cba0a4ee5ccd02fda0fe3f9a3e7b89fe-Paper.pdf)][[Code](https://github.com/lucidrains/mlp-mixer-pytorch)] (Google 纯 MLP 架构卷土重来)
* [ACmix (CVPR 2022)](https://arxiv.org/abs/2111.14556) [[Paper](https://arxiv.org/pdf/2111.14556v1.pdf)][[Code](https://github.com/LeapLabTHU/ACmix)] (DenseNet 一作黄高老师组 CNN与transformer 融合的工作)
* [Stable Diffusion (CVPR 2022 oral)](https://arxiv.org/abs/2111.14556) [[Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html)][[Code](https://github.com/CompVis/stable-diffusion)] [[在线体验](https://huggingface.co/spaces/stabilityai/stable-diffusion)](火爆国内外，从文字描述生成图片内容)

### 科研
* [IEEE论文投稿流程（格式说明，新手指南，模板）](https://zhuanlan.zhihu.com/p/317281632)
* [机器学习科研：如何高效管理代码和实验](https://zhuanlan.zhihu.com/p/559085050)

## 🏷️ Funny AI
* [🤗 huggingface](https://huggingface.co/) (AI 模型试玩，数据、模型发布社区)

## 🏷️ Training Skills
* [PyTorch 技巧](https://github.com/lartpang/PyTorchTricks)🔥
* [ResNet strikes back: An improved training procedure in timm](https://paperswithcode.com/paper/resnet-strikes-back-an-improved-training) [[Paper](https://openreview.net/pdf?id=NG6MJnVl6M5)]🔥
* [深度学习技巧](https://github.com/Conchylicultor/Deep-Learning-Tricks)
* [PyTorch 炼丹过程常用小代码](pytorch_snippets.md)
* [SWA](https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/) (🔥无痛涨点训练方法)
* [EMA](https://github.com/lucidrains/ema-pytorch) (🔥指数滑动平均无痛涨点)
* [Fast.ai 推崇的 One Cycle 训练策略](https://fastai1.fast.ai/callbacks.one_cycle.html)
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
* [IG65M](https://github.com/moabitcoin/ig65m-pytorch) (Models and weights pre-trained on 65MM Instagram videos.)

### Optical Flow
* [denseflow](https://github.com/open-mmlab/denseflow)(TVL1等光流提取)
* [mmflow](https://github.com/open-mmlab/mmflow)( 商汤开源的光流提取代码库，包含多种知名算法)
* [PWCNet](https://github.com/NVlabs/PWC-Net)( PWC-Net 光流官方实现)
* [RAFT](https://github.com/princeton-vl/RAFT)(ECCV2020 Best Paper 深度学习高质量光流提取)

### Dataset-Related
* [Common Visual Data Foundation](https://github.com/cvdfoundation) (Kinetics400/600/700、AVA 等大型数据集便利下载)
* [VoTT](https://github.com/microsoft/VoTT) (微软出品的好用的标注工具) [[BiliBili](https://www.bilibili.com/video/BV1854y127gT)]

### 其他代码库
* [External-Attention-pytorch](https://github.com/xmu-xiaoma666/External-Attention-pytorch)(各种 Attention 机制的核心实现，简单易懂)🔥
* [Timm](https://github.com/rwightman/pytorch-image-models) (各种知名 Backbone 实现) 🔥
* [transformers](https://github.com/huggingface/transformers)(transformers 实现) 🔥

### Github 加速 ✈️

日常使用 github 下载代码时如果遇到网速慢、无法连接的情况，可以使用下面的方法加速

#### 1. 方法一（推荐）： [ghproxy.com](https://ghproxy.com/)

常规的面向 GitHub 的 clone 命令可能如下：
```
git clone https://github.com/author/repo
```
使用 ghproxy 代理加速后（添加 https://ghproxy.com/ 即可）：
```
git clone https://ghproxy.com/https://github.com/author/repo
```

#### 2. 方法二（推荐）： [fastgit.org](https://doc.fastgit.org/zh-cn/guide.html)

常规的面向 GitHub 的 clone 命令可能如下：
```
git clone https://github.com/author/repo
```
使用 fastgit 时， 命令修改为如下即可：
```
git clone https://hub.fastgit.xyz/author/repo
```


## 🏷️  Useful Tools

* [decord](https://github.com/dmlc/decord) (高性能视频读取库)
* [profile](https://github.com/shibing624/python-tutorial/blob/master/06_tool/profiler%E5%B7%A5%E5%85%B7.md) (Python 代码性能分析)

## 🏷️ Linux 使用
* [Linux 就该这么学](https://www.linuxprobe.com/) (免费PDF教材)
* [Oh-my-zsh](https://zhuanlan.zhihu.com/p/35283688) 🚀 (配置好用的命令行)
* [Tmux](https://zhuanlan.zhihu.com/p/98384704) (远程连接服务器后台运行代码) [使用手册](http://louiszhai.github.io/2017/09/30/tmux/)

## 🏷️ Github :octocat:
* [Best-README-Template](https://github.com/yzfly/Best-README-Template)

## 🏷️ Others
* [Cuda 安装和问题解决](./nvidia_gpu.md)

* [Anaconda 安装使用](https://blog.csdn.net/a745233700/article/details/109376667)(方便 Python 环境管理)

* [VS Code 远程开发](https://zhuanlan.zhihu.com/p/141344165) (远程连接服务器开发程序， PyCharm 也具备该功能)

