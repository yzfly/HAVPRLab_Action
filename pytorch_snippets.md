## 环境配置
- Anaconda + Python3.6+
```bash
conda update -n base -c defaults conda #update conda
conda create -n new_env python=3.8
conda activate new_env

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

- pip 
```bash
python -m pip install --upgrade pip
pip install scipy numpy pandas scikit-learn pillow opencv-python \
matplotlib tqdm  ipdb
```
## GitHub 加速

1. 使用国内镜像加速:  [**FastGit**](https://doc.fastgit.org/zh-cn/guide.html)
```bash
git config --global url."https://hub.fastgit.org/".insteadOf "https://github.com/"
git config protocol.https.allow always

# 取消设置
git config --global --unset url.https://github.com/.insteadof
```
> [https://github.com/](https://raw.githubusercontent.com/) 修改为 [https://hub.fastgit.org/](https://raw.fastgit.org/)
> [https://raw.githubusercontent.com/](https://raw.githubusercontent.com/) 修改为 [https://raw.fastgit.org/](https://raw.fastgit.org/)


## 运行参数设置

- argparse
```python
import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--imfile1', type=str, default='FLOW_DEMOS/tcm_results/img_1.jpg',
                        help='Input image1 path')
    parser.add_argument('--results-path', type=str, default='./results/',
                        help='results image save path')
    parser.add_argument('--pretrained', type=str, default='RAFT_things_fullmodel.pth',
                        help='pretrained model')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++', 'layercam'],
                        help='Can be gradcam/gradcam++/layercam')

    args = parser.parse_args()
    return args
```

- [Fire](https://github.com/google/python-fire)
```python
# pip install fire
import fire

def hello(name="World"):
  return "Hello %s!" % name

if __name__ == '__main__':
  fire.Fire(hello)

```
## 数据加载
### 图片读取
> OpenCV 方法一般更快，因为 PIL 的图片读取是不完全的，所以测试时会快一些。其他 Python 库中提供的图像读取方法大多数是 PIL 的封装，在此不赘述。

- PIL 方法
```python
from PIL import Image
import numpy as np

im = Image.open(img_file)
im = im.resize((new_width, new_height))
im.save("name.jpg")

im = np.array(im).astype(np.uint8)
```

- OpenCV 方法
```python
import cv2
im = cv2.imread(img_file)[:, :, ::-1]  # bgr to rgb
# im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # method 2: bgr to rgb
im = cv2.resize(im, (224, 224))
cv2.imwrite('name.jpg',im) 

rgb_img = np.float32(im) / 255
```
## 模型定义

- 尽量使用nn.Sequential()
- 经常使用的结构封装成子Module（比如GoogLeNet的Inception结构，ResNet的Residual Block结构）
- 重复且有规律性的结构，用函数生成（ResNet多种变体都是由多个重复卷积层组成）

将模型定义保存在 models/ 目录下，在 models/__init__.py 文件中引入（以 AlexNet 为例子）
```python
from .AlexNet import AlexNet
```
之后引入模型可以有下面三种方式：
```python
# Method 1
from models import AlexNet

# Method 2
import models
model = models.AlexNet()

# Method 3
import models
model = getattr(models, 'AlexNet')()
```
其中第三种方式可以使用字符串的方式指定模型，十分方便。

## 指定GPU/CPU设备
```python
if args.use_gpu and torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')
```
## 混合精度训练

- [PyTorch的自动混合精度（AMP）](https://zhuanlan.zhihu.com/p/165152789)
```python
from torch.cuda.amp import autocast as autocast

# 创建model，默认是torch.FloatTensor
model = Net().cuda()
optimizer = optim.SGD(model.parameters(), ...)

# 在训练最开始之前实例化一个GradScaler对象
scaler = GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()

        # 前向过程(model + loss)开启 autocast
        with autocast():
            output = model(input)
            loss = loss_fn(output, target)

        # Scales loss. 为了梯度放大.
        scaler.scale(loss).backward()

        # scaler.step() 首先把梯度的值unscale回来.
        # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
        # 否则，忽略step调用，从而保证权重不更新（不被破坏）
        scaler.step(optimizer)

        # 准备着，看是否要增大scaler
        scaler.update()
```
## 显存节省

- 尽量使用  inplace=True 操作
- del 删除不必要中间变量
- 显存清理 torch.cuda.empty_cache()
- checkpoint() 时间换空间： [torch.utils.checkpoint 简介 和 简易使用](https://blog.csdn.net/one_six_mix/article/details/93937091)

## 论文可复现

- 指定随机数种子
```python
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()
```

- 使用确定性操作 ( PyTorch >= 1.7 )
```python
torch.set_deterministic(True)
```
## Grad-Cam 热力图绘制

- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
```python
pip install grad-cam
```
## 参考资料

- [PyTorch 编程风格](https://github.com/IgorSusmelj/pytorch-styleguide)
- [PyTorch 常用技巧](https://github.com/lartpang/PyTorchTricks)
