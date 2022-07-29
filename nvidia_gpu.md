系统： ubuntu18.04

## 安装
知乎文章 [【Ubuntu 18.04 安装 NVIDIA 显卡驱动】](https://zhuanlan.zhihu.com/p/59618999) 写的很详细，总结的很全了

个人推荐下面两种方法之一：

### 法一：自动安装
自动安装推荐版本
```bash
sudo ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
```
### 法二：PPA安装
添加库并安装指定版本，命令2输入连续两次按 tab 键可以查看当前可用版本，选择相应版本
```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt install nvidia-driver-440
```
### 关闭 x server
执行 nvidia-smi 命令时会看到下面：
![image.png](https://cdn.nlark.com/yuque/0/2020/png/211857/1592360474261-a11bb312-9d8c-45c8-9024-521210a37ec8.png#align=left&display=inline&height=66&margin=%5Bobject%20Object%5D&name=image.png&originHeight=132&originWidth=888&size=9469&status=done&style=none&width=444)
这是 x server 进程，关闭即可
```bash
sudo systemctl stop lightdm
sudo systemctl disable light
```
## 卸载
```bash
sudo apt remove --purge nvidia*
```
使用 zsh 的话，上述命令可能会无效，使用下面的命令
```bash
sudo apt remove --purge "nvidia*"
```

## 多版本 cuda 共存
此处以　cuda10.0 与 cuda10.2 为例。电脑上已有 cuda10.0，现在再安装 cuda10.2，两个版本共存。

1. 安装最新版驱动
```bash
sudo apt install nvidia-driver-440 -y
```

2. 依据引导下载对应版本的 cuda 安装包，推荐 runfile版本。
> nvidia-cuda 最新版下载地址：[https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
> 各历史版本下载地址：[https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)

```bash
wget -c http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
```

3. 下载完成后执行安装
```bash
sudo sh cuda_10.2.89_440.33.01_linux.run
```

4. 安装过程中选择不安装显卡驱动，选择不覆盖 /usr/local/cuda 文件。
4. 安装完成后，在  /usr/local 文件夹下出现 cuda10.2 文件夹
4. 通过在 ~/.bashrc 中设置环境变量的方法来调整使用哪个 cuda 版本

.bashrc 中添加下面内容
```bash
export CUDA_HOME=/usr/local/my_cuda 
export PATH=$PATH:$CUDA_HOME/bin 
export LD_LIBRARY_PATH=/usr/local/my_cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

```
### 切换 cuda10.2
若需要使用 cuda10.2 ，则创建 cuda10.2 软链接到 my_cuda
```bash
cd /usr/local
sudo rm my_cuda
sudo ln -s /usr/local/cuda-10.2 /usr/local/my_cuda
source ~/.bashrc
```
查看cuda 版本是否切换成功
```bash
nvcc --version
```
### 切换 cuda10.0 版本
```bash
cd /usr/local
sudo rm my_cuda
sudo ln -s /usr/local/cuda-10.0 /usr/local/my_cuda
source ~/.bashrc
```

## 内核升级后无法与显卡通信
若更新系统后出现如下故障，则可能是升级内核后导致了驱动故障
> #### NVIDIA-SMI has failed because it couldn’t communicate with the NVIDIA driver 


参考文章 [【ubuntu环境下，系统无法与NVIDIA通信的解决方法】](https://wangpei.ink/2019/01/19/NVIDIA-SMI-has-failed-because-it-couldn't-communicate-with-the-NVIDIA-driver%E7%9A%84%E8%A7%A3%E5%86%B3%E6%96%B9%E6%B3%95/)

查看内核版本
```bash
uname -r
```
使用dkms 重新将驱动注册到内核
```bash
sudo apt-get install dkms
sudo dkms build -m nvidia -v 440.82   # 最后的版本号通过/usr/src目录下名为nvidia-***.**的文件夹获得
sudo dkms install -m nvidia -v 440.82
```
检查是否成果
```bash
nvidia-smi  # 正确显示显卡信息则成功
```
## GCC 版本问题
在上述安装过程中如果一直无法成功，检查系统 gcc 版本，gcc7 以上才行，gcc-4.8、gcc-4.9 都会导致失败。

查看当前 gcc 和 g++ 版本
```bash
gcc --version
g++ --version
```
查看系统中有的 gcc 版本
```bash
ls /usr/bin/gcc*
```
设置系统当前使用的 gcc 版本
```bash
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 40  # 最后的数字代表优先级
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 40  # 最后的数字代表优先级
```
选择 gcc-7
```bash
sudo update-alternatives --config gcc  # 输入对应的编号即可
```
用同样的方法配置 g++，将上述命令中的 gcc 换成 g++ 即可

## 切换内核问题
ubuntu 内核升级常常会导致驱动挂掉，切回原来的内核也是方法之一，在此记录下相关操作

参考：[如何降级/切换 Ubuntu 系统 Linux 内核启动版本](https://zhengdao.github.io/2018/10/09/switch-ubuntu-linux-kernel/)

1. 查看正在使用的内核版本
```bash
uname -r
```

2. 查看系统可用的 Linux 内核
```bash
grep menuentry /boot/grub/grub.cfg
```
查看子选项：
> ‘Ubuntu, with Linux 4.4.0-104-generic’


3. 设置内核启动版本
> sudo vi /etc/default/grub


> 将 GRUB_DEFAULT=0  // 0表示系统当前启动的内核序号
修改为想要启动的内核版本对应子选项：GRUB_DEFAULT=“Advanced options for Ubuntu > Ubuntu, with Linux 4.4.0-104-generic”

4. 检查是否有误
```bash
sudo grub-mkconfig -o /boot/grub/grub.cfg
```

5. 无误则更新 grub
```bash
sudo update grub
```

6. 重启系统
```bash
sudo reboot
```

7. 查看内核是否切换成功
```bash
uname -r
```

### Linux内核安装和卸载

1. 查看可安装内核
```bash
sudo apt-cache search linux-image | grep generic
```

2. 安装新内核
```bash
sudo apt-get install linux-image-4.15.0-106-generic
```

3. 卸载Linux内核
```bash
sudo apt-get remove linux-image-4.4.0-101-generic
```

