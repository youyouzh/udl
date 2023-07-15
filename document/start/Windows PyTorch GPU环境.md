# Windows PyTorch GPU环境

## Windows安装PyTorch GPU环境

基本情况：2023年07月，Python版本3.10，PyTorch最新版2.0.1，cuda支持到12.0。要注意安装PyTorch的时候CPU和GPU版本是不一样的。

官网地址：<https://pytorch.org/get-started/locally/>

### 安装CUDA

在cmd中输入命令`mvidia-smi`可以查看显卡信息，以及支持的CUDA版本。

- 查看显卡算力： [点击查看显卡算力表](https://developer.nvidia.com/cuda-gpus)，有不同显卡的算力，算力至少3.5以上才能安装CUDA
- 查看显卡驱动版本： [点击下载更新驱动](https://www.nvidia.com/Download/index.aspx)，右键空白地方 -> NIVIDIA 控制面板 -> 左下角系统消息 -> 显示 -> 查看驱动版本
- 查看显卡支持CUDA版本： [点击查看显卡驱动对于CUDA版本](https://docs.nvidia.com/cuda/archive/10.0/cuda-toolkit-release-notes/index.html)，按照步骤3，点击组件，其中有`NVCUDA.dll`，就是当前支持的CUDA版本
- 安装Visual Studio 2015、2017 和 2019 的 Microsoft Visual C++ 可再发行组件： [软件下载地址](https://support.microsoft.com/zh-cn/help/2977003/the-latest-supported-visual-c-downloads)，下载并安装`vc_redist.x64.exe`，安装完要重启
- 安装CUDA： [CUDA下载地址](https://developer.nvidia.com/cuda-toolkit-archive)，通过上面几步，找到合适自己环境版本的CUDA，**版本一定要对得上**
- 安装cuDNN，cuDNN（CUDA Deep Neural Network library）是NVIDIA打造的针对深度神经网络的加速库，[cuDNN地址](https://developer.nvidia.com/rdp/cudnn-archive)，注意要和CUDA版本匹配，下载后解压分别替换CUDA安装目录下的文件夹`include`、`lib`、`bin`即可

这儿安装[CUDA-11.6.0](https://developer.nvidia.com/cuda-11-6-2-download-archive)，下载后直接安装即可。安装完成后需要添加下面的环境变量：

- CUDA_PATH: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6`
- CUDA_PATH_V11_6: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6`
- NVTOLLSEXT_PATH: `C:\Program Files\NVIDIA GPU Computing Toolkit\NvTollsExt`
- Path中需要将`include`、`bin`、`lib`、`libnvvp`四个目录都添加到进去

在cmd中执行命令`nvcc --version`和`set data`查看是否安装成功。

### 安装PyTorch的GPU版本

可以到官网下载PyTorch的GPU版本：<https://pytorch.org/get-started/previous-versions/>，也可以直接安装，注意如果之前安装过CPU版本，需要先卸载掉。注意版本号中待cu的表示GPU版本，否则表示CPU版本。

```shell
# 查看安装的包版本
pip list

# 卸载以前的版本
pip uninstall torch torchvision torchaudio

# 安装cuda-v11.6版本对应的PyTorch，安装包有好几G，需要几分钟
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu116
```

安装完成后，执行`torch.cuda.is_available()`返回True说明已经在使用GPU版本了。这儿安装的PyTorch版本是`1.13.1+cu116`并不是`2.0.1`版本。**注意安装的时候不要勾选`Driver`**。

### 安装PyTorch的2.0.1版本

`CUDA-11.8`下载地址：<https://developer.nvidia.com/cuda-11-8-0-download-archive>。可以同时安装多个CUDA版本，注意环境变量配置决定用哪一个版本。

截止到2023年7月PyTorch最新版为`2.0.1`，支持的`CUDA-11.7`和`CUDA-11.8`，可以通过命令进行安装：`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`。

### 设置使用GPU进行训练

单个GPU或CPU可通过`torch.cuda.is_available()`来设置是否使用GPU，对于多GPU设备而言，如果要定义设备，则需要使用torch.nn.DataParallel来将模型加载到多个GPU中。

```python
import torch
from torch import nn

# 单GPU或者CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 判断GPU设备大于1，也就是多个GPU，则使用多个GPU设备的id来加载
if torch.cuda.device_count() > 1:
  model = nn.DataParallel(model, device_ids=[0,1,2])
```

在pytorch中可以通过`.cuda()`以及`.to(device)`两种方式将模型和数据复制到GPU的显存中计算。以下说明是在单GPU或者CPU中的情况。

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 或device = torch.device("cuda:0")
# 或device = torch.device("cuda:1")
print(device)

# 需要提前初始化一下model
model.to(device)

# 计算的时候将数据加载到device
for batch_idx, (img, label) in enumerate(train_loader):
    img=img.to(device)
    label=label.to(device)

#指定某个GPU
os.environ['CUDA_VISIBLE_DEVICE']='1'

# 判断GPU是否能用，如果能用，则将数据和模型加载到GPU上
if torch.cuda.is_available():
    data  = data.cuda()
    model = model.cuda()
```

## modelscop安装

```shell
# 创建conda环境
conda create -n modelscope python=3.7
conda activate modelscope

# 安装pytorch
pip3 install torch torchvision torchaudio

# 安装pytorch-使用清华的源
pip3 install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装深度学习核心框架
pip install modelscope

# 所有ModelScope平台上支持的各个领域模型功能
pip install "modelscope[audio,cv,nlp,multi-modal,science]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

## 版本对应关系

- torch           1.12.1
- torchaudio      0.13.1
- torchmetrics    0.11.1
- torchvision     0.13.1
- python          3.7.6
- cuda            11.6.0
