# Dive into Deep Learning

动手学深度学习

📖网页: [CN](https://zh.d2l.ai/index.html); [EN](https://d2l.ai/)

📺视频: [Bilibili](https://space.bilibili.com/1567748478/channel/seriesdetail?sid=358497)

## 知识框架

![book content](./img/book-org.svg)

## 计划

### 目标

首战，2月拿下🎉🎉🎉

### WBS

| 状态 |  #   |       章节       | 页码 | 计划时间 | 完成时间 |           备注           |
| :--: | :--: | :--------------: | :--: | :------: | :------: | :----------------------: |
|  ✅   |  1   |       前言       |  22  |          | 1月26日  |                          |
|  ✅   |  2   |     预备知识     |  44  |          | 1月26日  |                          |
|  ✅   |  3   |   线性神经网络   |  42  | 1月26日  | 1月26日  |                          |
|  ✅   |  4   |    多层感知机    |  64  | 1月28日  | 1月28日  | 本章后半部分有待深入学习 |
|  ✅   |  5   |   深度学习计算   |  26  | 1月29日  | 1月29日  |                          |
|  ✅   |  6   |   卷积神经网络   |  30  | 1月30日  | 2月7日   |                          |
|  ✅   |  7   | 现代卷积神经网络 |  42  |  2月8日  | 2月8日   |                          |
|  ✅   |  8   |   循环神经网络   |  46  | 2月16日 | 2月26日 |                          |
|  ✅   |  9   | 现代循环神经网络 |  48  | 2月18日 | 2月28日 |                          |
|  ✅   |  10  |    注意力机制    |  46  | 2月20日 | 2月25日  |                          |
|  ⏳   |  11  |     优化算法     |  80  |          |          |                          |
|  ✅   |  12  |     计算性能     |  46  |  2月9日  | 2月9日         |  计算机硬件有待进一步理解 |
|  ✅  |  13  |    计算机视觉    | 100  | 2月11日  | 2月14日 | R-CNN需深入学习 |
|  ⏳   |  14  |   NLP: 预处理    |  54  |          |          |                          |
|  ⏳   |  15  |    NLP: 应用     |  38  |          |          |                          |

## Installation

```bash
conda create --name dl python=3.8 -**y**
pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip3 install d2l
```

在**Amax 机器**机器运行

```python
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
```

## Ref

Jupyter: [Amax](http://192.168.135.15:8888/tree?)

👨‍💻Code: [d2l-zh-2.0.0.zip](https://zh-v2.d2l.ai/d2l-zh-2.0.0.zip)
