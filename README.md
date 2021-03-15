# Pix2Pix-Keras
基于pix2pix模型的动漫图片自动上色 2019-2-25

###### 环境
- tensorflow-gpu                     1.12.0
- Keras                              2.2.4

### 数据集的准备：
- 把训练的彩色图片放入datasets\OriginalImages文件夹
- 运行prepare.py进行数据集的处理与准备
- 注意：当前datasets中的训练集过少，可增加训练集进行训练，防止过拟合
### 训练模型：
- 若要加载预训练权重，将权重文件放入weights文件夹
- 最新权重(在30020张图片的大数据集下训练后的权重)下载
- 链接：https://pan.baidu.com/s/1tmfMTCpAFVpC6Z7GWvkrfA 提取码：cac2 
- 在demo.py中新建一个pix2pix模型实例
- 调用实例中的train函数进行训练
### 为新的图片上色:
- 将权重文件放入weights文件夹
- 新建一个pix2pix模型实例
- 调用demo.py中的predict_single_image函数进行图片上色
### Demo(中间的为AI预测，右边的为原始图像):

<div align="center">
  <img src="images/sample_1.jpg" height="256" width="256" >
  <img src="images/sample_2.jpg" height="256" width="256" >
  <img src="images/sample_4.jpg" height="256" width="256" >
  <img src="images/sample_5.jpg" height="256" width="256" >
  <img src="images/sample_7.jpg" height="256" width="256" >
  <img src="images/sample_13.jpg" height="256" width="256" >
  <img src="images/sample_3.jpg" height="256" width="256" >
  <img src="images/sample_8.jpg" height="256" width="256" >
  <img src="images/sample_9.jpg" height="256" width="256" >
</div>

### 泛化能力测式（给漫画上色）：

<div align="center">
  <img src="images/test_1.jpg" height="256" width="265" >
  <img src="images/generate_test_1.jpg" height="256" width="256" >
</div>

<div align="center">
  <img src="images/test_2.jpg" height="256" width="256" >
  <img src="images/generate_test_2.jpg" height="256" width="256" >
</div>
