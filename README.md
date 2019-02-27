# Pix2Pix-Keras
基于pix2pix模型的动漫图片自动上色 2019-2-25
### 数据集的准备：
1. 把训练的彩色图片放入datasets\OriginalImages文件夹
2. 运行prepare.py进行数据集的处理与准备
### 训练模型：
1. 将权重文件放入weights文件夹
2. 训练好的最新权重下载地址：https://pan.baidu.com/s/1IUamednTkPE0qPw736Crzw
3. 在demo.py中新建一个pix2pix模型实例
4. 调用实例中的train函数进行训练
### 为新的图片上色:
1. 将权重文件放入weights文件夹
2. 新建一个pix2pix模型实例
2. 调用demo.py中的predict_single_image函数进行图片上色

