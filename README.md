# Pelvis Angle System

这是一个自动识别骨盆关键点并计算角度的带前端的系统
<br>

web应用框架：flask

<br>

前端：HTML+boostrap+js

<br>

关键点识别模型：[YOLO](https://github.com/MIRACLE-Center/YOLO_Universal_Anatomical_Landmark_Detection)

## 目录

[TOC]

## 界面展示

![](data/1.png)

## 安装

**Clone the repo and install dependencies.<br>**

```shell
git clone https://github.com/huangchenxi304/PelvisAngle
pip install - r requirements.txt
```

## Project Structure

```
PelvisAngle
│  app.py（★★★前端与后端交互的地方）
│  README.md（☆☆☆）
│  requirements.txt（☆☆☆）
│      
├─static（☆☆☆，所有前端资源存放的地方）
│  ├─css
│  │      
│  ├─images（☆☆☆，只有确保图片在这个文件夹里，图片才能在前端显示出来）
│  │      1.2.156.112605.215507183267565.210106021535.4.9624.104022.jpg
│  │      1.2.156.112605.215507183267565.210106021535.4.9624.104022.png
│  │      
│  └─js
│          
├─templates（★★☆前端的页面在这里写，包括样式、布局与一些简单的逻辑）
│      base.html
│      image.html
│      
├─UALD
│  │          
│  ├─data（★★★数据集放这里！）
│  │  ├─gupen
│  │  │  ├─labels
│  │  │  │      1.3.12.2.1107.5.3.58.40252.12.202101010955490437.json
│  │  │  │      1.3.12.2.1107.5.3.58.40252.12.202101011540210171.json
│  │  │  │      ......
│  │  │  │      
│  │  │  └─pngs
│  │  │          1.3.12.2.1107.5.3.58.40252.12.202101010955490437.jpg
│  │  │          1.3.12.2.1107.5.3.58.40252.12.202101011540210171.jpg
│  │  │          ......
│  │          
│  ├─runs（存放unet和GU2Net两种模型的训练、验证和测试结果，包括权重文件）
│  │  └─unet2d_runs
│  │  │        config_origin.yaml
│  │  │        config_train.yaml
│  │  │   
│  │  ├─GU2Net_runs
│  │  │  │  config_origin.yaml（☆☆☆）
│  │  │  │  config_single.yaml（☆☆☆）
│  │  │  │  config_test.yaml（☆☆☆）
│  │  │  │  config_train.yaml（☆☆☆）
│  │  │  │  learning_rate.png（☆☆☆）
│  │  │  │  loss.png（☆☆☆）
│  │  │  │  network_graph.txt（☆☆☆）
│  │  │  │  
│  │  │  ├─checkpoints（★☆☆训练生成的GU2Net模型权重文件）
│  │  │  │      best_GU2Net_runs_epoch098_train5234.624233_val1716.078491.pt
│  │  │  │      best_GU2Net_runs_epoch098_train73.344884_val17.906089.pt
│  │  │  │      ......
│  │  │  │      train_val_loss.txt
│  │  │  │      
│  │  │  └─results（☆☆☆，都是一些没什么用的结果，用不到）
│  │  │      ├─loss
│  │  │      │      epoch_098_loss_54.583926.txt
│  │  │      │      epoch_099_loss_17.908456.txt
│  │  │      │      ......
│  │  │      │      
│  │  │      ├─single_epoch000(☆☆☆存放中间结果，如生成的heatmap之类的)
│  │  │      │  └─gupen
│  │  │      │          1.2.156.112605.215507183267565.210112031150.4.9460.116412.jpg_gt-pred.png
│  │  │      │          1.2.156.112605.215507183267565.210112031150.4.9460.116412.jpg_gt.npy
│  │  │      │          1.2.156.112605.215507183267565.210112031150.4.9460.116412.jpg_gt.png
│  │  │      │          1.2.156.112605.215507183267565.210112031150.4.9460.116412.jpg_input.npy
│  │  │      │          
│  │  │      └─train_epoch099(☆☆☆存放训练的中间结果，如生成的heatmap之类的)
│  │  │          └─gupen
│  │  │                  1.3.46.670589.26.902153.4.20180821.102032.657106.0.jpg_gt-pred.png
│  │  │                  1.3.46.670589.26.902153.4.20180821.102032.657106.0.jpg_gt.npy
│  │  │                  1.3.46.670589.26.902153.4.20180821.102032.657106.0.jpg_gt.png
│  │  │                  1.3.46.670589.26.902153.4.20180821.102032.657106.0.jpg_input.npy               
│  │          
│  └─universal_landmark_detection
│      │  config.yaml（★★☆一些次要配置，比如关键点数量，图片resize大小,数据集路径等等）
│      │  evaluation.py（★★★用于生成带关键点的结果图，以及预测关键点坐标，计算CE角等所有角度指标并保存）
│      │  main.py（★★★★★配置主要参数，如使用的模型、权重文件、是要训练还是测试等等。也是程序入口，训练、测试时运行这│      │			个文件）
│      │  
│      ├─.eval（★★★存放预测结果，包括带关键点的图、各个角度值、预测关键点坐标。还存了预测表现的评价）
│      │  └─.._runs_GU2Net_runs_results_single_epoch000
│      │      │  distance.yaml（☆☆☆跟金标准比，预测表现的评价）
│      │      │  summary.yaml（☆☆☆跟金标准比，预测表现的评价）
│      │      │  
│      │      └─gupen
│      │          ├─gt_laels
│      │          │      
│      │          ├─images（★★★存放生成的计算出的各个角度值（.txt）和结果图，结果图上有关键点的那种）
│      │          │      1.2.156.112605.215507183267565.210112031150.4.9460.116412.png
│      │          │      1.2.156.112605.215507183267565.210112031150.4.9460.116412.txt
│      │          │      
│      │          └─labels（★☆☆存放生成的预测关键点坐标）
│      │                  1.2.156.112605.215507183267565.210112031150.4.9460.116412.jpg.txt
│      │                  
│      └─model
│          │  runner.py（★★☆它负责把模型跑起来！）
│          │  
│          ├─datasets
│          │  │  gupen.py（★★★gupen数据集的读取，包括图片和json格式的关键点坐标；训练、验证和测试的比例分割）
│          │  │  data_pre_loc.py（☆☆☆，读取关键点坐标的工具类）
│          │  │  __init__.py
│          │          
│          ├─networks（☆☆☆存放u-net和GU2Net两个模型）
│          │  │  gln.py
│          │  │  gln2.py
│          │  │  globalNet.py
│          │  │  loss_and_optim.py
│          │  │  u2net.py
│          │  │  unet2d.py
│          │          
│          ├─utils（☆☆☆一些模型工具类）             
```

## 基本用法

以flask server运行app.py后，本地浏览器访问http://127.0.0.1:5000

![](data/flask.png)

## 数据集

数据集存放目录如下：

```
├─UALD
│  │          
│  ├─data
│  │  ├─gupen
│  │  │  ├─labels
│  │  │  │      1.3.12.2.1107.5.3.58.40252.12.202101010955490437.json
│  │  │  │      1.3.12.2.1107.5.3.58.40252.12.202101011540210171.json
│  │  │  │      ......
│  │  │  │      
│  │  │  └─pngs
│  │  │          1.3.12.2.1107.5.3.58.40252.12.202101010955490437.jpg
│  │  │          1.3.12.2.1107.5.3.58.40252.12.202101011540210171.jpg
│  │  │          ......
```

## 训练

### 1.训练和验证数据集分割（可选）

默认数据集比例训练：验证：测试 = 8：1：1。可根据数据集情况自行确定并修改gupen.py 42、43行代码<br>

gupen.py文件位置为：

```
├─UALD 
│  └─universal_landmark_detection             
│      └─model
│          ├─datasets
│          		└─gupen.py
```

```python
        n = len(files)
        train_num = round(n*0.8)
        val_num = round(n*0.1)
        test_num = n - train_num - val_num
```

### 2.设置关键点数量（可选）

```
├─UALD    
│  └─universal_landmark_detection
│      └─config.yaml
```

```yaml
dataset:
  gupen:
    prefix: '../data/gupen'
    num_landmark: 14
    size: [ 512, 512 ] # resize
    
gupen_net:
  in_channels: 1
  out_channels: 14
```

### 3.参数设置（必选）

默认为预测单张图片模式【“single”】。训练时应修改main.py中启动参数-p 的默认值为【“train”】

main.py文件位置为：

```
├─UALD                     
│  └─universal_landmark_detection
│      └─main.py
```



```python
parser.add_argument("-p", "--phase", choices=['train', 'validate', 'test', 'single'], default='single') 
# 训练时修改为default='train'
# single代表专用于前端的单张图片预测模式，生成的图片上不包含金标准关键点
# test可以同时测试多张图片（默认为数据集中最后10%的图片），且生成的图片包含金标准关键点
```

### 4.模型选择（可选）

本关键点检测算法提供两种模型：U-Net model和GU2Net model。GU2Net model表现较好。<br>

使用不同模型训练时应修改main.py中参数

- #### 用U-Net model训练

```python
parser.add_argument("-m", "--model", type=str, default="unet2d")
parser.add_argument("-l", "--localNet", type=str)
```

- #### 用GU2Net model训练

```python
parser.add_argument("-m", "--model", type=str, default="gln")
parser.add_argument("-l", "--localNet", type=str, default="u2net")
```

### 5.从0开始或载入权重训练（必选）

- #### 从0开始训练

```python
parser.add_argument("-c", "--checkpoint", help='checkpoint path')
```

- #### 载入权重训练

```python
parser.add_argument("-c", "--checkpoint", help='checkpoint path',
                        default='../runs/GU2Net_runs/checkpoints/best_GU2Net_runs_epoch098_train5234.624233_val1716.078491.pt')
```

### 6.开始训练

**直接运行main.py文件开始训练!**

文件位置如下

```
├─UALD                     
│  └─universal_landmark_detection
│      └─main.py
```

也可使用命令行开始训练

首先进入指定目录下：

```shell
cd UALD/universal_landmark_detection
```

训练U-Net model:

```shell
python main.py -d ../runs -r unet2d_runs -p train -m unet2d -e 100
```

训练GU2Net model:

```python
python main.py -d ../runs -r GU2Net_runs -p train -m gln -l u2net -e 100
```

加载checkpoint（权重）进行训练：

```shell
python3 main.py -d ../runs -r GU2Net_runs -p train -m gln -l u2net -e 100 -c ../runs/GU2Net_runs/checkpoints/best_GU2Net_runs_epoch098_train5234.624233_val1716.078491.pt
```

### 7.训练结果/测试结果

```
├─UALD        
│  ├─runs（存放unet和GU2Net两种模型的训练、验证和测试结果，包括权重文件）
│  │  └─unet2d_runs
│  │  │   
│  │  ├─GU2Net_runs
│  │  │  │  learning_rate.png（☆☆☆）
│  │  │  │  loss.png（☆☆☆）
│  │  │  │  network_graph.txt（☆☆☆）
│  │  │  │  
│  │  │  ├─checkpoints（★☆☆训练生成的GU2Net模型权重文件）
│  │  │  │      best_GU2Net_runs_epoch098_train5234.624233_val1716.078491.pt
│  │  │  │      best_GU2Net_runs_epoch098_train73.344884_val17.906089.pt
│  │  │  │      ......
│  │  │  │      train_val_loss.txt
│  │  │  │      
│  │  │  └─results（☆☆☆，都是一些没什么用的结果，用不到）
│  │  │      ├─loss
│  │  │      │      epoch_098_loss_54.583926.txt
│  │  │      │      epoch_099_loss_17.908456.txt
│  │  │      │      ......
│  │  │      │      
│  │  │      ├─single_epoch000(☆☆☆存放中间结果，如生成的heatmap之类的)
│  │  │      │  └─gupen
│  │  │      │          1.2.156.112605.215507183267565.210112031150.4.9460.116412.jpg_gt-pred.png
│  │  │      │          1.2.156.112605.215507183267565.210112031150.4.9460.116412.jpg_gt.npy
│  │  │      │          1.2.156.112605.215507183267565.210112031150.4.9460.116412.jpg_gt.png
│  │  │      │          1.2.156.112605.215507183267565.210112031150.4.9460.116412.jpg_input.npy
│  │  │      │          
│  │  │      └─train_epoch099(☆☆☆存放训练的中间结果，如生成的heatmap之类的)
│  │  │          └─gupen
│  │  │                  1.3.46.670589.26.902153.4.20180821.102032.657106.0.jpg_gt-pred.png
│  │  │                  1.3.46.670589.26.902153.4.20180821.102032.657106.0.jpg_gt.npy
│  │  │                  1.3.46.670589.26.902153.4.20180821.102032.657106.0.jpg_gt.png
│  │  │                  1.3.46.670589.26.902153.4.20180821.102032.657106.0.jpg_input.npy 
```

## 测试

训练完成之后（默认100轮），会**自动**进行测试

也可以**手动**测试

修改main.py里的参数，然后运行main.py

```python
parser.add_argument("-c", "--checkpoint", help='checkpoint path',                        default='../runs/GU2Net_runs/checkpoints/best_GU2Net_runs_epoch098_train5234.624233_val1716.078491.pt')

parser.add_argument("-p", "--phase", choices=['train', 'validate', 'test', 'single'], default='test') 
# 训练时修改为default='test'
# single代表专用于前端的单张图片预测模式，生成的图片上不包含金标准关键点
# test可以同时测试多张图片（默认为数据集中最后10%的图片），且生成的图片包含金标准关键点
```

或命令行执行：（确保当前目录是`UALD/universal_landmark_detection`，不是就cd到该目录）

```shell
python main.py -d ../runs -r GU2Net_runs -p test -m gln -l u2net -c ../runs/GU2Net_runs/checkpoints/best_GU2Net_runs_epoch098_train5234.624233_val1716.078491.pt
```

测试结果存储位置与训练结果相同

## 验证（生成带有预测点和金标准的结果图）

命令行执行：（确保当前目录是`UALD/universal_landmark_detection`，不是就cd到该目录）

*-s -d是为了生成可视化结果*

```shell
python evaluation.py -i ../runs/GU2Net_runs/results/test_epoch000 -s -d
```

运行结果存放在：

```
├─UALD        
│  └─universal_landmark_detection
│      ├─.eval（★★★存放预测结果，包括带关键点的图、各个角度值、预测关键点坐标。还存了预测表现的评价）
│      │  └─.._runs_GU2Net_runs_results_single_epoch000
│      │      │  distance.yaml（☆☆☆跟金标准比，预测表现的评价）
│      │      │  summary.yaml（☆☆☆跟金标准比，预测表现的评价）
│      │      │  
│      │      └─gupen
│      │          ├─gt_laels
│      │          │      
│      │          ├─images（★★★存放生成的计算出的各个角度值（.txt）和结果图，结果图上有关键点的那种）
│      │          │      1.2.156.112605.215507183267565.210112031150.4.9460.116412.png
│      │          │      1.2.156.112605.215507183267565.210112031150.4.9460.116412.txt
│      │          │      
│      │          └─labels（★☆☆存放生成的预测关键点坐标）
│      │                  1.2.156.112605.215507183267565.210112031150.4.9460.116412.jpg.txt
```

## References

https://github.com/MIRACLE-Center/YOLO_Universal_Anatomical_Landmark_Detection

## Acknowledge:bouquet::bouquet::bouquet:

仅以此项目献给臭宝:penguin:

