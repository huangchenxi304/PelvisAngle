先在这10个方法上做
了解一下基于对抗的域对齐是怎么做的
如果最后novelty不够，可以采用A+B结合的方法，就是把不同的LOSS加在一起
只需要考虑一种backbone：1D-CNN（最小最快）
跑出来10个看看哪种效果最好
考虑3种隐私扰动方法：
源域数据-->nonDP-GAN-->生成的源域数据-->DA
源域数据-->DP-GAN-->生成的源域数据-->DA
源域数据-->直接在数据上加噪-->生成的源域数据-->DA

<img src="D:\hcx\FL\code\2023-02-24T22_26_06.png" style="zoom:50%;" />

## 2.25

找了3个DP-GAN的code

1个是不针对TS的： dpwgan

2个是专门针对TS数据的：
1个是tf写的：security-research-differentially-private-generative-models(secureGAN)，[另一个是pytorch写的](https://github.com/paper-code-anon/dp-gan)：dp-gan
看看pytortch的数据集格式长啥样（ROD（类似UCIHAR）和PPG）

### 数据集描述

ROD（occupation detection）：https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+

PPG：https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6971339/

HAR: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

ROD有7个features，HAR有 561个features

`D:\hcx\FL\data\UCI HAR Dataset\train\Inertial Signals`这里放的是传感器9个通道的数据

所以实际上原始数据是128x9=1152个特征

train和test下Inertial Signals路径下的数据，也就是未经特征工程处理的数据

X_train.txt和X_test.txt中的特征是经过[特征工程](https://so.csdn.net/so/search?q=特征工程&spm=1001.2101.3001.7020)得到的，由每个窗口中提取的常用的一些时间和频率特征构成，共561个特征



ROD除去标签列和时间戳、date两列实际特征只有4列，然后实际只用了3列：['Temperature', 'Humidity', 'CO2']



ROD样本数：20560，HAR样本数：10299

ROD是二分类的，PPG和HAR是多分类的



ROD output: [20560/30, 30*3+1label]

HAR output: [10299/9，9*1152+1label]





[ADATIME](https://github.com/emadeldeen24/AdaTime)里的数据集用的public的，但也提供处理后的

```shell
python main.py  --experiment_description DDC_HAR_original --run_description DDC_HAR_original_2 --da_method DDC --is_sweep True --sweep_project_wandb TEST_DDC_HAR_DPGAN_2 --dataset HAR_DPGAN
```



## 2.26

记得交workshop

ADATIME跑通了

有个问题就是超参数搜索AUTOML，github上给的示例是参数`--num_sweeps 50`，也就是50次随机超参数搜索，但1次就要10分钟

所以设为5

下一步是弄通DP-GAN数据集生成



单目视觉深度估计测距：https://zhuanlan.zhihu.com/p/56263560

单目多帧自监督深度估计(2021-2022)研究进展：https://blog.csdn.net/CV_Autobot/article/details/127168596



要把HAR转成ROG格式的

## 2.27

成功在DP-GAN上跑通了HAR，accuracy=88%

如何设置让梯度扰动和输入扰动的隐私水平一样？

下一步：

把生成的数据集在ADATIME上跑一遍

找离散傅里叶扰动的代码并跑通（或者常规Laplace的）；

在non-DP-GAN上跑，0隐私对应的隐私参数是？



## 2.28

生成合成数据时是否处理imbalance labels？

inverse_transform_eps的值？

occupancy_dp_gan_2023-02-28_13_37_13.437962：只是没有处理imbalance labels

occupancy_dp_gan_2023-02-28_13_51_24.640837: inverse_transform_eps的值从100000改为1，不要用了

occupancy_dp_gan_2023-02-28_20_51_54.806392：加了点reverse_scaling,不要用了

occupancy_dp_gan_2023-02-28_21_24_36.654459：scaling设为false

是不是要反scaling?

nz是啥？

02-27的生成的数据非normalized是10299，90的



## 3.2

The emotional TTS system can be built using the following tech stacks:

1. Text classification models: LSTM and Transformer can be used for mapping the translated text to different emotional polarities.
2. Emotional embedding: The sentiment descriptors need to be modeled and embedded as emotional embedding in the speech synthesis model. This can be done using techniques like Word2Vec or GloVe.
3. Speech synthesis models: WaveNet and Tacotron can be used for speech synthesis.
4. Audio processing libraries: Librosa and PyDub can be used for audio processing tasks like converting speech to spectrograms and vice versa.

The general organization of the emotional TTS system can be as follows:

1. The input text is first passed through the text classification model to determine its emotional polarity.
2. The emotional polarity is then used to generate the corresponding emotional embedding.
3. The emotional embedding is combined with the linguistic embedding to generate the final embedding for speech synthesis.
4. The final embedding is then passed through the speech synthesis model to generate the corresponding speech waveform.
5. The speech waveform is then processed using audio processing libraries to apply post-processing effects like pitch modulation, tempo adjustment, etc.



Text-to-Speech (TTS) systems use a combination of different technologies to convert written text into spoken words. The technologies used in TTS systems can vary depending on the specific implementation, but some common components and stacks include:

1. Text pre-processing: This involves cleaning and normalizing the text, including removing punctuation, correcting spelling errors, and handling abbreviations and acronyms.
2. Text-to-Phoneme conversion: This is the process of converting written words into a sequence of phonemes, which are the smallest units of sound that make up a language.
3. Prosody generation: This involves adding intonation, stress, and other aspects of speech to the synthesized speech to make it sound more natural.
4. Speech synthesis: This involves generating the actual speech waveform from the processed text, using techniques such as concatenative synthesis or parametric synthesis.
5. Neural network-based approaches: TTS systems can also use neural networks, such as WaveNet or Tacotron, for speech synthesis. These models are trained on large amounts of speech data to generate speech waveforms that are more natural and human-like.

```lua
+----------------------+        +----------------------+          +------------------------+
|   Text Input (text)  |        |  Phoneme Generation   |   Text   |    Spectrogram (mel)    |
|                      +------->+     (Text-to-Phoneme)  +--------->+       Generation       |
+----------------------+        +----------------------+          +------------------------+
                                              |
                                              | Phoneme sequence
                                              |
                                              v
+----------------------+        +----------------------+          +------------------------+
|    Phoneme Sequence   |        |  Mel-Spectrogram      |  Audio   |       Waveform          |
|                      +------->+  Synthesis           +--------->+     Synthesis (Vocoder)  |
+----------------------+        +----------------------+          +------------------------+

```

1. WaveNet: WaveNet is a deep neural network architecture developed by DeepMind that can generate high-quality speech waveforms. The model takes in a sequence of phoneme or character embeddings as input and outputs a sequence of raw audio samples. The model uses a dilated convolutional architecture, where the receptive field of the convolutional filters grows exponentially with depth, allowing the model to capture long-term dependencies in the input. To generate speech, the model takes a text input and generates a sequence of audio samples, which are then post-processed to remove noise and enhance the quality of the speech signal.
2. Tacotron: Tacotron is another deep neural network architecture for speech synthesis. Unlike WaveNet, Tacotron generates speech by predicting a sequence of spectrogram frames from the input text, and then converting these spectrograms to a waveform using a vocoder such as Griffin-Lim. The model consists of an encoder, which converts the input text into a sequence of hidden states, and a decoder, which predicts a sequence of spectrogram frames from the hidden states. The decoder also incorporates a mechanism for attention, allowing it to focus on different parts of the input text at different times.

### Emotinal TTS

- Data collection and preprocessing: Collect and preprocess speech and text data, and label them with emotional tags.
- Text classification: Train a text classification model (e.g., LSTM, Transformer) to classify the text into different emotional categories.
- Emotional embedding: Embed the emotional tags into a low-dimensional vector space, and concatenate this emotional embedding with the linguistic embedding of the text input. This can be done using techniques like Word2Vec or GloVe.
- Acoustic modeling: Train a TTS model (e.g., WaveNet, Tacotron) to generate speech from the emotional-linguistic embedding.
- Synthesize speech: Use the trained model to synthesize speech with emotional expression.

```lua
+--------------------+      +------------------------+      +--------------------+
| Data collection and |      |                        |      |                    |
| preprocessing      |      | Text classification     |      | Acoustic modeling   |
|                    |      |                        |      |                    |
|                    +----->+                        +----->+                    |
|                    |      | (e.g., LSTM, Transformer)|    |   WaveNet/Tacotron |
|                    |      |                        |      |                    |
+--------------------+      +------------------------+      +--------------------+
                                       |
                                       |
                                       |
                                       v
                            +---------------------------+
                            |                           |
                            | Emotional embedding       |
                            |   Word2Vec/GloVe                        |
                            |                           |
                            +---------------------------+
                                       |
                                       |
                                       |
                                       v
                            +---------------------------+
                            |                           |
                            | Synthesize speech         |
                            |                           |
                            |                           |
                            +---------------------------+

```

## 3.7

### sensing project proposal

#### related work:

##### measure the distance of the hand from the camera：

1. use stereo vision, which involves using two cameras to capture two slightly different views of the same scene, and then using the disparity between the two images to calculate the distance to objects in the scene.
2. use depth cameras or sensors, such as Microsoft's Kinect or Intel's RealSense cameras
3. supervised ML-based method using just a single camera: need a large dataset of hand images with known distances
4. unsupervised DL-based method using just a single camera: applied in auto-driving scenario but not implement in hand case.



总体上，相机成像可以分为四个步骤：刚体变换、透视投影、畸变校正和数字化图像。

![b376460f1517ccc24084848128675a5c](D:\download\b376460f1517ccc24084848128675a5c.png)

<img src="C:\Users\huang chenxi\AppData\Roaming\Typora\typora-user-images\image-20230307164907355.png" alt="image-20230307164907355" style="zoom:50%;" />



##### [distortion in hand disatnce measurement](https://www.pianshen.com/article/64851854382/):

理想的针孔成像模型确定的坐标变换关系均为线性的，而实际上，现实中使用的相机由于镜头中镜片因为光线的通过产生的不规则的折射，镜头畸变（lens distortion）总是存在的，即根据理想针孔成像模型计算出来的像点坐标与实际坐标存在偏差。

畸变导致的成像失真可分为*径向失真和切向失真*两类。径向扭曲导致直线呈现弯曲,离图像中心越远，径向畸变就越大。例如，下面示出一个图像。

![img](https://img-service.csdnimg.cn/img_convert/37e5af7715d6c732d9100861520b7347.png)



径向畸变校正Radial distortion correction can be represented as follows:

![Image for post](https://img-service.csdnimg.cn/img_convert/f8680b50ffe786f2e560ffb0e7f36d95.png)

类似地，发生切向失真是因为图像拍摄镜头未完全平行于成像平面对齐。

![img](https://img-service.csdnimg.cn/img_convert/b8d569b06e095b08171cc12955a86df7.png)

因此，图像中的某些区域可能看起来比预期的更近。切向畸变量校正可表示如下Similarly, tangential distortion occurs because the image-taking lense is not aligned perfectly parallel to the imaging plane. So, some areas in the image may look nearer than expected. The amount of tangential distortion can be represented as below:

![Image for post](https://img-service.csdnimg.cn/img_convert/d911819b0da254fbaf537b5c33827fcc.png)

##### [correction of distortion](https://blog.csdn.net/weixin_26752765/article/details/108132288)

使用traditional camera calibration and undistortion的缺点：

1. Limited field of view: The camera calibration assumes a specific field of view and distortion model, and can only undistort images within that field of view. If the hand moves outside of this area, the undistortion may not be accurate.
2. Non-rigid hand movement: The hand is a non-rigid object, which means that its shape and size can change dynamically. The camera calibration assumes a rigid calibration object (e.g., a chessboard), and may not be able to accurately capture the hand's changing shape and size.
3. User variability: The accuracy of the measurement may also depend on the user's ability to place their hand in the correct position relative to the camera. If the user's hand is not in the correct position, the measurement may be inaccurate.

calibration and undistortion is a traditional method for camera calibration and has been widely used for decades.

but it only corrects radial distortion.

Tangential distortion can be corrected by affine transformations only if it is small. However, if the tangential distortion is significant, it cannot be corrected by affine transformations alone.

Therefore, some more sophisticated camera calibration technique that estimates both radial and tangential distortion coefficients were developed, such as 

1. Zhang's Camera Calibration Algorithm
2. Bouguet's Camera Calibration Toolbox
3. Tsai's Camera Calibration Algorithm

However, they require a significant amount of calibration data and may not be suitable for real-time applications or situations where the camera and objects are in motion.

Recently, many monocular multi-frame self-supervised depth estimation methods have emerged, such as

MonoDepth(CVPR 2017)

ManyDepth(CVPR2021)

DepthFormer(CVPR2022)

DynamicDepth(ECCV2022)

MonoDepth(CVPR 2017): 

1. Encode the input single frame image into depth through encoder+decoder.
2. The input image is obtained through the encoder to obtain the external parameter matrix between them.
3. Through the external parameter matrix and depth, we can warp the image at the moment to the moment, get the predicted image at the moment, and then calculate the loss with the real image at the moment to complete the self-supervised training.

<img src="C:\Users\huang chenxi\AppData\Roaming\Typora\typora-user-images\image-20230307150735484.png" alt="image-20230307150735484" style="zoom: 33%;" />

ManyDepth(CVPR2021): Extend the input of a single frame image to a cost volume constructed from multiple frames of images as input

<img src="C:\Users\huang chenxi\AppData\Roaming\Typora\typora-user-images\image-20230307150854614.png" alt="image-20230307150854614" style="zoom:33%;" />

DepthFormer(CVPR2022): Use cross attention(Transformers) to build cost volume

<img src="C:\Users\huang chenxi\AppData\Roaming\Typora\typora-user-images\image-20230307151034932.png" alt="image-20230307151034932" style="zoom:33%;" />

DynamicDepth(ECCV2022): Dealing with dynamic scenes

<img src="C:\Users\huang chenxi\AppData\Roaming\Typora\typora-user-images\image-20230307151134833.png" alt="image-20230307151134833" style="zoom: 50%;" />

#### datasets

use self-collected dataset. Use the laptop's built-in camera to collect data sets, collect 2 frames of images per second and save them

#### Proposed technical approach

<img src="C:\Users\huang chenxi\AppData\Roaming\Typora\typora-user-images\image-20230307162734757.png" alt="image-20230307162734757" style="zoom:50%;" />

#### Experimental design

benchmark: MonoDepth, traditional camera calibration

metrics: three most common metrics in depth estimation task: abs. rel. error(absolute relative error), RMSE and MAE

### 3.10

#### presentation

`in terms of depth estimation, generally, there are 4 ways measuring the distance of the hand from the camera. 考虑到我们的应用需要尽量轻量的设备，我们考虑使用一个单独摄像头时的优化。我们的主要对象——手——是不规则的、可自由旋转和平移的物体。因此我们需要处理lens distortion的对成像的影响。传统的camera calibration要么不能同时处理Radial distortion and tangential distortion，要么会在同时处理时 require a significant amount of calibration data，且它无法处理objects are in motion。此外，收集大量手部到摄像头距离的带标签数据是困难的。综合以上考虑和动作的流畅性需要，我们考虑采用Monocular self-supervised machine learning-based method, which is applied in auto-driving scenario but not implement in hand case.`



`我们调查了近年在Monocular multi-frame self-supervised depth estimation的研究，结果表明我们的想法是可行的。`



`so this is our depth estimator. In this estimator, we have 2 trainable networks:  a depth network and a pose network. More specifically, our depth network includes three components: a feature extractor, an encoder, and a depth decoder. 相邻帧和当前帧送入posenet提取位姿，最后将warped的特征和当前帧的特征进行cost volume构建，送入网络输出frame depth`



Regarding depth estimation, there are generally four ways to measure the hand-to-camera distance. 

As our application requires lightweight device, we optimize for using just a single camera. 

Since hands are irregular objects that can freely rotate and move, we need to address the impact of lens distortion on imaging. Traditional camera calibration methods// either cannot handle both radial and tangential distortion simultaneously// or require a significant amount of calibration data, and it cannot handle objects in motion. 

Additionally, it is challenging// to collect a large amount of labeled data on hand-to-camera distance. 

Considering these limitations and the need for smooth motion, we propose using a monocular self-supervised machine learning-based method, which has been applied in auto-driving scenarios// but not yet implemented for hand distance measurement.

NEXT



we investigate recent research in Monocular multi-frame self-supervised depth estimation. and it shows that our idea is feasible.

We listed some of the important algorithms here.

NEXT



This is our depth estimator, which consists of two trainable networks: a depth network and a pose network. 

The depth network has three components: a feature extractor, an encoder, and a depth decoder. 

The pose network extracts pose// from adjacent frames and the current frame, and the warped features and current features// are used to construct a cost volume for the network to output frame depth.

NEXT



### 3.14

#### 找一下GAN-TS（不带DP）

1. Multivariate Time Series Imputation with Generative Adversarial Networks, NIPS 2018

   code: [GitHub - Luoyonghong/Multivariate-Time-Series-Imputation-with-Generative-Adversarial-Networks: NIPS2018 paper](https://github.com/Luoyonghong/Multivariate-Time-Series-Imputation-with-Generative-Adversarial-Networks)

2. Time-series Generative Adversarial Networks, NIPS 2019

   code: https://github.com/jsyoon0823/TimeGAN

   pytorch_code: https://github.com/zzw-zwzhang/TimeGAN-pytorch

3. Generative Adversarial Networks in Time Series: A Systematic Literature Review, ACM Computing Surveys

4. TTS-GAN: A Transformer-based Time-Series Generative Adversarial Network, AIME 2022: https://github.com/imics-lab/tts-gan

5. Generative Adversarial Network to create synthetic time series: https://github.com/mirkosavasta/GANetano

6. Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs: https://github.com/ratschlab/RGAN

### 3.16

secureGAN的bug解决了，但是有新bug，只能处理catergorical的数据集，而且可能只能处理二分类？（原数据集用的是adult）

dpwgan也是只能处理catergorical的数据集



### 3.20

table t_order：
pattype字段：
i是住院：inpatient
e是急诊 emergency
o是门诊 outpatient
n是住院急诊

里主键：accno


table x_order里

pattype：h是急诊
主键：examnum

happentime：开单时间

需求：
分别查2021 2022的ct门诊，ct急诊，mr门诊，mr急诊（4个组合）（modality=ct或mr）（急诊也包括住院急诊），examdt减去happentime的结果，分别算一整年平均的值
以及2021到2022的增长率



我有一张表：t_order，它包括以下字段：pattype, accno, examdt, modality. pattype 代表 department type, 有i, e, o, n四个值，分别代表住院、急诊、门诊、住院急诊。主键是accno. examdt代表检查时间。 modality代表检查类型，有ct和mr两种值。

我还有一张表： x_order. 它包括以下字段：pattype, examnum, happentime, modality. pattype 代表 department type, 有i, h, o, n四个值，分别代表住院、急诊、门诊、住院急诊。modality代表检查类型，有ct和mr两种值。主键是examnum。happentime代表检查单开具时间。

现在我需要你写sql语句完成以下任务：

分别计算2021年和2022年的ct门诊，ct急诊，mr门诊，mr急诊（4个组合）（modality=ct或mr）（急诊也包括住院急诊），examdt减去happentime的结果，分别给出一整年平均的值，并给出2021年到2022年的增长率。请使用两张表的主键进行对齐

你的结果应该是如下形式：

4行3列，其中每列字段名分别是：2021，2022，2021年到2022年的增长率. 每行字段代表ct门诊，ct急诊，mr门诊，mr急诊。

optional tips: 2021年的检查你可以用sql语句：happentime > '2021-01-01' and t.happentime < '2021-12-31'

2022年的检查你可以用sql语句：happentime > '2022-01-01' and t.happentime < '2022-12-31'



I have a table: t_order, which includes the following fields: pattype, accno, examdt, modality. pattype represents department type, and has four values i, e, o, n, which represent inpatient, emergency, outpatient, and inpatient emergency. The primary key is accno. examdt stands for exam time. Modality represents the inspection type, and has two values: ct and mr.

I also have a table: x_order. It includes the following fields: pattype, examnum, happentime, modality. pattype represents department type, and has four values of i, h, o, n, which represent inpatient, emergency, outpatient, and inpatient emergency. Modality represents the inspection type, and has two values: ct and mr. The primary key is examnum. happentime represents the time when the checklist is issued.

Now I need you to write sql statements to complete the following tasks:

In 2021 and 2022, calculate the results of CT inpatient, CT outpatient, MR inpatient, and MR outpatient(4 combinations) (modality=CT or MR)， examdt minus happentime. Then give the average values for the whole year in 2021 and 2022 respectively, and give the growth rate from 2021 to 2022. Please use the primary keys of the two tables for alignment

Your result should be of the form:

4 rows and 3 columns, where the field names of each column are: 2021, 2022, and the growth rate from 2021 to 2022. The fields in each row represent ct outpatient, ct emergency, mr outpatient, and mr emergency.

Optional tips: You can use sql statement to check in 2021: happentime > '2021-01-01' and t.happentime < '2021-12-31'

For 2022 check you can use sql statement: happentime > '2022-01-01' and t.happentime < '2022-12-31'



```sql
SELECT 
    AVG((t.examdt - x.happentime) * 24 * 60 * 60) AS "2021", 
    AVG((t.examdt - x.happentime) * 24 * 60 * 60) AS "2022", 
    AVG((t.examdt - x.happentime) * 24 * 60 * 60 * 100 / 
         NULLIF((t.examdt - x.happentime) * 24 * 60 * 60, 0) - 100) AS "growth_rate"
FROM 
    t_order t 
    JOIN x_order x ON t.pattype = x.pattype AND t.modality = x.modality 
WHERE 
    (t.modality = 'CT' OR t.modality = 'MR') 
    AND (t.pattype = 'o' OR t.pattype = 'i' OR x.pattype = 'o' OR x.pattype = 'i') 
    AND (
        (x.happentime >= '2021-01-01 00:00:00' AND x.happentime < '2022-01-01 00:00:00') 
        OR (x.happentime >= '2022-01-01 00:00:00' AND x.happentime < '2023-01-01 00:00:00')
    );
```





i have to point out that, i want the 4 combinations: CT outpatient, CT emergency, MR outpatient, and MR emergency. Given the case that emergency also includes inpatient emergency, in terms of the 'pattype', they should be: 
1. t.pattype = 'o'
2. t.pattype = 'e' AND t.pattype = 'n'
3. x.pattype = 'o'
4. x.pattype = 'h' AND x.pattype = 'n'





```sql

```



```sql

```





```sql
SELECT 
  AVG((t.examdt - x.happentime) * 24 * 60 * 60) FILTER (WHERE t.modality = 'CT' AND t.pattype = 'o') AS "CT_outpatient_2021", 
  AVG((t.examdt - x.happentime) * 24 * 60 * 60) FILTER (WHERE t.modality = 'CT' AND t.pattype = 'e' AND t.pattype = 'n') AS "CT_emergency_2021", 
  AVG((t.examdt - x.happentime) * 24 * 60 * 60) FILTER (WHERE t.modality = 'MR' AND t.pattype = 'o') AS "MR_outpatient_2021", 
  AVG((t.examdt - x.happentime) * 24 * 60 * 60) FILTER (WHERE t.modality = 'MR' AND x.pattype = 'h' AND x.pattype = 'n') AS "MR_emergency_2021", 
  AVG((t.examdt - x.happentime) * 24 * 60 * 60) FILTER (WHERE t.modality = 'CT' AND t.pattype = 'o') AS "CT_outpatient_2022", 
  AVG((t.examdt - x.happentime) * 24 * 60 * 60) FILTER (WHERE t.modality = 'CT' AND t.pattype = 'e' AND t.pattype = 'n') AS "CT_emergency_2022", 
  AVG((t.examdt - x.happentime) * 24 * 60 * 60) FILTER (WHERE t.modality = 'MR' AND t.pattype = 'o') AS "MR_outpatient_2022", 
  AVG((t.examdt - x.happentime) * 24 * 60 * 60) FILTER (WHERE t.modality = 'MR' AND x.pattype = 'h' AND x.pattype = 'n') AS "MR_emergency_2022", 
  AVG((t.examdt - x.happentime) * 24 * 60 * 60 * 100 / NULLIF((t.examdt - x.happentime) * 24 * 60 * 60, 0) - 100) AS "growth_rate" 
FROM 
  t_order t 
  JOIN x_order x ON t.pattype = x.pattype AND t.modality = x.modality 
WHERE 
  (t.modality = 'CT' OR t.modality = 'MR') AND 
  (
    (t.pattype = 'o' AND x.pattype IS NULL) OR 
    (t.pattype = 'e' AND t.pattype = 'n' AND x.pattype IS NULL) OR 
    (x.pattype = 'o' AND t.pattype IS NULL) OR 
    (x.pattype = 'h' AND x.pattype = 'n' AND t.pattype IS NULL)
  ) AND 
  (
    (x.happentime >= '2021-01-01 00:00:00' AND x.happentime < '2022-01-01 00:00:00') OR 
    (x.happentime >= '2022-01-01 00:00:00' AND x.happentime < '2023-01-01 00:00:00')
  );
```



this is a sql statement. can you help reformat it into sql format? SELECT AVG(CASE WHEN t.modality = 'CT' AND t.pattype = 'o' THEN (t.examdt - x.happentime) * 24 * 60 * 60 ELSE NULL END) AS "CT_outpatient_2021", AVG(CASE WHEN t.modality = 'CT' AND t.pattype = 'o' THEN (t.examdt - x.happentime) * 24 * 60 * 60 ELSE NULL END) AS "CT_outpatient_2022", AVG(CASE WHEN t.modality = 'CT' AND t.pattype = 'o' THEN (t.examdt - x.happentime) * 24 * 60 * 60 * 100 / NULLIF((t.examdt - x.happentime) * 24 * 60 * 60, 0) - 100 ELSE NULL END) AS "CT_outpatient_growth_rate", AVG(CASE WHEN t.modality = 'CT' AND t.pattype = 'n' AND t.pattype = 'e' THEN (t.examdt - x.happentime) * 24 * 60 * 60 ELSE NULL END) AS "CT_emergency_2021", AVG(CASE WHEN t.modality = 'CT' AND t.pattype = 'n' AND t.pattype = 'e' THEN (t.examdt - x.happentime) * 24 * 60 * 60 ELSE NULL END) AS "CT_emergency_2022", AVG(CASE WHEN t.modality = 'CT' AND t.pattype = 'n' AND t.pattype = 'e' THEN (t.examdt - x.happentime) * 24 * 60 * 60 * 100 / NULLIF((t.examdt - x.happentime) * 24 * 60 * 60, 0) - 100 ELSE NULL END) AS "CT_emergency_growth_rate", AVG(CASE WHEN t.modality = 'MR' AND t.pattype = 'o' THEN (t.examdt - x.happentime) * 24 * 60 * 60 ELSE NULL END) AS "MR_outpatient_2021", AVG(CASE WHEN t.modality = 'MR' AND t.pattype = 'o' THEN (t.examdt - x.happentime) * 24 * 60 * 60 ELSE NULL END) AS "MR_outpatient_2022", AVG(CASE WHEN t.modality = 'MR' AND t.pattype = 'o' THEN (t.examdt - x.happentime) * 24 * 60 * 60 * 100 / NULLIF((t.examdt - x.happentime) * 24 * 60 * 60, 0) - 100 ELSE NULL END) AS "MR_outpatient_growth_rate", AVG(CASE WHEN t.modality = 'MR' AND x.pattype = 'h' AND x.pattype = 'n' THEN (t.examdt - x.happentime) * 24 * 60 * 60

```sql
SELECT 
  AVG(CASE WHEN t.pattype = 'i' AND t.modality = 'CT' THEN EXTRACT(epoch FROM (t.examdt - x.happentime))) AS ct_inpatient_2021,
  AVG(CASE WHEN t.pattype = 'i' AND t.modality = 'MR' THEN EXTRACT(epoch FROM (t.examdt - x.happentime))) AS mr_inpatient_2021,
  AVG(CASE WHEN t.pattype = 'o' AND t.modality = 'CT' THEN EXTRACT(epoch FROM (t.examdt - x.happentime))) AS ct_outpatient_2021,
  AVG(CASE WHEN t.pattype = 'o' AND t.modality = 'MR' THEN EXTRACT(epoch FROM (t.examdt - x.happentime))) AS mr_outpatient_2021,
  AVG(CASE WHEN t.pattype = 'i' AND t.modality = 'CT' THEN EXTRACT(epoch FROM (t.examdt - x.happentime))) AS ct_inpatient_2022,
  AVG(CASE WHEN t.pattype = 'i' AND t.modality = 'MR' THEN EXTRACT(epoch FROM (t.examdt - x.happentime))) AS mr_inpatient_2022,
  AVG(CASE WHEN t.pattype = 'o' AND t.modality = 'CT' THEN EXTRACT(epoch FROM (t.examdt - x.happentime))) AS ct_outpatient_2022,
  AVG(CASE WHEN t.pattype = 'o' AND t.modality = 'MR' THEN EXTRACT(epoch FROM (t.examdt - x.happentime))) AS mr_outpatient_2022,
  AVG(CASE WHEN t.pattype = 'i' AND t.modality = 'CT' THEN EXTRACT(epoch FROM (t.examdt - x.happentime))) / AVG(CASE WHEN t.pattype = 'i' AND t.modality = 'CT' THEN EXTRACT(epoch FROM (t.examdt - x.happentime))) * 100 AS ct_inpatient_growth_rate,
  AVG(CASE WHEN t.pattype = 'i' AND t.modality = 'MR' THEN EXTRACT(epoch FROM (t.examdt - x.happentime))) / AVG(CASE WHEN t.pattype = 'i' AND t.modality = 'MR' THEN EXTRACT(epoch FROM (t.examdt - x.happentime))) * 100 AS mr_inpatient_growth_rate,
  AVG(CASE WHEN t.pattype = 'o' AND t.modality = 'CT' THEN EXTRACT(epoch FROM (t.examdt - x.happentime))) / AVG(CASE WHEN t.pattype = 'o' AND t.modality = 'CT' THEN EXTRACT(epoch FROM (t.examdt - x.happentime))) * 100 AS ct_outpatient_growth_rate,
  AVG(CASE WHEN t.pattype = 'o' AND t.modality = 'MR' THEN EXTRACT(epoch FROM (t.examdt - x.happentime))) / AVG(CASE WHEN t.pattype = 'o' AND t.modality = 'MR' THEN EXTRACT(epoch FROM (t.examdt - x.happentime))) * 100 AS mr_outpatient_growth_rate
FROM 
  t_order t
  JOIN x_order x ON t.accno = x.examnum
WHERE 
  x.happentime >= '2021-01-01' AND x.h
```



### 3.21

先训练source domain 的分类器

用encoder得到source domain的特征。

这些特征拿来做dpgan（和label一起），因为是9x128，不要flatten，是一个矩阵形式，不然就丢失关联了。

dpgan生成的特征，直接丢进分类器，（不要训练了！！！），看看分类结果咋样



这样比直接在TS数据集上DP-GAN，既没有因为flatten丢失关联信息，也可以以后应用到不同类型数据集



backbone network就是feature extractor

src_x 是个（32，9，128）的tensor

src_fea是个（32，128）的tensor

why 32? 

32是batch_size：单次传递给程序用以训练的数据(样本)个数

label呢？

src_y 是个 （32，）的tensor

