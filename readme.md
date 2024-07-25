# 一.简介

该项目为一次大模型+游戏的尝试，成功实现了一个与原神内角色的对话系统。

演示视频：https://www.bilibili.com/video/BV1b48CeuELL

模型下载链接：

链接：https://删pan.bai掉du.com/中s/1q文n8g7dEljWLPKFuGCoV61A?p再wd=yo用vn 
提取码：yovn 
--来自百度网盘超级会员V5的分享

* 整个项目分为三部分：
  * 声音读取
  * 大模型对话：我将在后面介绍如何基于中文预训练模型Atom训练自己的大模型
  * 文本输出：利用BertVITS2训练需要的角色的声音，并输出，完成对话
* 为了能够成功跑通项目，需要下载一系列模型，它们分别是：
  * bert：直接放在根目录下即可
  * Atom：这个建议根据自己下载，因为还会有其他各种各样的东西
  * 语音模型：放在./CapsWriterOffline/models下即可
  * 模型下载地址：

# 二.声音读取

## 2.1项目介绍

* 原项目github地址：[HaujetZhao/CapsWriter-Offline: CapsWriter 的离线版，一个好用的 PC 端的语音输入工具 (github.com)](https://github.com/HaujetZhao/CapsWriter-Offline)

* B站视频教程：https://www.bilibili.com/video/BV1tt4y1d75s

* 这个项目已经打包好了exe执行文件，使用起来很方便，但是为了构建对话系统，在使用前自己需要下载好两个模型：
  * paraformer：这个模型我是用作者给的百度网盘链接下载的
  * punc_ct-transformer：这个虽然网盘链接也有，但是我实际运行起来会缺文件（config.yaml和tokens.json），最后选择使用git下载，成功解决

```
标点模型：
git clone https://www.modelscope.cn/damo/punc_ct-transformer_cn-en-common-vocab471067-large-onnx.git punc_ct-transformer_cn-en
```

## 2.2接口调整

* 为了能构建一个完整的系统，我根据源代码设置的了一个接口。

* 分析：如果不使用作者打包好的exe执行文件，想要启动程序分别需要运行./CapsWriterOffline下的start_client.py和start_server.py，其中前者是控制麦克风输入的，后者是加载模型的。

* 接口：为了将麦克风采集到的输入转化成文字并返回到程序，我修改了client端的代码并集成到了根目录下的genshin_talk.py中，因此在后续使用中只需要同时启动根目录下的**genshin_talk.py**和**start_server.py**即可

# 三.大模型对话

其实我一开始只是想学习大模型怎么用，最后的对话系统属于是为这叠醋包的饺子，在这一部分我讲介绍怎么训练一个自己喜欢角色的大模型。

## 3.1数据

在根目录下有个LLM的文件夹，可以用来训练自己的大模型，首先你需要在LLM/data下创建一个自己的数据集.json文件，形式如下：

![image-20240721161150036](C:\Users\lcz\AppData\Roaming\Typora\typora-user-images\image-20240721161150036.png)

刻晴的数据集是我手打的，主要来源为(这个自己打字太花时间了，我就只做了100条数据，请大家根据自己的需要制作自己的角色的数据集，也可以根据自己的理解增加数据)：

* 原神里刻晴的好感度对话
* 原神里刻晴的主线剧情：https://www.bilibili.com/video/BV1hr4y1F7H

## 3.2开始训练

* 前往LLM文件夹下的config.yaml修改LLM_dataset_path，修改为你自己创建的数据集的路径
* LLM/config.yaml里面还有一个LoRA_save_path，这是你自己利用大模型微调后的LoRA信息，也可以修改为自己喜欢的路径
* 运行keqing.py开始训练就可以了
  * 在训练前程序会自动从huggingface上面下载Atom模型，这是一个适用于中文的、对标llama2的大模型。然而国内是不允许访问huggingface的，所以在第一次运行这部分程序时需要用魔法。模型很大，我下载完13个G左右。默认存储地址为：C:\Users\用户名\\\.cache\huggingface\hub
  * 关于数据集的一点坑：这个数据集第一次处理完后，后续你去更改原始文件而不更改文件名，它也不会再去处理文件了。如果你希望更改数据集，需要先去C:\Users\用户名\\\.cache\huggingface\datasets\generator下把之前缓存的文件夹先删除了
  * 大模型的训练效果根据数据集的丰富程度来决定，如果你对于这段对话有自己的情感需求，你在制作数据集的时候可以有意识地制作相关的数据（例如价值观、人生观）

# 四.文本输出

## 4.1项目介绍

为了增强对话的体验感，选择使用其他大佬制作的BertVITS2模型训练对应角色的声音。

项目原版视频：[BV1hp4y1K78E](https://www.bilibili.com/video/BV1hp4y1K78E/?spm_id_from=333.788.video.desc.click)

项目原版github地址：https://github.com/Stardust-minus/Bert-VITS2

BertVITS2-2.3整合版教程：https://www.bilibili.com/video/BV13p4y1d7v9

在本项目中，进行声音训练的时候，2.3整合版的教程也完全使用。

如果你是想要训练原神、崩铁、鸣潮相关角色的声音，那么可以前往[主页 | AI-Hobbyist 资源网盘](https://pan.ai-hobbyist.com/)下载，我访问这个网站也需要用魔法。

## 4.2接口调整

* 当你训练好模型后，你的模型文件会被保存在根目录下Data\项目名\models\下，以G开头的.pth中，选择一个模型并保存好它的路径，前往webui.py中修改这一行代码：

```python
net_g = get_net_g(model_path="你刚才保存好的模型路径", version="2.3", device="cuda:0", hps=hps)
```

* 当你在训练这个项目时肯定会尝试测试，这个时候你会看见一系列的参数。你在网页上看见的参数，可以在对应的./LLM/config.yaml下设置，至少要将speaker换成你自己当初创建的名字。

# 五.总结

* 运行根目录下的genshin_talk.py和start_server.py，等待两边把模型加载好就行了，然后按下Caps Locks进行对话即可
* 有任何问题欢迎联系我，邮箱：2744334724@qq.com  QQ：2744334724
* 如果可以的话请给我一个star，这对我非常重要