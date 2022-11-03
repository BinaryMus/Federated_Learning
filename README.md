# Federated_Learning

- [Federated_Learning](#federated_learning)
  - [背景](#背景)
  - [参数说明](#参数说明)
  - [示例](#示例)
  - [拓展](#拓展)
    - [数据集拓展](#数据集拓展)
    - [模型拓展](#模型拓展)
    - [优化器拓展](#优化器拓展)
    - [算法拓展](#算法拓展)

## 背景

同步的、以FedAVG为基础的、可拓展的联邦学习框架，支持以下方式的联邦学习实验

1. 主进程做服务器开启多个子进程作为客户端
2. 开启多个进程分别做服务器和客户端（多进程）
3. 多台机器分别做服务器和客户端（集群）

通信基于python的内置socket模块（目前存在潜在bug）

核心内容位于./federated/中，目录结构：

```
├───core
│   │   utils.py
│   │
│   ├───clients
│   │   └───client.py
│   │
│   ├───configs
│   │   └───config.py
│   │
│   ├───monitors
│   │
│   ├───server
│   │   └───server.py
│   │   
│   └───trainers
│       └───mp_trainer.py
│
├───datasets
│   │   cifar10.py
│   │   dataset.py
│   └───mnist.py
│
└───models
    │   resnet.py
    │   simpleCNN.py
    └───vgg.py
```

core下utils包含一些工具函数（设置随机数种子以及清空模型参数）
core下clients和server模块中的client.py和server.py完成了FedAVG算法，拓展其余算法的Client和Server需要继承其中的BaseClient和BaseServer
core下的config包含了配置文件的解析以及多进程和集群环境下的运行
core下的trainers包含主进程开启多个子进程的内容
core下的monitors主要用于记录训练过程、以及训练结束后的绘图等工作（后续添加）

datasets下的dataset.py中的Data是所有数据的基类，添加的数据集都要继承这个类，它可以使用狄利克雷分布切割数据集，默认添加了MNIST数据集和CIFAR10数据集

models下主要包括用torch编写的网络，默认添加了一个简单的CNN网络，VGG11和ResNet18

## 参数说明

所有的配置信息使用yaml收集，其中

- 主进程开多个子进程版本
  - server:
    - ip: 服务器的ip
    - port： 服务器的端口号
  - n_clients： 客户端数量
  - client1：
    - ip： 客户端ip
    - port：客户端端口号
  - （其余client信息，从1开始编号）
  - optimizer：
    - type：优化器版本（目前只有SGD）
    - lr：学习率
  - model：模型（支持SimpleCNN，VGG11，ResNet18）
  - data：数据集（MNIST和CIFAR10）
  - batch_size：mini-batch的大小
  - path：数据集的路径
  - alpha：控制数据集切割的NON-IID程度
  - local_epoch：本地迭代次数
  - global_epoch：全局迭代次数
  - algorithm：算法（目前只有FedAVG）
  - device：使用设备（cpu或者cuda）
- 多进程/分布式版本（需要创建多个配置文件）
  - **以下是server和client共有的配置**
  - role：角色（client或者server）
  - global_epoch：全局迭代轮次
  - algorithm：算法（FedAVG）
  - device：cpu或者cuda
  - model：模型（SimpleCNN，VGG11，ResNet18）
  - data：数据集（MNIST,CIFAR10）
  - ip：ip地址
  - port：使用的端口号
  - **以下是server独有的**
  - n_clients：客户端数量
  - bacth_size：mini-batch的大小
  - path：数据集路径
  - alpha：控制数据集NON-IID程度
  - **以下是client独有的**
  - idx：client的编号（从1开始）
  - server_ip：服务器的ip
  - server_port：服务器的端口
  - optimizer：优化器（SGD）
  - lr：学习率
  - local_epoch：本地迭代轮次

## 示例

- 主进程开多个子进程版本

  使用默认的config_mp.yml文件

  ``````
  python main.py --config="config_mp.yml" --seed=42
  ``````

  该文件配置了IID情况下，将MNIST的训练集分到十个客户端进行联邦学习（测试集放在server中），使用了一个简单的CNN网络，
  每个客户端的mini-batch的大小为64，学习率为0.1，采用FedAVG算法。本地多轮迭代次数为3，全局迭代次数为10，
  所有设备使用cpu。

  在设置随机数种子为42的情况下，最佳的Top-1 准确率为98.87%，最佳的Top-5 准确率为100%。
- 多进程版版本

  使用./config_distributed目录下的配置文件，首先打开服务器

  ``````
  python main.py --config="./config_distributed/server.yml" --arch="distributed" --seed=42
  ``````

  再打开三个终端分别输入

  ``````
  python main.py --config="./config_distributed/client1.yml" --arch="distributed" --seed=42
  python main.py --config="./config_distributed/client2.yml" --arch="distributed" --seed=42
  python main.py --config="./config_distributed/client3.yml" --arch="distributed" --seed=42
  ``````

  除了客户端数量改为3，其余配置与开子进程一致。

  在随机数种子为42的情况下，最佳的Top-1 精度为98.75%，最佳的Top-5精度为100%。
- 分布式版本

  开启服务器后，服务器会依据狄利克雷分布切分数据集到./distributed_data目录下，暂时需要手动移动这些数据到对应的client设备上

  并分别在各自的设备上敲入

  ``````
  python main.py --config="./config_distributed/client1.yml" --arch="distributed" --seed=42
  python main.py --config="./config_distributed/client2.yml" --arch="distributed" --seed=42
  python main.py --config="./config_distributed/client3.yml" --arch="distributed" --seed=42
  ``````

  结果与多进程版本一致
  
  在随机数种子为42的情况下，最佳的Top-1 精度为99.00%，最佳的Top-5精度为100%。

## 拓展

主要从从以下四个方向进行拓展

### 数据集拓展

在./federated/datasets下添加要添加的数据集读入代码，需要继承自./federated/datasets/dataset中的Data类。

Data类说明：构造函数传入的参数为绘图路径（该路径用于绘制使用狄利克雷分布切割后的数据集在客户端的分布）
成员变量train_set为训练集，validate_set为验证集（测试集）。成员函数train_loader传入alpha（狄利克雷参数），n_clients（客户端数量），batch_size(切割的每个数据集的batch大小)，flag（是否绘制数据分布图像）最后返回一个包括n_clients数量的DataLoader列表（torch.utils.data.DataLoader），每个DataLoader的数据量以及总数据量。
静态方法split_non_iid使用numpy的狄利克雷分布对数据集进行切分。

添加的新数据可参考mnist.py，继承Data后调用父类构造函数，需要指定train_set和validate_set（类型为torch.utils.data.Dataset），接着调用父类的train_loader成员函数和validate_loader成员函数得到相应内容。

最后在./federated/core/trainers/mp_trainer.py的all_data字典中注册新的数据集并命名，例如"MNIST": Mnist（类名），接着配置文件的data可以改为"MNIST"。

**注**：由于项目结构有些混乱，所以所有用于注册的字典不在同一目录中（不然存在相互引用的关系）

### 模型拓展

在./federated/models下新建模型。完全的torch模型，可以使用torchvision下的models，例如resnet和vgg就是导入的torchvision，也可以自行搭建模型，例如SimpleCNN

最后在./federated/clients/client.py中的all_arch中注册新模型，则可以在配置文件中使用新的模型。

### 优化器拓展

自定义的优化器需要继承torch.optim.Optimizer。定义后在./federated/clients/client.py中注册新的优化器，则可以在配置文件中使用自定义优化器。

### 算法拓展

这里的算法主要指修改聚合算法、损失函数等方向的算法（server端或者client完成）

若需要修改client端，需要继承自./federated/core/clients/client.py下的BaseClient。

BaseClient类说明：成员变量model为模型，optimizer为优化器，ip为自身的IP地址，port为自身和服务器通信使用的端口（这里我使用了固定的端口），server_ip和server_port为服务器ip和端口，criterion为损失函数，data为数据（torch.utils.data.DataLoader），sample_num为样本数量（用于聚合），device为自身使用的设备，lr为学习率，global_epoch为全局迭代次数，local_epoch为本地迭代次数，loss记录了损失的变化清空，model_name为模型名称，optim_name为优化器名称，n_classes为分类的数量。\
成员函数first_pull使用socket拉取第一次服务器的的参数，train函数训练模型一轮，push_pull上传自身参数以及下载服务器集合后的参数。client的行为都被封装在了run函数中，实例化client后调用run函数即可。\
修改client的损失函数：实例化client时传入自定义的criterion（默认是交叉熵，这里不好写入yaml文件中，需要自定义）\
对client上传数据进行压缩：重写push_pull函数对sendall的数据进行压缩（量化、稀疏化、低秩、编码等等。后续会进行一些功能的添加），server端接受也需要写解码的接口。

若需要修改server端，需要继承自./federated/core/server/server.py下的BaseServer。

BaseServer类说明：成员变量ip为自身ip地址，port为自身通信的端口号，global_epoch为全局迭代次数，n_clients为客户端数量，data为自身拥有的测试集（用于对集合后的数据验证），device为自身使用的设备，model为模型，cnt代表每一轮收到的参数数量，server_socket为服务器的套接字，clients_socket为进行TCP连接的客户端套接字列表，para_cache存储每一轮接收到的参数，round记录完成的全局迭代轮次，total为所有client的数据量。\
成员函数first_push为训练前发布服务器的参数，pull为拉取所有client的参数，aggregate为聚合所有的参数，push发布自身参数，validate验证聚合后的参数性能，server的行为被封装进了run函数中。\
修改server的聚合策略：新的server类只需重写aggregate函数，聚合后继续训练也可以在aggregate函数后添加训练的代码。\
对server发布数据进行压缩：在push函数中sendall前添加压缩操作，对于client端也需要写解码的接口。

在重写完client和server后需要./federated/core/trainer/mp_trainers.py中的all_server和all_server注册新的客户端和服务器。
