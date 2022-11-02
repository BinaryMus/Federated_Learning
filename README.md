# Federated_Learning

## 1.参数说明

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
    - 这些是共有的
    - role：角色（client或者server）
    - global_epoch：全局迭代轮次
    - algorithm：算法（FedAVG）
    - device：cpu或者cuda
    - model：模型（SimpleCNN，VGG11，ResNet18）
    - data：数据集（MNIST,CIFAR10）
    - 以下是server独有的
    - ip：ip地址
    - port：使用的端口号
    - n_clients：客户端数量
    - bacth_size：mini-batch的大小
    - path：数据集路径
    - alpha：控制数据集NON-IID程度
    - 以下是client独有的
    - idx：client的编号（从1开始）
    - server_ip：服务器的ip
    - server_port：服务器的端口
    - optimizer：优化器（SGD）
    - lr：学习率
    - local_epoch：本地迭代轮次

## 2.示例

- 主进程开多个子进程版本

  使用默认的config_mp.yml文件
  ``````
  python3 main.py --config="config_mp.yml" --seed=42
  ``````
- 多进程版版本

  使用./config_distributed目录下的配置文件，首先打开服务器
  ``````
  python3 main.py --config="./config_distributed/server.yml" --seed=42
  ``````
  再打开三个终端分别输入
  ``````
  python3 main.py --config="./config_distributed/client1.yml" --seed=42
  python3 main.py --config="./config_distributed/client2.yml" --seed=42
  python3 main.py --config="./config_distributed/client3.yml" --seed=42
  ``````
- 分布式版本
  
  开启服务器后，服务器会依据狄利克雷分布切分数据集到./distributed_data目录下，暂时需要手动移动这些数据到对应的client设备上
  
  并分别在各自的设备上敲入
  ``````
  python3 main.py --config="./config_distributed/client1.yml" --seed=42
  python3 main.py --config="./config_distributed/client2.yml" --seed=42
  python3 main.py --config="./config_distributed/client3.yml" --seed=42
  ``````
## 3.拓展

### 3.1数据集拓展

### 3.2模型拓展

### 3.3优化器拓展

### 3.4算法拓展

### 3.5通信压缩拓展
