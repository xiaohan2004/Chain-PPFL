> 本项目参考并复现自 [Chain-PPFL](https://github.com/ITSEG-MQ/Chain-PPFL) ，原项目由 Yong Li 开发，遵循 MQ License。

# Chain-PPFL: 基于链式SMC的隐私保护联邦学习框架

Chain-PPFL是一个基于链式安全多方计算(SMC)的隐私保护联邦学习框架，旨在解决传统联邦学习中的隐私保护问题。本框架通过创新的链式结构和安全多方计算方法，在保证学习效果的同时增强了数据隐私保护能力。

## 项目特点

- **链式SMC架构**：采用创新的链式结构进行安全多方计算，更好地保护用户隐私
- **多种隐私保护方案**：支持普通联邦学习、差分隐私(DP)和串行化聚合等多种方案
- **灵活的模型支持**：提供多种深度学习模型，如MLP、CNN和ResNet等
- **多数据集兼容**：支持MNIST、CIFAR等多种数据集
- **丰富的评估指标**：提供了准确率、损失等多种评估指标的可视化工具

## 系统架构

Chain-PPFL框架主要包含以下几个核心组件：

1. **数据分割模块**：支持IID和非IID的数据分布方式
2. **本地训练模块**：负责在客户端进行本地模型训练
3. **安全聚合模块**：使用链式SMC结构进行模型参数的安全聚合
4. **全局模型更新**：服务器端进行全局模型的更新与分发
5. **隐私保护机制**：包括差分隐私、串行化聚合等多种隐私保护方案

## 许可证

本项目基于 Yong Li 发布的开源协议规范文档（遵循 MQ License）进行修改，原始内容版权归其作者所有。

本项目的新增和修改部分由 [xiaohan2004] 于 [2025] 发布，并同样遵循 MQ License。

## 原项目地址

[Chain-PPFL](https://github.com/ITSEG-MQ/Chain-PPFL)
