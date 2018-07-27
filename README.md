v0: 2017年暑假用于Human Activity Detection的版本，最初版。
    主要特征：label读取采用Textgrid，卷积神经网络，不添加任何修改。
    网络返回值：2个，取大者为output label

v1: 在v0的基础上添加如下修改：
    网络最后的输出层添加sigmoid函数，单值输出，表示该帧为语音帧的概率
    从segment中读取label，