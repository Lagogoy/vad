##  v0: 2017年暑假用于Human Activity Detection的版本，最初版。
    特征为cmn后的 MFCC + delta1 + delta2
    label读取采用Textgrid
    通过feats_io.py存储feats和labels文件，格式为：*.idx(索引)，*.ark(数据)
    卷积神经网络，网络返回值：2分类，取大者为output label


##  v1: 在v0的基础上添加如下修改
    从segment中读取label
    网络最后的输出层添加sigmoid函数，单值输出，表示该帧为语音帧的概率


##  v2：在v0的基础上增加大幅度的修改
    通过prepare_data.py提取特征为±40帧的频谱，合并保存为*.npy文件。
    卷积神经网络，只保留需要的模型，其他删除。网络返回值：2分类。
    网络存在待优化问题，本Version需要改进，如果表现不如v1可以考虑删除
    
##  v3：基于SCD任务的LSTM版本更改
    修改标签计算和网络模型