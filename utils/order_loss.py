import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 这个函数的目的是将真实标签转化为软变迁
# 生成一个定制化的损失函数，它考虑到了类别之间的距离或者差异,有序数回归问题
def true_metric_loss(true, no_of_classes, scale=1):
    # true: 真实的标签向量，其中每个元素代表一个样本的类别标签。
    # no_of_classes: 数据集中总的类别数量。
    # scale: 一个可选的缩放因子，默认值为1，用于调节类别之间差异的影响。
    batch_size = true.size(0)  # 批次中样本的数量。

    true = true.view(batch_size, 1)  # 将真实标签向量转换为(batch_size, 1)的形状。
    # 将真实标签向量转换为LongTensor类型，并在列方向上重复no_of_classes次，形成一个矩阵，然后转换为浮点数。这个矩阵的每一行都是相同的真实标签。
    if not isinstance(true, torch.cuda.LongTensor):
        true = true.long().to(true.device)
    true_labels = true.repeat(1, no_of_classes).float()
    # class_labels = torch.arange(no_of_classes).float().cuda()：生成一个从0到no_of_classes-1的连续整数向量，然后转换为浮点数并移动到CUDA设备上。
    class_labels = torch.arange(no_of_classes, dtype=torch.float32, device=true.device)
    # 计算class_labels向量和true_labels矩阵之间的绝对差值，然后乘以缩放因子scale
    phi = (scale * torch.abs(class_labels - true_labels))
    # 对phi矩阵的每一行进行softmax操作，得到一个概率分布。
    y = nn.Softmax(dim=1)(-phi)  # 用-phi是为了让距离较小（即类别接近真实标签）的类别有较高的概率值。
    return y


def loss_function(output, labels, loss_type="ordered", expt_type=5, scale=2.5):
    """
    支持多种损失函数类型，当前主要使用有序分类损失
    """
    if loss_type == "ordered":
        targets = true_metric_loss(labels, expt_type, scale)
        return torch.sum(-targets * F.log_softmax(output, -1), -1).mean()
    else:
        return F.cross_entropy(output, labels)


def gr_metrics(op, t):
    """
    计算分级精度 GP、召回 GR、F-score FS 和过估计错误率 OE
    op: 预测结果列表
    t: 真实标签列表
    """
    op = np.array(op)  # 确保 op 是 NumPy 数组
    t = np.array(t)    # 确保 t 是 NumPy 数组

    # TP（True Positive）：预测正确的情况
    TP = (op == t).sum()

    # FN（False Negative）：实际为正例，但预测为负例的数量。
    FN = (t > op).sum()

    # FP（False Positive）：实际为负例，但预测为正例的数量。
    FP = (t < op).sum()

    # 避免除零
    GP = TP / (TP + FP + 1e-9)
    GR = TP / (TP + FN + 1e-9)

    FS = 2 * GP * GR / (GP + GR + 1e-9) if (GP + GR) > 0 else 0.0

    # 过估计错误率（OE, Overestimation Error）
    OE = (t - op > 1).sum()  # 计算预测值与真实标签之差大于1的次数
    OE = OE / op.shape[0]    # 过估计错误率

    return GP, GR, FS, OE
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
#
#
# def true_metric_loss(true, no_of_classes, scale=1):
#     batch_size = true.size(0)
#     true = true.view(batch_size, 1)
#
#     # 修复：简化类型判断，确保为长整数
#     if true.dtype != torch.long:
#         true = true.long().to(true.device)  # 强制转换为长整数并保持设备一致
#
#     true_labels = true.repeat(1, no_of_classes).float()
#     class_labels = torch.arange(no_of_classes, dtype=torch.float32, device=true.device)
#     phi = scale * torch.abs(class_labels - true_labels)
#     y = nn.Softmax(dim=1)(-phi)
#     return y
#
# def loss_function(output, labels, loss_type="ordered", expt_type=5, scale=2.5):
#     """支持多种损失函数类型，当前主要使用有序分类损失"""
#     if loss_type == "ordered":
#         # 修复：正确判断标签是否为长整数类型（使用 .dtype 或 isinstance 与 torch.LongTensor）
#         if not isinstance(labels, torch.LongTensor) and labels.dtype != torch.long:
#             labels = labels.long()  # 转换为长整数类型
#         targets = true_metric_loss(labels, expt_type, scale)
#         log_softmax = F.log_softmax(output, dim=-1)
#         return torch.sum(-targets * log_softmax, dim=-1).mean()
#     else:
#         return F.cross_entropy(output, labels)
#
# def gr_metrics(op, t):
#     """
#     计算分级任务评估指标（GP, GR, FS, OE）
#     """
#     op = np.array(op)
#     t = np.array(t)
#     assert op.shape == t.shape, "预测与真实标签形状必须一致"
#
#     # 基础指标计算
#     TP = (op == t).sum()  # 完全正确预测
#     FN = (t > op).sum()  # 低估（预测值 < 真实值）
#     FP = (t < op).sum()  # 高估（预测值 > 真实值）
#
#     # 分级精度、召回率、F1分数
#     GP = TP / (TP + FP + 1e-9)  # 避免除零
#     GR = TP / (TP + FN + 1e-9)
#     FS = 2 * GP * GR / (GP + GR + 1e-9) if (GP + GR) > 0 else 0.0
#
#     # 过估计错误率（差异 > 1 的样本比例）
#     OE = (np.abs(t - op) > 1).sum() / op.size
#
#     return GP, GR, FS, OE