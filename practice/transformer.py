import torch

import torch.nn.functional as F

# 这个的理解方式就是两个批次，然后三个字，然后每个字用四个数来表示
x = torch.rand(2,3,4)
# print(x)

W_q = torch.rand(4,4)
W_k = torch.rand(4,4)
W_v = torch.rand(4,4)

# Q K V 得到的就是2x3x4
Q = x @ W_q
K = x @ W_k
V = x @ W_v

# print(Q, K, V)
# 这个地方得到的是2x3x3，就是每个向量点乘，结果可以理解为每个字相对三个字的逐个字的相关程度。
scores = Q @ K.transpose(-2, -1)

# print(scores)
# 这个就是所谓的根号dk
scores = scores / (Q.shape[-1] ** 0.5)
# print(scores)

# 所有的值作为 x 然后计算 e 的 x 次幂，然后求和，然后每个 e 的 x 次幂除以这个和。
attn_weights = F.softmax(scores, dim=-1)
# print(attn_weights)

# 结果的维度等同于初始的维度，这样就可以一层层的往上叠了
output = attn_weights @ V
# print(output)
# print(output.shape)

