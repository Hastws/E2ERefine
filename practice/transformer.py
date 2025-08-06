import torch
import torch.nn as nn
import torch.nn.functional as F


# # 这个的理解方式就是两个批次，然后三个字，然后每个字用四个数来表示
# x = torch.rand(2,3,4)
# # print(x)
#
# W_q = torch.rand(4,4)
# W_k = torch.rand(4,4)
# W_v = torch.rand(4,4)
#
# # Q K V 得到的就是2x3x4
# Q = x @ W_q
# K = x @ W_k
# V = x @ W_v
#
# # print(Q, K, V)
# # 这个地方得到的是2x3x3，就是每个向量点乘，结果可以理解为每个字相对三个字的逐个字的相关程度。
# scores = Q @ K.transpose(-2, -1)
#
# # print(scores)
# # 这个就是所谓的根号dk
# scores = scores / (Q.shape[-1] ** 0.5)
# # print(scores)
#
# # 所有的值作为 x 然后计算 e 的 x 次幂，然后求和，然后每个 e 的 x 次幂除以这个和。
# attn_weights = F.softmax(scores, dim=-1)
# # print(attn_weights)
#
# # 结果的维度等同于初始的维度，这样就可以一层层的往上叠了
# output = attn_weights @ V
# # print(output)
# # print(output.shape)
#
# def get_position_encoding(n_position, d_hidden):
#     # 生成一个列向量，但是扩展了维度，使其可以被广播
#     position_encoding = torch.arange(n_position, dtype=torch.float32).unsqueeze(1)
#     print(position_encoding)
#     # 换底公式得到的次公式 torch.log 只能传 tensor 作为参数 得到一个行向量
#     # 先算 2i 序列，然后用这个序列来乘以定值即可。
#     div_term = torch.exp(torch.arange(0,d_hidden,2).float() * (-torch.log(torch.tensor(10000.0)) / d_hidden))
#     print(div_term)
#     pe = torch.zeros(n_position, d_hidden).float()
#     print(pe)
#     # 然后基于广播的机制，将 position_encoding 中的 squeeze 1 给广播填充，变成 position 乘以 hidden 的一半
#     # 然后交替使用 sin cos 填充得到结果
#     pe[:, 0::2] = torch.sin(position_encoding * div_term)
#     pe[:, 1::2] = torch.cos(position_encoding * div_term)
#     print(pe)
#     return pe
#
# get_position_encoding(3,6)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model, bias=False)

    # def forward(self, x):
    #     batch_size, seq_len, _ = x.size()
    #
    #     Q = self.W_q(x)
    #     K = self.W_k(x)
    #     V = self.W_v(x)
    #
    #     Q = Q.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
    #     K = K.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
    #     V = V.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
    #
    #     scores = Q @ K.transpose(-2, -1)/(self.d_head ** 0.5)
    #     attn_weights = F.softmax(scores, dim=-1)
    #     attn_output = attn_weights @ V
    #
    #     attn_output = attn_output.transpose(1, 2).contiguous()
    #     attn_output = attn_output.view(batch_size, seq_len, self.d_model)
    #
    #     output = self.W_o(attn_output)
    #
    #     return output

    def forward(self, x, mask):
        batch_size, seq_len, _ = x.size()

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1)/(self.d_head ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = attn_weights @ V

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        output = self.W_o(attn_output)

        return output

def generate_subsequent_mask(size):
    mask = torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)
    return mask

mask = generate_subsequent_mask(4)
print(mask)

# # batch size，seq_len, d_model
# # 多头注意力机制对除了d_model有影响以外，其他的没有影响，然后影响完d_model还需要再拼接回去
# x = torch.randn(3, 4, 8)
# mha = MultiHeadAttention(8, 2)
# # mask的宽度必须是seq长度
# out = mha(x, mask)
# print(out.size())
#
# print(out)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        ff_output = self.ffn(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x

# 这代表是2个batch，4个seq length，8个d_model， 其中mask的宽度必须等于4，因为需要对齐seq length
x = torch.randn(2,4,8)
# d_model 等于8，然后头是2个，然后是mlp是32.
encoder_layer = EncoderLayer(8,2,32)
out = encoder_layer(x, mask)
print(out)