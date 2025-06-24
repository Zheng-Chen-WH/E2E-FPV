import torch

# 在这里定义你的张量并测试 einsum 表达式
 
# 第一题
a = torch.randn(3)
b = torch.randn(3)
# 错误：result = torch.einsum('i,i->1', a, b)
# 正确：'：i,i->', i 是共享的索引，它在输入中出现两次，但在输出中没有出现，所以沿着 i 进行乘积和求和。

# 第二题
a=torch.randn(5,3)
b=torch.randn(3)
result = torch.einsum('ab,b->a', a, b)

# 第三题
a=torch.randn(5,3)
b=torch.randn(3,6)
result = torch.einsum('ab,bc->ac', a, b)

# 第四题
a=torch.randn(5)
b=torch.randn(3)
result = torch.einsum('a,b->ab', a, b)

# 第五题
a=torch.randn(5,3)
result = torch.einsum('ab->ba', a)

# 第六题 求迹？
a=torch.randn(5,5)
# 正确：'ii->',索引 i 在输入矩阵 A 的标签中出现两次 (ii)，表示我们只关心对角线元素 (A[i,i])。因为它没有出现在输出中，所以这些对角线元素会被加起来。

# 第七题
a=torch.randn(3,4,5)
b=torch.randn(3,5,6)
result = torch.einsum('abc,acd->abd', a, b)

# 第八题
a=torch.randn(3,4,5)
b=torch.randn(4,6)
result = torch.einsum('abc,bd->acd', a, b)

# 第九题
a=torch.randn(3,4,5)
b=torch.randn(3,4,5)
result = torch.einsum('abc,abc->1', a, b)
# 正确：'xyz,xyz->' 所有索引 x,y,z 在两个输入中都匹配，并且它们都没有出现在输出中。这意味着对应元素相乘后，对所有维度进行求和。
