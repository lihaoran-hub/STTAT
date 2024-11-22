import dgl
import numpy as np
import torch as th
from dgl.nn.pytorch import GATConv

 # Case 1: Homogeneous graph
g = dgl.graph(([0, 1, 2, 3, 2, 5], [1, 2, 3, 4, 0, 3]))
g = dgl.add_self_loop(g)
feat = th.ones(6, 10)
gatconv = GATConv(10, 2, num_heads=3)
res = gatconv(g, feat)
print(res)
"""
tensor([[[3.4570, 1.8634],
         [1.3805, -0.0762],
         [1.0390, -1.1479]],
        [[3.4570, 1.8634],
         [1.3805, -0.0762],
         [1.0390, -1.1479]],
        [[3.4570, 1.8634],
         [1.3805, -0.0762],
         [1.0390, -1.1479]],
        [[3.4570, 1.8634],
         [1.3805, -0.0762],
         [1.0390, -1.1479]],
        [[3.4570, 1.8634],
         [1.3805, -0.0762],
         [1.0390, -1.1479]],
        [[3.4570, 1.8634],
         [1.3805, -0.0762],
         [1.0390, -1.1479]]], grad_fn= < BinaryReduceBackward >)
"""

 # Case 2: Unidirectional bipartite graph
u = [0, 1, 0, 0, 1]
v = [0, 1, 2, 3, 2]
g = dgl.heterograph({('A', 'r', 'B'): (u, v)})
u_feat = th.tensor(np.random.rand(2, 5).astype(np.float32))
v_feat = th.tensor(np.random.rand(4, 10).astype(np.float32))
gatconv = GATConv((5, 10), 2, 3)
res = gatconv(g, (u_feat, v_feat))
res
"""
tensor([[[-0.6066, 1.0268],
         [-0.5945, -0.4801],
         [0.1594, 0.3825]],
        [[0.0268, 1.0783],
         [0.5041, -1.3025],
         [0.6568, 0.7048]],
        [[-0.2688, 1.0543],
         [-0.0315, -0.9016],
         [0.3943, 0.5347]],
        [[-0.6066, 1.0268],
         [-0.5945, -0.4801],
         [0.1594, 0.3825]]], grad_fn= < BinaryReduceBackward >)
"""