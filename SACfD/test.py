from utils import build_sincos_pos_embed
import numpy as np
A = np.array([[1,2,3,4,5],[1,2,3,4,6],[2,4,5,6,2]])
print(build_sincos_pos_embed(6,10))