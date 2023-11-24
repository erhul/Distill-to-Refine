# import numpy as np
# confidence = np.random.rand(10)
# print(confidence)
# print((confidence > 0.5))
# estimated_noise_ratio = (confidence > 0.5).mean()
# print(estimated_noise_ratio)
# clean_labels = np.array([0,1,2,3,4,5,6,7,8,9])
# noise_labels = np.array([1,5,2,3,6,5,6,7,1,9])
# print((clean_labels == noise_labels))
# print((confidence > 0.5) == (clean_labels == noise_labels))
# noise_accuracy = ((confidence > 0.5) == (clean_labels == noise_labels)).mean()
# print(noise_accuracy)

import torch


x = torch.tensor([[2., 3., 5., -1.], [-1., -2., 1., 4.], [0.5, -2., 7., 2.]])


l2_norm_all = torch.norm(x)
print("L2 norm of all elements:", l2_norm_all.item())


l2_norm_rows = torch.norm(x, dim=1)
print("L2 norm of rows:", l2_norm_rows.numpy())


l1_norm_cols = torch.norm(x, p=1, dim=-1)
print("L1 norm of columns:", l1_norm_cols.numpy())