import numpy as np
import torch

### Case 1: chain constraint
# intuitively, we can think of z_i's as vectors that point
# from one monomer in the chain to the next
# which makes tilde x_i the total offset of a monomer from the origin
# then, tilde x_i - \sum_k (tilde x_k) / N represents the total offset from center of mass
# and delta x_i designates the offset of the center of mass from the origin

# We can write the matrix R as
N = 4
delta = 1e-9
# gamma = 1

C = torch.arange(N).unsqueeze(0).expand(N, -1) / N
delta_x1 = torch.zeros(N, N)
delta_x1[:, 0] = delta
Q = (1 - torch.tril(torch.ones(N,N)))

R = C - Q + delta_x1
# essentially, (C-Q) represents the chain constraint
# while delta_x1 represents the offset
# it's not entirely clear to me why this has to happen at x1
# but i also think this isn't super important
print("R:")
print(R)


## Computing R_inv
# intuitively, the reverse process is simply computing the distance between any two adjacent
# monomers to recover z_i
# the way they write it in chroma seems complicated but essentially much of the equation
# cancels out, but they seem to write it with these constants since it keeps the equation "clean"
# e.g. x_1's position is affected by 1/delta \sum_k x_k / N
# but nothing else depends on this offset since all the distances build off of x_1 iteratively

# The inverse R is now

R_inv = torch.eye(N, N)
# this is a hacky way of adding in the lower diagonal -1s
lower_diag = torch.diag_embed(-torch.ones(torch.diag(R_inv, -1).numel()), -1)
R_inv += lower_diag
if delta == 0:
    R_inv[0] = 1
else:
    R_inv[0] = 1/(N * delta)


# essentially, the first row represents the offset
# while the rest are simply the difference between two adjacent units
print("R_inv:")
print(R_inv)

# as a sanity check we multiply the two together
print("R @ R_inv")
print(R @ R_inv)
print("R_inv @ R")
print(R_inv @ R)



# now if we compute the covariance matrix from these we get
inv_cov_compute = R_inv.T @ R_inv
print("Computed inverse covariance matrix:")
print(inv_cov_compute)

# and based on their formula we're supposed to get
tmp = 2 * torch.eye(N, N)
tmp[0][0] = 1
tmp[-1][-1] = 1
upper_diag = torch.diag_embed(-torch.ones(torch.diag(tmp, 1).numel()), 1)
lower_diag = torch.diag_embed(-torch.ones(torch.diag(tmp, -1).numel()), -1)
tmp += upper_diag
tmp += lower_diag
print(tmp)

inv_cov_formula = tmp + 1/(N * delta) ** 2 * torch.ones(N, N)
print("Formula inverse covariance matrix:")
print(inv_cov_formula)

### Case 2: Arbitrary R_g scaling
eta = 0.1
a = 1
b = 1 - 1e-9

R_center = torch.eye(N) - eta / N

R_sum = torch.zeros(N, N) - 1
for i in range(N):
    if i == 0:
        R_sum[i] = torch.arange(N)
    else:
        R_sum[i][i:] = torch.arange(N)[:-i]

R_sum = R_sum.T
mask = (R_sum == -1)
R_sum = b ** R_sum
R_sum[mask] = 0

R_init = torch.eye(N)
R_init[0][0] = 1/np.sqrt(1-b**2)

R = a *  R_center @ R_sum @ R_init

print("R:")
print(R)
print(torch.inverse(R))
print(torch.inverse(R_center))
