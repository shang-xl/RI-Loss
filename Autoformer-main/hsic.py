import torch
# def pairwise_distances(x):
#     #x should be two dimensional
#     instances_norm = torch.sum(x**2,-1).reshape((-1,1))
#     return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()
#
# def GaussianKernelMatrix(x, sigma=1):
#     pairwise_distances_ = pairwise_distances(x)
#     return torch.exp(-pairwise_distances_ /sigma)
#
# def HSIC(x, y, s_x=1, s_y=1):
#     m,_ = x.shape #batch size
#     K = GaussianKernelMatrix(x,s_x)
#     L = GaussianKernelMatrix(y,s_y)
#     H = torch.eye(m) - 1.0/m * torch.ones((m,m))
#     H = H.to(x.device)
#     # H = H.double().cuda()
#     HSIC = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)
#     return HSIC

# outputs = torch.randn(32,7).cuda()
# batch_y = torch.normal(1,1,(32,7)).cuda()
#
# print(HSIC(outputs,batch_y))
#
def pairwise_distances(x):

    instances_norm = torch.sum(x**2, dim=-1, keepdim=True)
    return -2 * torch.matmul(x, x.transpose(-2, -1)) + instances_norm + instances_norm.transpose(-2, -1)

def GaussianKernelMatrix(x, sigma=1):

    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ / sigma)

def HSIC(x, y, s_x=1, s_y=1):

    B, L, _ = x.shape
    K = GaussianKernelMatrix(x, s_x)
    L_mat = GaussianKernelMatrix(y, s_y)

    H = torch.eye(L, device=x.device) - 1.0 / L * torch.ones((L, L), device=x.device)

    HKH = torch.matmul(H, torch.matmul(K, H))

    HKH = torch.matmul(H.expand(B, L, L), torch.matmul(K, H.expand(B, L, L)))

    HSIC_vals = torch.einsum("bij,bij->b", L_mat, HKH) / ((L - 1) ** 2)
    return HSIC_vals.sum()
