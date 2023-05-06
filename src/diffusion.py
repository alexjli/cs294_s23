import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dist

from .utils import infer_O

# pylint: disable=no-member

## R for ideal chains

def R_ideal_chain(N, delta, gamma):
    with torch.no_grad():
        C = torch.arange(N).unsqueeze(0).expand(N, -1) / N
        delta_x1 = torch.zeros(N, N)
        delta_x1[:, 0] = delta
        Q = (1 - torch.tril(torch.ones(N,N)))
        R = C - Q + delta_x1

    return gamma * R

def R_inv_ideal_chain(N, delta, gamma):
    with torch.no_grad():
        R_inv = torch.eye(N, N)
        # this is a hacky way of adding in the lower diagonal -1s
        lower_diag = torch.diag_embed(-torch.ones(torch.diag(R_inv, -1).numel()), -1)
        R_inv += lower_diag
        R_inv[0] = 1/(N * delta)

    return R_inv / gamma


## R with controlled radius of gyration R_g

def b_effective(N, a, r, v):
    return 3/N + np.float_power(N, -v) * np.sqrt(np.float_power(N, 2 * (v-1)) * (N*N + 9)- a*a/(r*r))

def R_general(N, a, r, v, eta):
    b = b_effective(N, a, r, v)
    # b = 1 - 1e-9
    assert b < 1, f"b = {b}"

    with torch.no_grad():
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

    return R

def R_inv_general(N, a, r, v, eta):
    b = b_effective(N, a, r, v)
    # b = 1 - 1e-9
    print("b:", b)

    with torch.no_grad():
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

        R_inv = 1/a *  torch.inverse(R_center) @ torch.inverse(R_sum) @ torch.inverse(R_init)

    return R_inv

def cov_inv_general_centered(N, a, r, v, eta):
    b = b_effective(N, a, r, v)
    # b = 1 - 1e-9
    print("b:", b)

    with torch.no_grad():
        Sig_inv = torch.eye(N) * b**2
        Sig_inv[0][0] = 1
        Sig_inv[-1][-1] = 1

        # this is a hacky way of adding in the upper/lower diagonal -1s
        lower_diag = torch.diag_embed(-torch.ones(torch.diag(Sig_inv, -1).numel()) * b, -1)
        upper_diag = torch.diag_embed(-torch.ones(torch.diag(Sig_inv, 1).numel()) * b, 1)
        Sig_inv += lower_diag
        Sig_inv += upper_diag

    return Sig_inv


## Diffusers


class PositiveLinear(nn.Module):
    def forward(self, weight):
        return torch.abs(weight)


class WhitenedDiffuser(nn.Module):
    """ Diffusion in whitened space """
    def _gamma_tilde(self, t):
        return self.l1(t) + self.l3(F.sigmoid(self.l2(self.l1(t))))  # B x 1

    def _gamma(self, t):
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * (
                self._gamma_tilde(t) - self._gamma_tilde(torch.zeros_like(t))
            ) / (
                self._gamma_tilde(torch.ones_like(t)) - self._gamma_tilde(torch.zeros_like(t))
            )
        return gamma  # B x 1

    def snr(self, t):
        return torch.expm1(-self._gamma(t)) + 1 # B x 1

    def snr_derivative(self, t):
        g_t, g_t_grad = torch.func.jvp(self.snr, (t,), (torch.ones_like(t),)) # B x B
        return g_t, g_t_grad  # B x 1, B x 1

    def alpha(self, t):
        return F.sigmoid(-self._gamma(t))  # B x 1

    def coords_to_vecs(self, coords):
        N, CA, C, O = coords[..., 0:1, :], coords[..., 1:2, :], coords[..., 2:3, :], coords[..., 3:, :]
        CA_N = N - CA
        CA_C = C - CA
        CA_O = O - CA
        return torch.cat([CA, CA_N, CA_C, CA_O], dim=-2)

    def forward_diffusion(self, coords, num_samples, ts=None):
        """
        Parameters
        ----------
        coords: torch.Tensor
            one single protein coord set
        """

        # coords: N x 3 x 3
        num_res = coords.shape[0]

        # sample timepoints
        if ts is None:
            ts = self.t_dist.sample((num_samples, 1)).to(coords.device)  # B x 1
        alpha_ts = self.alpha(ts)  # B x 1

        # make number of copies
        coords_copies = coords.unsqueeze(0).expand(*([num_samples] + [-1 for _ in coords.shape]))  # B x N x 3 x 3
        flat_chain_copies = coords_copies.view([num_samples, -1, 3]) # B x (N x 3) x 3
        R = self.get_R(num_res * 3).to(coords.device)
        noise = torch.randn_like(flat_chain_copies) # B x (N x 3) x 3
        noised_flat_chains = torch.sqrt(alpha_ts).unsqueeze(-1) * flat_chain_copies + \
            torch.sqrt(1 - alpha_ts).unsqueeze(-1) * torch.matmul(R.unsqueeze(0), noise) # B x (N x 3) x 3
        noised_sample = noised_flat_chains.view([num_samples, -1, 3, 3])  # B x N x 3 x 3
        return noised_sample, ts, self._gamma(ts) # ts

    def denoising_loss(self, coords, num_samples):
        # coords: N x 4 x 3
        num_res = coords.shape[0]
        is_nan_node = coords.isnan().any(dim=-1).any(dim=-1)  # B x N
        coords[is_nan_node] = 0

        flat_chain = coords.view([-1, 3])  # (N x 4) x 3
        noised_samples, ts, gts = self.forward_diffusion(coords, num_samples)  # B x N x 3 x 3
        # denoised_samples, _ = self.model(noised_samples, gts)  # B x N x 3 x 3, B x N
        denoised_samples, _ = self.model(noised_samples)  # B x N x 3 x 3, B x N
        # print("denoised", denoised_samples)
        snr_ts, tau_ts = self.snr_derivative(ts)  # B x 1
        # print("snr", ts, snr_ts, tau_ts)

        denoised_flat_chains = denoised_samples.view([num_samples, -1, 3])

        R_inv = self.get_R_inv(num_res * 3).to(coords.device)  # (N x 3) x (N x 3)
        dual_space_projector = (R_inv + self.omega * torch.eye(num_res * 3).to(coords.device))  # (N x 3) x (N x 3)
        # print("R", dual_space_projector)
        dual_space_diff = torch.matmul(dual_space_projector.unsqueeze(0),
                                       (denoised_flat_chains - flat_chain.unsqueeze(0))) # B x (N x 3) x 3
        # print("diff", dual_space_diff)
        dual_space_l2 = torch.square(dual_space_diff).view([num_samples, num_res, 3, 3]) # B x N x 3 x 3
        dual_space_l2 = dual_space_l2.sum(-1).sum(-1) # B x N
        print(dual_space_l2.mean())
        # print("l2", dual_space_l2)

        return -torch.mean(0.5 * tau_ts * dual_space_l2), ts

    def denoise(self, coords, steps=1000):
        coords = coords[:, :3]
        def flatten(a):
            return a.view([1, -1, 3])
        def squish(a):
            return a.view([1, -1, 3, 3])

        with torch.no_grad():
            device = self.l1.weight.device
            N = coords.shape[0]
            flat_coords = coords.view([-1, 3])
            R = self.get_R(N * 3).to(device)
            delta_t = 1/steps
            t = torch.tensor([[1.]]).to(device)
            while t > 0:
                alpha_t = self.alpha(t)
                snr_t, tau_t = self.snr_derivative(t)
                beta_t = tau_t * (1 - alpha_t) ** 2 # * -1?

                # drift term
                # a = -0.5 * snr_t * (flat_coords - flatten(self.model(squish(flat_coords), t)[0]) / torch.sqrt(alpha_t))
                a = 0.5 * snr_t (flat_coords - flatten(self.model(squish(flat_coords))[0]) / torch.sqrt(alpha_t))
                a = a * beta_t * delta_t
                print(a)

                # noise term
                b = torch.sqrt(beta_t) * R @ (torch.randn(1, N * 3, 3).to(device) * np.sqrt(delta_t))
                print(b)

                # update
                flat_coords = flat_coords + a + b
                t -= delta_t
                # break

            return infer_O(squish(flat_coords))

    def sample(self, N, steps=1000, return_trajectory=True):
        def flatten(a):
            return a.view([1, -1, 3])
        def squish(a):
            return a.view([1, -1, 3, 3])

        trajectory = []
        one_shots = []
        with torch.no_grad():
            device = self.l1.weight.device
            whitened_coords = torch.randn(1, N * 3, 3).to(device)
            R = self.get_R(N * 3).to(device)
            flat_coords = R @ whitened_coords
            delta_t = 1/steps
            t = torch.tensor([[1.]]).to(device)
            while t > 0:
                trajectory.append(infer_O(squish(flat_coords)))
                alpha_t = self.alpha(t)
                snr_t, tau_t = self.snr_derivative(t)
                beta_t = - tau_t * (1 - alpha_t) ** 2 # * -1?
                print(beta_t)
                # print(alpha_t, beta_t, snr_t, tau_t)

                # drift term
                a = 0.5 * beta_t / (1-alpha_t) * (flatten(self.model(squish(flat_coords), t)[0]) * torch.sqrt(alpha_t) - flat_coords)
                # one_shot = self.model(squish(flat_coords))[0]
                # one_shot = self.model(squish(flat_coords), t)[0]
                # one_shots.append(one_shot)
                # a = -0.5 * snr_t * (1-alpha_t) * (flat_coords - flatten(self.model(squish(flat_coords))[0]) / torch.sqrt(alpha_t))
                a = a * delta_t

                # noise term
                b = torch.sqrt(beta_t) * R @ (torch.randn(1, N * 3, 3).to(device) * np.sqrt(delta_t))

                # update
                flat_coords = flat_coords + a + b
                t -= delta_t


        if return_trajectory:
            return trajectory, one_shots
        else:
            return trajectory[-1], one_shots[0]



class WhitenedDiffuser_IdealChain(WhitenedDiffuser):
    def __init__(self,
                 model,
                 gamma_0=-7,
                 gamma_1=13.5,
                 omega=1,
                 delta=10,
                 r_gamma=1.54  # average C-C bond length?
                 ):
        super().__init__()

        self.model = model

        self.delta = delta
        self.r_gamma = r_gamma
        self.omega = omega

        self.t_dist = dist.Uniform(0, 1)

        ## beta schedule parameters
        self.l1 = nn.Linear(1, 1)
        self.l2 = nn.Linear(1, 1024)
        self.l3 = nn.Linear(1024, 1)
        nn.utils.parametrize.register_parametrization(self.l1, "weight", PositiveLinear())
        nn.utils.parametrize.register_parametrization(self.l1, "bias", PositiveLinear())
        nn.utils.parametrize.register_parametrization(self.l2, "weight", PositiveLinear())
        nn.utils.parametrize.register_parametrization(self.l2, "bias", PositiveLinear())
        nn.utils.parametrize.register_parametrization(self.l3, "weight", PositiveLinear())
        nn.utils.parametrize.register_parametrization(self.l3, "bias", PositiveLinear())

        self.gamma_0 = gamma_0
        self.gamma_1 = gamma_1

    def get_R(self, N):
        return R_ideal_chain(N, self.delta, self.r_gamma)  # N x N

    def get_R_inv(self, N):
        return R_inv_ideal_chain(N, self.delta, self.r_gamma)  # N x N


class WhitenedDiffuser_ConstrainedChain(WhitenedDiffuser):
    """ Diffusion in whitened space """
    def __init__(self,
                 model,
                 gamma_0=-7,
                 gamma_1=13.5,
                 omega=1,
                 # https://pubs.acs.org/doi/10.1021/jp037128y, spherical folded protein
                 a=3,
                 r=1,  # r vs a?
                 v=0.29,
                 eta=0,
                 ):
        super().__init__()

        self.model = model

        # whitening params
        self.a = a
        self.r = r
        self.v = v
        self.eta = eta
        self.omega = omega

        self.t_dist = dist.Uniform(0, 1)

        ## beta schedule parameters
        self.l1 = nn.Linear(1, 1)
        self.l2 = nn.Linear(1, 1024)
        self.l3 = nn.Linear(1024, 1)
        nn.utils.parametrize.register_parametrization(self.l1, "weight", PositiveLinear())
        nn.utils.parametrize.register_parametrization(self.l1, "bias", PositiveLinear())
        nn.utils.parametrize.register_parametrization(self.l2, "weight", PositiveLinear())
        nn.utils.parametrize.register_parametrization(self.l2, "bias", PositiveLinear())
        nn.utils.parametrize.register_parametrization(self.l3, "weight", PositiveLinear())
        nn.utils.parametrize.register_parametrization(self.l3, "bias", PositiveLinear())

        self.gamma_0 = gamma_0
        self.gamma_1 = gamma_1
    def get_R(self, N):
        return R_general(N, self.a, self.r, self.v, self.eta)  # N x N

    def get_R_inv(self, N):
        return R_inv_general(N, self.a, self.r, self.v, self.eta)  # N x N


if __name__ == '__main__':
    # from .utils import infer_O
    diffuser = WhitenedDiffuser_IdealChain(None)
    print(diffuser._gamma(torch.tensor([[0.]])), diffuser._gamma(torch.tensor([[1.]])))
    print(diffuser.alpha(torch.tensor([[0.]])), diffuser.alpha(torch.tensor([[1.]])))
    print(diffuser.snr_derivative(torch.tensor([[0.]])), diffuser.snr_derivative(torch.tensor([[1.]])))
    print(diffuser.snr_derivative(torch.tensor([[0.25]])), diffuser.snr_derivative(torch.tensor([[0.75]])))

    N = 100
    whitened_noise = torch.randn(N, 3)

    R = R_general(N, 1, 1, 0.29, 0)

    chain = torch.matmul(R, whitened_noise)

    import matplotlib.pyplot as plt
    chain = chain.numpy()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(chain[:, 0], chain[:,1], chain[:, 2], marker="^")
    plt.show()
