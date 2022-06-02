from ..rmptree import Rmp
import torch
import torch.autograd as autograd


class NaturalGradientDescentForceController(Rmp):
    def __init__(self, G, B, Phi=None, del_Phi=None, ds_type='gds', return_natural=True, device=torch.device('cpu')):
        '''
        Geometric dynamical system (GDS)
        :param G: Riemennaian metric (D x D array function )
        :param B: Damping matrix (D x D array function)
        :param Phi: Potential function (1x1 array function)
        :param del_Phi: Derivative Potential function (Dx1 array function)
        :param Xi: Curvature mass (DxD array function)
        :param xi: Curvature force (Dx1 array function)
        :param ds_type: Dynamical system type ('gds' (default) or 'lds')
        '''

        super(NaturalGradientDescentForceController, self).__init__(return_natural=return_natural)

        self.G = G
        self.B = B
        self.Phi = Phi
        self.del_Phi = del_Phi
        self.ds_type = ds_type
        self.device = device

        if self.ds_type is not 'gds':
            raise NotImplementedError

        if type(self.B) == torch.Tensor:  # 2-D tensor
            self.damping_force = lambda x, xd: torch.matmul(xd, self.B.T)
        elif type(self.B) == float:  # scalar
            self.damping_force = lambda x, xd: self.B*xd
        else:                   # 3-D tensor function
            self.damping_force = lambda x, xd: torch.einsum('bij,bj->bi', self.B(x, xd), xd)

        if self.del_Phi is None:
            assert self.Phi is not None, ValueError('Phi has to be specified if del_Phi isnt available')

            if hasattr(self.Phi, 'grad'):
                print('Using Phi.grad')
                self.del_Phi = self.Phi.grad
            else:
                print('Using numerical del_Phi')
                self.del_Phi = self.find_potential_gradient

    def eval_natural(self, x, xd, t=None):
        del_Phi = self.del_Phi(x)
        # M = self.G(x,xd)
        G, Xi, xi = self.find_curvature_terms(x, xd)
        M = G + Xi
        f = -del_Phi - xi - self.damping_force(x, xd)
        # f = -del_Phi - self.damping_force(x, xd)
        return f, M

    def find_potential_gradient(self, x):
        n, d = x.size()
        x.requires_grad_(True)
        Phi = self.Phi(x).reshape(-1, 1)

        if Phi.requires_grad:
            mask = torch.ones(1, 1, device=self.device).repeat(n, 1)
            dPhi = autograd.grad(Phi, x, mask, create_graph=True)[0]
        else:  # if requires grad is False, then output has no dependence of input
            dPhi = torch.zeros(n, d, device=self.device)

        # TODO: Not sure if setting require_grad to False for input effects the graph
        x.requires_grad_(False)
        # torchviz.make_dot(dPhi).view()
        return dPhi

    def find_curvature_terms(self, x, xd):
        n, d = x.size()
        x_m = x.repeat(d ** 2, 1)
        x_m.requires_grad_(True)

        xd_m = xd.repeat(d ** 2, 1)
        xd_m.requires_grad_(True)

        G_m = self.G(x_m, xd_m)
        G = G_m[:n]

        if G_m.requires_grad:
            gmf = G_m.reshape(-1, d ** 2)
            mask = torch.eye(d ** 2, device=self.device).repeat_interleave(n, dim=0)

            # Finding curvature mass
            dgdxd, = autograd.grad(gmf, xd_m, mask, create_graph=True, allow_unused=True, retain_graph=True)
            if dgdxd is None:  # if theres no dependence on xd
                Xi = torch.zeros(n, d, d, device=self.device)
            else:
                # dgdxd = dgdxd.detach().reshape(d, d, n, d)
                dgdxd = dgdxd.reshape(d, d, n, d)
                dgdxd = torch.einsum('ijkl->kijl', dgdxd)
                Xi = torch.einsum('bikj, bk->bij', dgdxd, xd)

            # Finding curvature force
            dgdx, = autograd.grad(gmf, x_m, mask, create_graph=True, allow_unused=True)
            if dgdx is None:
                xi = torch.zeros_like(x, device=self.device)
            else:
                # dgdx = dgdx.detach().reshape(d, d, n, d)
                dgdx = dgdx.reshape(d, d, n, d)
                dgdx = torch.einsum('ijkl->kijl', dgdx)
                cvt1 = torch.einsum('bijk,bj,bk->bi', dgdx, xd, xd)
                cvt2 = torch.einsum('bjki,bj,bk->bi', dgdx, xd, xd)
                xi = cvt1 - 0.5 * cvt2
        else:
            Xi = torch.zeros(n, d, d, device=self.device)
            xi = torch.zeros_like(x, device=self.device)

        return G, Xi, xi


# -----------------------------------------------
# class LagrangianDynamicalSystem(Rmp):
#     def __init__(self, n_dim, G, B, Phi=None, del_Phi=None):
#         '''
#         Geometric dynamical system (GDS)
#         :param dof: no of dims of the configuration space
#         :param G: Riemennaian metric (D x D array function )
#         :param B: Damping matrix (D x D array function)
#         :param Phi: Potential function (1x1 array function)
#         :param del_Phi: Derivative Potential function (Dx1 array function)
#         :param Xi: Curvature mass (DxD array function)
#         :param xi: Curvature force (Dx1 array function)
#         '''
#
#         super(LagrangianDynamicalSystem, self).__init__()
#
#         self.n_dim = n_dim
#         self.G = G
#         self.B = B
#         self.Phi = Phi
#         self.del_Phi = del_Phi
#
#         if type(self.B) == torch.Tensor:  # 2-D tensor
#             self.damping_force = lambda x, xd: torch.matmul(xd, self.B.T)
#         elif type(self.B) == float:  # scalar
#             self.damping_force = lambda x, xd: self.B*xd
#         else:                   # 3-D tensor function
#             self.damping_force = lambda x, xd: torch.einsum('bij,bj->bi', self.B(x, xd), xd)
#
#         if self.del_Phi is None:
#             print('Using numerical del_Phi')
#             assert self.Phi is not None, ValueError('Phi has to be specified if del_Phi isnt available')
#             self.del_Phi = self.find_potential_gradient
#
#     def forward(self, t, x, xd):
#         del_Phi = self.del_Phi(x)
#         G = self.G(x, xd)
#         Xi, xi = self.find_curvature_terms(x, xd)
#
#         M = G + Xi
#         f = -del_Phi - xi - self.damping_force(x, xd)
#         return f, M
#
#     def find_potential_gradient(self, x):
#         n = x.size()[0]
#         x.requires_grad_(True)
#         Phi = self.Phi(x).reshape(-1,1)
#
#         if Phi.requires_grad:
#             mask = torch.ones(1, 1).repeat(n, 1)
#             dPhi = autograd.grad(Phi, x, mask, create_graph=True)[0]
#             dPhi = dPhi.detach()
#         else:  # if requires grad is False, then output has no dependence of input
#             dPhi = torch.zeros(n, self.n_dim)
#         x.requires_grad_(False)
#         return dPhi
#
#     def find_curvature_terms(self, x, xd):
#         n, d = x.size()
#         x_m = x.repeat(self.n_dim ** 2, 1)
#         x_m.requires_grad_(True)
#
#         xd_m = xd.repeat(self.n_dim ** 2, 1)
#         xd_m.requires_grad_(True)
#
#         G_m = self.G(x_m, xd_m)
#
#         if G_m.requires_grad:
#             gmf = G_m.reshape(-1, d ** 2)
#             mask = torch.eye(d ** 2,).repeat_interleave(n, dim=0)
#
#             # Finding curvature mass
#             dgdxd, = autograd.grad(gmf, xd_m, mask, create_graph=True, allow_unused=True)
#             if dgdxd is None:  # if theres no dependence on xd
#                 Xi = torch.zeros(n, self.n_dim, self.n_dim)
#             else:
#                 dgdxd = dgdxd.detach().reshape(d, d, n, d)
#                 dgdxd = torch.einsum('ijkl->kijl', dgdxd)
#                 Xi = torch.einsum('bikj, bk->bij', dgdxd, xd)
#
#                 Xi = torch.einsum('bljk, bj->bkl', dgdxd, xd) \
#                      + torch.einsum('bkjl, bj->bkl', dgdxd, xd)
#
#             # Finding curvature force
#             dgdx, = autograd.grad(gmf, x_m, mask, create_graph=True, allow_unused=True)
#             if dgdx is None:
#                 xi = torch.zeros_like(x)
#             else:
#                 dgdx = dgdx.detach().reshape(d, d, n, d)
#                 dgdx = torch.einsum('ijkl->kijl', dgdx)
#                 cvt1 = torch.einsum('bijk,bj,bk->bi', dgdx, xd, xd)
#                 cvt2 = torch.einsum('bjki,bj,bk->bi', dgdx, xd, xd)
#                 xi = cvt1 - 0.5 * cvt2
#         else:
#             Xi = torch.zeros(n, self.n_dim, self.n_dim)
#             xi = torch.zeros_like(x)
#
#         return Xi, xi






# -----------------------------------------------
if __name__ == '__main__':
    d = 2
    G = lambda x, xd: torch.einsum('bi,bj->bij', x, x*xd)
    B = torch.zeros(d, d)
    Phi = lambda x: (0.5*torch.norm(x, dim=1)**2).reshape(-1, 1)
    gds = NaturalGradientDescentForceController(G=G, B=B, Phi=Phi)
    x = torch.ones(2,d)
    xd = torch.ones(2,d)
    f, M = gds(x, xd)