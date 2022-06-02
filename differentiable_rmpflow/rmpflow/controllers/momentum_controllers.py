from ..rmptree import Rmp
import torch.autograd as autograd
import torch


class NaturalGradientDescentMomentumController(Rmp):
    '''
    A first-order system based on natural gradient descent
    '''
    def __init__(self, G, Phi=None, del_Phi=None, return_natural=True, device=torch.device('cpu')):
        super(NaturalGradientDescentMomentumController, self).__init__(return_natural=return_natural)
        self.G = G
        self.Phi = Phi
        self.del_Phi = del_Phi
        self.device = device

        if self.del_Phi is None:
            assert self.Phi is not None, ValueError('Phi has to be specified if del_Phi isnt available')

            if hasattr(self.Phi, 'grad'):
                print('Using Phi.grad')
                self.del_Phi = self.Phi.grad
            else:
                print('Using numerical del_Phi')
                self.del_Phi = self.find_potential_gradient

    def eval_natural(self, x, xd=None, t=None):
        p = -self.del_Phi(x)
        M = self.G(x)
        return p, M

    def find_potential_gradient(self, x):
        n, d = x.size()
        x.requires_grad_(True)
        Phi = self.Phi(x).reshape(-1,1)

        if Phi.requires_grad:
            mask = torch.ones(1, 1, device=self.device).repeat(n, 1)
            dPhi = autograd.grad(Phi, x, mask, create_graph=True)[0]
            # dPhi = dPhi.detach()
        else:  # if requires grad is False, then output has no dependence of input
            dPhi = torch.zeros(n, d, device=self.device)
        x.requires_grad_(False)
        return dPhi
