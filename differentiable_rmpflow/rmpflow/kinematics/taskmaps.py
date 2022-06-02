import torch
import torch.nn as nn
import torch.autograd as autograd


class TaskMap(nn.Module):
    def __init__(self, n_inputs, n_outputs, psi, J=None, J_dot=None, device=torch.device('cpu')):
        super(TaskMap, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.psi = psi
        self.J_ = J
        self.J_dot_ = J_dot
        self.device = device

    def J(self, x):
        if self.J_ is not None:
            J = self.J_(x)
            if not self.training:
                J = J.detach()
                return J

        n = x.size()[0]
        x_m = x.repeat(1, self.n_outputs).view(-1, self.n_inputs)
        x_m.requires_grad_(True)
        y_m = self.psi(x_m)
        if y_m.requires_grad:
            mask = torch.eye(self.n_outputs, device=self.device).repeat(n, 1)
            J = autograd.grad(y_m, x_m, mask, create_graph=True, allow_unused=True)[0]
            if J is None:
                J = torch.zeros(n, self.n_outputs, self.n_inputs, device=self.device)
            else:
                J = J.reshape(n, self.n_outputs, self.n_inputs)
        else:  # if requires grad is False, then output has no dependence of input
            J = torch.zeros(n, self.n_outputs, self.n_inputs, device=self.device)

        if not self.training:
            J = J.detach()

        return J

    def J_dot(self, x, xd):
        if self.J_dot_ is not None:
            Jd = self.J_dot_(x, xd)
            return Jd

        if self.J_ is not None:
            n = x.size()[0]
            x_m = x.repeat(1, self.n_outputs * self.n_inputs).view(-1, self.n_inputs)
            x_m.requires_grad_(True)
            J_m = self.J_(x_m)

            if J_m.requires_grad:
                J_m_vec = J_m.reshape(-1, self.n_outputs * self.n_inputs)
                mask = torch.eye(self.n_outputs * self.n_inputs, device=self.device).repeat(n, 1)
                dJ_m_vec, = autograd.grad(J_m_vec, x_m, mask, create_graph=True, allow_unused=True, retain_graph=True)
                dJ_m_vec = dJ_m_vec
                dJ_m_xd_vec = dJ_m_vec * xd.repeat(1, self.n_outputs * self.n_inputs).view(-1, self.n_inputs)
                dJ_m_xd = dJ_m_xd_vec.sum(dim=1)
                Jd = dJ_m_xd.reshape(-1, self.n_outputs, self.n_inputs)
            else:
                Jd = torch.zeros(n, self.n_outputs, self.n_inputs, device=self.device)
        else:
            n = x.size()[0]
            x_m = x.repeat(1, self.n_outputs).view(-1, self.n_inputs)
            x_m.requires_grad_(True)
            y_m = self.psi(x_m)

            if y_m.requires_grad:
                mask = torch.eye(self.n_outputs, device=self.device).repeat(n, 1)
                J = autograd.grad(y_m, x_m, mask, create_graph=True, allow_unused=True)[0]
                if J is None:
                    J = torch.zeros(n, self.n_outputs, self.n_inputs, device=self.device)
                else:
                    J = J.reshape(n, self.n_outputs, self.n_inputs)
            else:  # if requires grad is False, then output has no dependence of input
                J = torch.zeros(n, self.n_outputs, self.n_inputs, device=self.device)

            Jd = torch.zeros(n, self.n_outputs, self.n_inputs, device=self.device)
            if J.requires_grad:  # if requires grad is False, then J has no dependence of input
                # Finding jacobian of each column and applying chain rule
                for i in range(self.n_inputs):
                    Ji = J[:, :, i]
                    mask = torch.eye(self.n_outputs, self.n_inputs, device=self.device).repeat(n, 1)
                    Ji_m = Ji.repeat(1, self.n_inputs).view(-1, self.n_inputs)
                    Ji_dx = autograd.grad(Ji_m, x_m, mask, create_graph=True)[0]
                    Ji_dx = Ji_dx.reshape(n, self.n_outputs, self.n_inputs)
                    Jd[:, :, i] = torch.einsum('bij,bj->bi', Ji_dx, xd)

        if not self.training:
            Jd = Jd.detach()

        return Jd

    def forward(self, x, xd=None, order=2):
        if self.J_ is not None:
            if order == 1:
                y = self.psi(x)
                J = self.J_(x)
                if not self.training:
                    y = y.detach()
                    J = J.detach()
                return y, J

            # if order == 2
            assert xd is not None, ValueError("Taskmap requires xd for the second-order version")
            if self.J_dot_ is not None:
                y = self.psi(x)
                J = self.J_(x)
                Jd = self.J_dot_(x, xd)
                yd = torch.bmm(xd.unsqueeze(1), J.permute(0, 2, 1)).squeeze(1)
                if not self.training:
                    y = y.detach()
                    J = J.detach()
                    Jd = Jd.detach()
                    yd = yd.detach()
                return y, yd, J, Jd
            else:
                n = x.size()[0]
                x_m = x.repeat(1, self.n_outputs*self.n_inputs).view(-1, self.n_inputs)
                x_m.requires_grad_(True)
                J_m = self.J_(x_m)

                if J_m.requires_grad:
                    J_m_vec = J_m.reshape(-1, self.n_outputs*self.n_inputs)
                    mask = torch.eye(self.n_outputs*self.n_inputs, device=self.device).repeat(n, 1)
                    dJ_m_vec, = autograd.grad(J_m_vec, x_m, mask, create_graph=True, allow_unused=True, retain_graph=True)
                    dJ_m_vec = dJ_m_vec
                    dJ_m_xd_vec = dJ_m_vec * xd.repeat(1, self.n_outputs*self.n_inputs).view(-1, self.n_inputs)
                    dJ_m_xd = dJ_m_xd_vec.sum(dim=1)
                    Jd = dJ_m_xd.reshape(-1, self.n_outputs, self.n_inputs)
                else:
                    Jd = torch.zeros(n, self.n_outputs, self.n_inputs, device=self.device)

                J = J_m[::self.n_outputs*self.n_inputs]
                yd = torch.bmm(xd.unsqueeze(1), J.permute(0, 2, 1)).squeeze(1)
                y = self.psi(x)

                if not self.training:
                    y = y.detach()
                    J = J.detach()
                    Jd = Jd.detach()
                    yd = yd.detach()

                return y, yd, J, Jd
        else:
                n = x.size()[0]
                x_m = x.repeat(1, self.n_outputs).view(-1, self.n_inputs)
                x_m.requires_grad_(True)
                y_m = self.psi(x_m)
                y = y_m[::self.n_outputs, :]

                if y_m.requires_grad:
                    mask = torch.eye(self.n_outputs, device=self.device).repeat(n, 1)
                    J = autograd.grad(y_m, x_m, mask, create_graph=True, allow_unused=True)[0]
                    if J is None:
                        J = torch.zeros(n, self.n_outputs, self.n_inputs, device=self.device)
                    else:
                        J = J.reshape(n, self.n_outputs, self.n_inputs)
                else:  # if requires grad is False, then output has no dependence of input
                    J = torch.zeros(n, self.n_outputs, self.n_inputs, device=self.device)

                if order == 1:
                    if not self.training:
                        J = J.detach()
                        y = y.detach()
                    return y, J

                # if order == 2
                assert xd is not None, ValueError("Taskmap requires xd for the second-order version")
                if self.J_dot_ is None: #TODO: use the same way as done in next condition (no for loops)
                    Jd = torch.zeros(n, self.n_outputs, self.n_inputs, device=self.device)
                    if J.requires_grad:  # if requires grad is False, then J has no dependence of input
                        # Finding jacobian of each column and applying chain rule
                        for i in range(self.n_inputs):
                            Ji = J[:, :, i]
                            mask = torch.eye(self.n_outputs, self.n_inputs, device=self.device).repeat(n, 1)
                            Ji_m = Ji.repeat(1, self.n_inputs).view(-1, self.n_inputs)
                            Ji_dx = autograd.grad(Ji_m, x_m, mask, create_graph=True)[0]
                            Ji_dx = Ji_dx.reshape(n, self.n_outputs, self.n_inputs)
                            Jd[:, :, i] = torch.einsum('bij,bj->bi', Ji_dx, xd)
                else:
                    Jd = self.J_dot_(x, xd)

                yd = torch.bmm(xd.unsqueeze(1), J.permute(0, 2, 1)).squeeze(1)
                if not self.training:
                    y = y.detach()
                    J = J.detach()
                    Jd = Jd.detach()
                    yd = yd.detach()

                return y, yd, J, Jd


class ComposedTaskMap(TaskMap):
    def __init__(self, taskmaps=None, use_numerical_jacobian=False, device=torch.device('cpu')):
        super(ComposedTaskMap, self).__init__(n_inputs=None,
                                              n_outputs=None,
                                              psi=self.psi, device=device)
        self.taskmaps = taskmaps
        self.device = device
        self.use_numerical_jacobian = use_numerical_jacobian  # use J/Jd for the entire chain using autodiff

    def forward(self, x, xd=None, order=2):
        if self.use_numerical_jacobian: # uses the autodiff version given by taskmap
            return super(ComposedTaskMap, self).forward(x=x, xd=xd, order=order)

        return self.composed_forward(x=x, xd=xd, order=order)

    def psi(self, x):
        for i in range(len(self.taskmaps)):
            x = self.taskmaps[i].psi(x)
        return x

    def J(self, x):
        if self.use_numerical_jacobian:
            return super(ComposedTaskMap, self).J(x=x)

        n, d = x.size()
        J = torch.eye(d, d, device=self.device).repeat(n, 1, 1)
        for i in range(len(self.taskmaps)):
            x, J_i = self.taskmaps(x=x, order=1)
            J = torch.bmm(J_i, J)
        return J

    def J_dot(self, x, xd):
        if self.use_numerical_jacobian:
            return super(ComposedTaskMap, self).J_dot(x=x, xd=xd)

        n, d = x.size()
        J = torch.eye(d, d, device=self.device).repeat(n, 1, 1)
        Jd = torch.zeros(n, d, d, device=self.device)
        for i in range(len(self.taskmaps)):
            x, xd, J_i, Jd_i = self.taskmaps[i](x=x, xd=xd, order=2)
            Jd = torch.bmm(J_i, Jd) + torch.bmm(Jd_i, J)
            J = torch.bmm(J_i, J)
        return Jd

    # analytically compose taskamps (including the jacobians!)
    def composed_forward(self, x, xd=None, order=2):
        n, d = x.size()
        J = torch.eye(d, d, device=self.device).repeat(n, 1, 1)

        if order == 1:
            for i in range(len(self.taskmaps)):
                x, J_i = self.taskmaps[i](x, order=order)
                J = torch.bmm(J_i, J)
            return x, J

        Jd = torch.zeros(n, d, d, device=self.device)
        for i in range(len(self.taskmaps)):
            x, xd, J_i, Jd_i = self.taskmaps[i](x, xd, order=order)
            Jd = torch.bmm(J_i, Jd) + torch.bmm(Jd_i, J)
            J = torch.bmm(J_i, J)
        return x, xd, J, Jd

    @property
    def taskmaps(self):
        return self.__taskmaps

    @taskmaps.setter
    def taskmaps(self, taskmaps):
        if taskmaps is None:
            self.__taskmaps = torch.nn.ModuleList()
        else:
            if type(taskmaps) == list:
                self.__taskmaps = torch.nn.ModuleList(taskmaps)
            else:
                self.__taskmaps = taskmaps

    @property
    def n_inputs(self):
        return self.taskmaps[0].n_inputs

    @n_inputs.setter
    def n_inputs(self, n_inputs):
        self.__n_inputs = n_inputs

    @property
    def n_outputs(self):
        return self.taskmaps[-1].n_outputs

    @n_outputs.setter
    def n_outputs(self, n_outputs):
        self.__n_outputs = n_outputs
