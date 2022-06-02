import torch
import torch.nn as nn


class ContextForceNet(nn.Module):
    def __init__(self, lagrangian_force_nets, n_dims, metric_scaling=None, name=None):
        super(ContextForceNet, self).__init__()
        self.lagrangian_force_nets = nn.ModuleList([net for net in lagrangian_force_nets])
        self.n_dims = n_dims
        self.name = name

        if metric_scaling is None:
            self.metric_scaling = [1. for net in self.lagrangian_force_nets]
        else:
            self.metric_scaling = metric_scaling

    def forward(self, state, q_leaf_list, qd_leaf_list, J_list, Jd_list, force_list, metric_list):
        assert state.dim() == 1 or state.dim() == 2 or state.dim() == 3
        if state.dim() == 1:
            state = state.unsqueeze(0)
        elif state.dim() == 3:
            state = state.squeeze(2)

        assert state.dim() == 2
        assert state.size()[1] == 2 * self.n_dims

        q = state[:, :self.n_dims]
        qd = state[:, self.n_dims:]

        n_samples = q.size(0)

        force_root = torch.zeros(n_samples, self.n_dims)
        metric_root = torch.zeros(n_samples, self.n_dims, self.n_dims)

        for net, x, xd, J, Jd, force, metric, scaling in zip(
                self.lagrangian_force_nets,
                q_leaf_list, qd_leaf_list,
                J_list, Jd_list,
                force_list, metric_list, self.metric_scaling):
            if net is not None:
                state = torch.cat((x, xd), dim=1)
                force, metric = net(state)
            force_root += scaling * torch.einsum('bji, bj->bi', J, force) - torch.einsum('bji, bjk, bkl, bl->bi', J, metric, Jd, qd)
            metric_root += scaling * torch.einsum('bji, bjk, bkl->bil', J, metric, J)

        return force_root, metric_root


class ContextMomentumNet(nn.Module):
    def __init__(self, lagrangian_vel_nets, n_dims, metric_scaling=None, name=None):
        super(ContextMomentumNet, self).__init__()
        self.lagrangian_vel_nets = nn.ModuleList([net for net in lagrangian_vel_nets])
        self.n_dims = n_dims
        self.name = name

        if metric_scaling is None:
            self.metric_scaling = [1. for net in self.lagrangian_vel_nets]
        else:
            self.metric_scaling = metric_scaling

        assert len(self.metric_scaling) == len(self.lagrangian_vel_nets)

    def forward(self, state, q_leaf_list, J_list, momentum_list, metric_list):
        assert state.dim() == 1 or state.dim() == 2 or state.dim() == 3
        if state.dim() == 1:
            state = state.unsqueeze(0)
        elif state.dim() == 3:
            state = state.squeeze(2)

        assert state.dim() == 2
        assert state.size()[1] == self.n_dims

        q = state

        n_samples = q.size(0)

        momentum_root = torch.zeros(n_samples, self.n_dims)
        metric_root = torch.zeros(n_samples, self.n_dims, self.n_dims)

        for net, x, J, momentum, metric, scaling in zip(
                self.lagrangian_vel_nets,
                q_leaf_list, J_list,
                momentum_list, metric_list, self.metric_scaling):
            if net is not None:
                momentum, metric = net(x)
            momentum_root += scaling * torch.einsum('bji, bj->bi', J, momentum)
            metric_root += scaling * torch.einsum('bji, bjk, bkl->bil', J, metric, J)

        return momentum_root, metric_root


class ContextMomentumNet2(nn.Module):
    def __init__(self, lagrangian_vel_nets, cspace_dims, metric_scaling=None, name=None):
        super(ContextMomentumNet2, self).__init__()
        self.lagrangian_vel_nets = nn.ModuleList([net for net in lagrangian_vel_nets])
        self.cspace_dims = cspace_dims
        self.name = name

        if metric_scaling is None:
            self.metric_scaling = [1. for net in self.lagrangian_vel_nets]
        else:
            self.metric_scaling = metric_scaling

        assert len(self.metric_scaling) == len(self.lagrangian_vel_nets)

    def forward(self, state, q_leaf_list, J_list, momentum_list, metric_list):
        assert state.dim() == 1 or state.dim() == 2 or state.dim() == 3
        if state.dim() == 1:
            state = state.unsqueeze(0)
        elif state.dim() == 3:
            state = state.squeeze(2)

        assert state.dim() == 2
        assert state.size()[1] == self.cspace_dims

        q = state
        n_samples = q.size(0)

        momentum_root = torch.zeros(n_samples, self.cspace_dims)
        metric_root = torch.zeros(n_samples, self.cspace_dims, self.cspace_dims)
        jacobians = []

        for net, x, J, momentum, metric, scaling in zip(
                self.lagrangian_vel_nets,
                q_leaf_list, J_list,
                momentum_list, metric_list, self.metric_scaling):
            if net is not None:
                momentum, metric = net(x)
                jacobians.append(J)
            momentum_root += scaling * torch.einsum('bji, bj->bi', J, momentum)
            metric_root += scaling * torch.einsum('bji, bjk, bkl->bil', J, metric, J)
        return momentum_root, metric_root, jacobians
