import torch


def strip_confidence_grads(state):
    if state['current_line_batch'].grad is not None:
        state['current_line_batch'].grad.data[:, :, -1] = 0.


def store_transport_plan(ot_loss, idx):

    def fn(state):
        if state['current_step'] == 0:
            return

        ot_loss.potentials = True

        vector_masses = state['vector_masses'][idx]
        vector_coords = state['vector_coords'][idx]
        raster_masses = state['raster_masses'][idx]
        raster_coords = state['raster_coords'][idx]

        N, M, D = vector_coords.shape[0], raster_coords.shape[0], vector_coords.shape[1]

        dual_f, dual_g = ot_loss(vector_masses, vector_coords, raster_masses, raster_coords)

        a_i, x_i = vector_masses.view(N, 1), vector_coords.view(N, 1, D)
        b_j, y_j = raster_masses.view(1, M), raster_coords.view(1, M, D)
        F_i, G_j = dual_f.view(N, 1), dual_g.view(1, M)

        C_ij = (1 / ot_loss.p) * ((x_i - y_j) ** ot_loss.p).sum(-1)  # (N,M) cost matrix
        eps = ot_loss.blur ** ot_loss.p  # temperature epsilon
        P_ij = ((F_i + G_j - C_ij) / eps).exp() * (a_i * b_j)  # (N,M) transport plan

        state['transport_plan'] = P_ij

    return fn


def store_render_difference(state):
    state['difference'] = torch.abs(state['render'] - state['raster'])


def store_grads(state):
    state['grads'] = state['current_line_batch'].grad.clone().detach().cpu().numpy()


class LossComposition:
    def __init__(self):
        self.loss_fns = []

    def add(self, fn):
        self.loss_fns.append(fn)

    def __call__(self, state):
        assert len(self.loss_fns) > 0

        value = self.loss_fns[0](state)
        for loss_fn in self.loss_fns[1:]:
            value += loss_fn(state)

        return value