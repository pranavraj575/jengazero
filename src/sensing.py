from src.tower import *
import torch
from itertools import chain

DEVICE = 'cpu'


class TorchBlock(torch.nn.Module):
    def __init__(self, pos, yaw):
        """
        block, implemented with pytorch for gradients
        """
        super().__init__()
        self.pos = torch.nn.Parameter(pos)
        self.yaw = torch.nn.Parameter(yaw)

    def vertices(self):
        """
        returns vertices of block
        :return: 8x3 array
        """

        dx, dy, dz = JENGA_BLOCK_DIM

        return torch.tensor([[
            (X,
             Y,
             self.pos[2] + dz*(z_i - .5),
             )
            for (X, Y) in self.vertices_xy()] for z_i in range(2)]).reshape((8, 3))

    def differiantable_rot_matrix(self):
        out = torch.zeros((2, 2), dtype=torch.double)
        out[0, 0] = torch.cos(self.yaw)
        out[0, 1] = torch.sin(self.yaw)
        out[1, 0] = -torch.sin(self.yaw)
        out[1, 1] = torch.cos(self.yaw)
        return out

    def differentiable_equations(self):
        vertex_cycle = self.vertices_xy()
        shifted_cycle = vertex_cycle[[(i + 1)%len(vertex_cycle) for i in range(len(vertex_cycle))]]
        diffs = shifted_cycle - vertex_cycle
        norms = diffs@torch.tensor([[0., 1.], [-1., 0.]], dtype=torch.double)

        norms = norms/torch.linalg.norm(norms, dim=-1).reshape((-1, 1))
        bees = torch.bmm(norms.reshape((-1, 1, 2)), vertex_cycle.reshape((-1, 2, 1))).reshape((-1, 1))
        return torch.cat((-norms, bees), dim=1)

    def vertices_xy(self):
        """
        returns xy projected vertices of block
        ordered in a cycle
        :return: 4x2 array
        """
        dx, dy, _ = JENGA_BLOCK_DIM

        location_array = torch.tensor([[
            (dx*(x_i - .5),
             dy*(y_i - .5),
             )
            for x_i in (range(2) if y_i == 0 else range(1, -1, -1))]  # go in reverse for y_i=1
            for y_i in range(2)]).reshape((4, 2))
        return self.pos[:2] + location_array@self.differiantable_rot_matrix()


def block_to_2d_point_cloud(block: Block, points=100, noise=0.):
    """
    returns a set of points sampled from the boundaries of the block
    uses following sampling algorithm:
        pick two adjacent vertices of block
        pick a random point between them
    :param points: number of points to sample
    :param noise: gaussian variance to add to each sampled point
    """
    vertex_cycle = block.vertices_xy()
    shifted_cycle = vertex_cycle[[(i + 1)%len(vertex_cycle) for i in range(len(vertex_cycle))]]

    start_end = np.stack((vertex_cycle, shifted_cycle), axis=1)
    adj_samples = start_end[np.random.randint(len(vertex_cycle), size=points), :, :]
    lambdas = np.random.random(points)  # proportion to go on line along adjacent points

    mult = np.array([lambdas, 1 - lambdas]).T
    mult = mult.reshape((points, 1, -1))
    out = (mult@adj_samples).reshape((points, 2))
    out += np.random.normal(scale=noise, size=out.shape)
    return out


def random_torch_block(L, i, pos_std=0., angle_std=0.):
    block = random_block(L=L, i=i, pos_std=pos_std, angle_std=angle_std)
    return TorchBlock(pos=torch.tensor(block.pos, requires_grad=True, device=DEVICE),
                      yaw=torch.tensor(block.yaw, requires_grad=True, device=DEVICE))


def which_points_in_equations(points, eqs, tolerance=TOLERANCE):
    """
    returns if points np array dim (...,2) are in hull
    eqs of dim (M,3) are the equations of the hull

    return ... matrix of booleans
    """
    # adds one to each row to dot product easier
    aug = torch.ones(points.shape[:-1]).unsqueeze(-1)
    aug_points = torch.cat((points, aug), dim=-1)

    all_dists = aug_points@eqs.T
    dists = torch.max(all_dists, dim=-1).values
    return dists <= tolerance


def block_loss(blocks: [TorchBlock], points, extra_loss=0.01):
    # extra loss adds error of distance of blocks to points not at min distance
    LARGE = 69
    # iterate over each block, find the closest distance of each point to the block
    # then take the min for each point
    # this is the 'error' of the point

    best_to_each = []
    for block in blocks:
        vertex_cycle = block.vertices_xy()

        vtx_diffs = points.reshape((-1, 1, 2)) - vertex_cycle.reshape((1, -1, 2))

        eqs = block.differentiable_equations()
        eq_dists = torch.cat((points, torch.ones((len(points), 1))), dim=1)@eqs.T
        projections = points.reshape((-1, 1, 2)) - (
                    eqs[:, :-1].reshape(1, -1, 2)*eq_dists.reshape((len(eq_dists), -1, 1)))

        proj_diffs = points.reshape((-1, 1, 2)) - projections

        bad_idxs = torch.logical_not(which_points_in_equations(projections, eqs))

        vtx_resid = vtx_diffs.unsqueeze(-2)@vtx_diffs.unsqueeze(-1)
        vtx_resid = vtx_resid.reshape(vtx_resid.shape[:2])

        proj_resid = proj_diffs.unsqueeze(-2)@proj_diffs.unsqueeze(-1)
        proj_resid = proj_resid.reshape(proj_resid.shape[:2])

        proj_resid[bad_idxs] += LARGE

        all_diffs = torch.cat((vtx_resid, proj_resid), dim=1)
        bests = torch.min(all_diffs, dim=1).values
        best_to_each.append(bests.reshape((-1, 1)))
    overall=torch.cat(best_to_each, dim=1)
    overall_best=torch.min(overall,dim=1)
    if len(blocks)>1:
        overall_worst=(torch.sum(overall,dim=1)-overall_best.values)/(len(blocks)-1)
        overall_worst=torch.mean(overall_worst)
    else:
        overall_worst=0.
    loss = torch.mean(overall_best.values)+extra_loss*overall_worst
    loss.backward()
    return loss


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    torch.autograd.set_detect_anomaly(True)
    target = random_block(0, 0, 0, 0)
    target2 = random_block(0, 2, 0, 0)
    points = torch.cat((torch.tensor(block_to_2d_point_cloud(block=target, noise=.001,points=100)),
                        torch.tensor(block_to_2d_point_cloud(block=target2, noise=.001,points=100))))

    block = random_torch_block(1, 0, 0, 0)
    block2 = random_torch_block(1, 1, 0, 0)

    optimizer = torch.optim.Adam(params=chain(block.parameters(), block2.parameters()), lr=.1)

    old = block.vertices_xy().detach().numpy()
    old2 = block2.vertices_xy().detach().numpy()
    for i in range(200):
        print('\r',i, end='')
        optimizer.zero_grad()
        loss = block_loss([block, block2], points, extra_loss=1/(i+1))
        optimizer.step()

    new = block.vertices_xy().detach().numpy()
    new2 = block2.vertices_xy().detach().numpy()

    plt.scatter(points[:, 0], points[:, 1])
    for (plotter, nm) in (old, 'old'), (old2, 'old2'), (new, 'new'), (new2, 'new2'):
        plotter = plotter[[i%len(plotter) for i in range(len(plotter) + 1)]]
        plt.plot(plotter[:, 0], plotter[:, 1], label=nm)
    plt.legend()
    plt.show()
