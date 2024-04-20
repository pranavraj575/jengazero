import os.path

from src.tower import *
import torch
from itertools import chain

DEVICE = 'cpu'
DIR = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))


class TorchBlock(torch.nn.Module):
    def __init__(self, pos, yaw):
        """
        block, implemented with pytorch for gradients
        """
        super().__init__()
        self.pos = torch.nn.Parameter(pos)
        self.yaw = torch.nn.Parameter(yaw)

    def to_vector(self):
        """
        returns block encoded as a vector

        x,y,z,angle
        """
        return np.concatenate((self.pos.detach().numpy(), [self.yaw.detach().item()]))

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


def torch_block_from_vector(vector):
    """
    returns the block encoded by vector

    x,y,z,angle = vector
    """
    return TorchBlock(pos=torch.tensor(vector[:3]), yaw=torch.tensor(vector[3]))


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


def block_loss(blocks: [TorchBlock], points, extra_loss=0.0, extra_distance_loss=0.0):
    # extra loss adds error of distance of blocks to points not at min distance
    # extra distance loss pushes blocks apart
    # returns (loss, distance of each point to closest block line)
    LARGE = 69
    # iterate over each block, find the closest distance of each point to the block
    # then take the min for each point
    # this is the 'error' of the point

    best_to_each = []
    add_distance = []
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
        for block2 in blocks:
            if block2 is not block:
                add_distance.append(torch.linalg.norm(block.pos - block2.pos).reshape((1, 1)))

    # overall = torch.cat(best_to_each, dim=1)
    # for some reason norm error works better than squared error
    overall = torch.sqrt(torch.cat(best_to_each, dim=1))
    overall_best = torch.min(overall, dim=1)
    if len(blocks) > 1:
        overall_worst = (torch.sum(overall, dim=1) - overall_best.values)/(len(blocks) - 1)
        overall_worst = torch.mean(overall_worst)

        distances_loss = torch.mean(torch.cat(add_distance))
    else:
        overall_worst = 0.

        distances_loss = 0.

    loss = torch.mean(overall_best.values) + extra_loss*overall_worst + extra_distance_loss*distances_loss

    return loss, overall_best.values


def infer_block_locations(points):
    """
    infers block numbers and positions from a 2d pointcloud
    :param points: N x 2 tensor of points to fit

    returns (dict(num_blocks -> (loss, solution, point distances, parameter history), predicted num_blocks)
        solution is a list of block vectors
    """

    assisted = 200
    normal_loss = 100
    range_blocks = range(1, 4)

    records = dict()
    for num_blocks in range_blocks:
        best = (None, None, None)
        for L in (0, 1):
            blocks = [random_torch_block(L, (2*i)%3, .01, .01) for i in range(num_blocks)]

            all_params = []
            for block in blocks:
                all_params += list(block.parameters())

            optimizer = torch.optim.Adam(params=all_params,
                                         # lr=.01
                                         )
            record = [
                [block.to_vector() for block in blocks]
            ]

            for i in range(assisted):
                optimizer.zero_grad()
                loss, distances = block_loss(blocks, points,
                                             extra_loss=1/(i + 1),
                                             # extra_distance_loss=1/(i + 1),
                                             )
                loss.backward()
                optimizer.step()
                record.append([block.to_vector() for block in blocks])

            for i in range(normal_loss):
                optimizer.zero_grad()
                loss, distances = block_loss(blocks, points,
                                             )
                loss.backward()
                optimizer.step()
                record.append([block.to_vector() for block in blocks])

            final_loss, distances = block_loss(blocks, points)
            final_loss = final_loss.item()
            if best[0] is None or final_loss < best[0]:
                best = (final_loss, record, distances)
        final_loss, record, distances = best

        # failed = distances > min(JENGA_BLOCK_DIM)/8
        # succ = torch.logical_not(failed)

        records[num_blocks] = (final_loss, record[-1], distances, record)

    predicted_blocks = 1

    if records[2][0] < 0.5*records[1][0]:
        predicted_blocks = 2
        if records[3][0] < (0.3)*records[2][0]:
            predicted_blocks = 3
    return records, predicted_blocks


if __name__ == "__main__":
    seed(69420)

    # torch.autograd.set_detect_anomaly(True)

    for which_blocks in range(1, 8):
        # encoded blocks to include
        # on [1,7]
        # ex: 7 has binary 111, so it contains all three blocks
        range_blocks = range(1, 4)
        which_blocks = bin(which_blocks)[2:]
        while len(which_blocks) < 3:
            which_blocks = '0' + which_blocks
        print('blocks:', which_blocks)

        targets = [random_block(0, i, .001, .003) for i in range(3) if which_blocks[i] == '1']
        points = torch.cat([torch.tensor(block_to_2d_point_cloud(block=target, noise=.001, points=100))
                            for target in targets])

        records, predicted_blocks = infer_block_locations(points)

        print('prediced number:', predicted_blocks)
        print('actual number:', len(targets))

        if False:
            continue
        from matplotlib import pyplot as plt
        import matplotlib.animation as animation

        save_folder = os.path.join(DIR, 'temp', which_blocks)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        plt.plot(range_blocks, [records[k][0] for k in range_blocks])
        plt.xlabel('number of blocks')
        plt.xticks(range_blocks)
        plt.ylabel('Loss')
        plt.ylim(0, plt.ylim()[1])
        plt.title("Final Loss Plot")
        plt.savefig(os.path.join(save_folder, 'loss_plot.png'))
        # plt.show()
        plt.close()

        for k in records:
            record = records[k][-1]

            fig, ax = plt.subplots()
            scat = ax.scatter(points[:, 0], points[:, 1])
            lines = [ax.plot(points[:, 0], points[:, 1])[0] for _ in record[0]]

            xbnd = list(plt.xlim())
            ybnd = list(plt.ylim())

            for i in range(len(record)):
                for vectored in record[i]:
                    blocked = torch_block_from_vector(vectored)
                    vertices = blocked.vertices_xy().detach().numpy()
                    xbnd[0] = min(xbnd[0], np.min(vertices[:, 0]))
                    xbnd[1] = max(xbnd[1], np.min(vertices[:, 0]))

                    ybnd[0] = min(ybnd[0], np.min(vertices[:, 1]))
                    ybnd[1] = max(ybnd[1], np.min(vertices[:, 1]))

            plt.xlim(xbnd)
            plt.ylim(ybnd)


            def update(frame):
                i = frame%len(record)
                for k, vectored in enumerate(record[i]):
                    blocked = torch_block_from_vector(vectored)
                    plotter = blocked.vertices_xy().detach().numpy()
                    plotter = plotter[[i%len(plotter) for i in range(len(plotter) + 1)]]
                    lines[k].set_xdata(plotter[:, 0])
                    lines[k].set_ydata(plotter[:, 1])

                return [scat] + lines


            ani = animation.FuncAnimation(fig=fig, func=update, frames=200, interval=60)
            ani.save(os.path.join(save_folder, str(k) + '_blocks.gif'))
            # plt.show()
            plt.close()

