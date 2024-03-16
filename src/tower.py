import numpy as np

JENGA_BLOCK_DIM = np.array((.075, .025, .015))  # in meters
JENGA_BLOCK_SPACING = .005  # in meters

WOOD_DENSITY = 0.5  # kg/m^3


class Block:
    def __init__(self, pos, orientation):
        """
        represents one block
        :param pos: position, np array
        :param orientation: roll, pitch, yaw, represents orientation

        Note: generally, roll and pitch are 0 and yaw is the only relevant one
        """
        self.pos = pos
        if orientation is None:
            orientation = np.zeros(3)
        self.orientation = orientation

    def com(self):
        """
        :return: np array, center of mass (is just pos)
        """
        return self.pos


def random_block(L, i, pos_std=0., angle_std=0.):
    """
    creates a randomly placed block at specified level and index
    :param L: level of block in tower
    :param i: index of block in level (increases +x for even levels, +y for odd levels)
        0 is the furthest negative, 2 is the furthest positive
    :param pos_std: std randomness of position
    :param angle_std: std randomness of angle
    :return: block object
    """

    rot = np.array((0., 0., (L%2)*np.pi/2))  # rotate yaw if odd level

    pos = np.zeros(3)
    pos += (0, 0, (L + 0.5)*JENGA_BLOCK_DIM[2])  # height of level + half the height of a block (since COM)

    width = JENGA_BLOCK_DIM[1]
    if L%2:
        pos += ((i - 1)*(width + JENGA_BLOCK_SPACING), 0, 0)
    else:
        pos += (0, (i - 1)*(width + JENGA_BLOCK_SPACING), 0)

    rot = rot + (0., 0., np.random.normal(0, angle_std))
    pos = pos + (np.random.normal(0, pos_std), np.random.normal(0, pos_std), 0.)

    return Block(pos=pos, orientation=rot)


class Tower:
    """
    jenga tower representation
    list of layers, each layer is a list of three booleans
    """

    def __init__(self, block_info=None, default_ht=18, pos_std=.001, angle_std=.001):
        """
        :param block_info: list of block triples, represents each layer
        :param default_ht: height to create tower if block info is None
        :param pos_std: stdev to use for positions if randomly initializing/randomly placing
        :param angle_std: stdev to use for angles if randomly initializing/randomly placing
        """
        self.pos_std = pos_std
        self.angle_std = angle_std
        if block_info is None:
            block_info = [
                [random_block(L=level, i=i, pos_std=pos_std, angle_std=angle_std) for i in range(3)]
                for level in range(default_ht)
            ]
        self.block_info = block_info

        # compute all of these
        self.update_info()

    def update_info(self):
        """
        computation of features to avoid having to compute them each time
        """
        # since each block has equal mass, the COM of tower is just the straight average of COMs of blocks
        # we compute COMs at each layer, the COM of 'subtowers' starting from a layer are of interest
        self.Ns = []  # number of blocks above each layer, including that layer
        self.COMs = []  # COM above each layer (inclusive, so COMs[0] is the COM of the tower, and COMs[-1] is the COM of last layer)

        # going backwards so we can accumulate
        N = 0  # running count
        MOMENT = np.zeros(3)  # running moment (sum of positions of blocks, not averaged yet)
        for layer in self.block_info[::-1]:
            N += sum([t is not None for t in layer])
            MOMENT += np.sum([b.com() for b in layer if b is not None], axis=0)

            self.Ns.append(N)
            self.COMs.append(MOMENT/N)
        self.Ns.reverse()
        self.COMs.reverse()

    def num_blocks(self):
        """
        :return: number of blocks
        """
        return self.Ns[0]

    def height(self):
        """
        :return: height of tower
        """
        return len(self.block_info)

    def com(self):
        """
        center of mass of tower
        :return: np array
        """
        return self.COMs[0]

    def remove_block(self, L, i):
        """
        removes specified block
        :param L: level of block
        :param i: index of block
        :return: Tower object with specified block removed
        """
        if L >= self.height() - 2:
            if not self.top_layer_filled():
                raise Exception("CANNOT REMOVE BLOCK BELOW INCOMPLETE TOP LAYER")
            elif L >= self.height() - 1:
                raise Exception("CANNOT REMOVE BLOCK ON TOP LAYER")

        return Tower(
            [
                [(None if eye == i and ell == L else block) for (eye, block) in enumerate(level)]
                for (ell, level) in enumerate(self.block_info)],
            pos_std=self.pos_std,
            angle_std=self.angle_std,
        )

    def top_layer_filled(self):
        """
        returns if top layer is filled
        """
        return self.Ns[-1] == 3

    def valid_place_blocks(self):
        if self.top_layer_filled():
            return [i for i in range(3)]
        return [i for i in range(3) if self.block_info[-1][i] is None]

    def add_block(self, i, blk_pos_std=None, blk_angle_std=None):
        """
        adds block at specified position
        :param i: position to add
        :param blk_pos_std: pos stdev, if different from default
        :param blk_angle_std: angle stdev, if different from default
        :return: Tower with block added
        """
        if blk_pos_std is None:
            blk_pos_std = self.pos_std
        if blk_angle_std is None:
            blk_angle_std = self.angle_std
        if i not in self.valid_place_blocks():
            raise Exception("i=" + str(i) + " DOES NOT FIT IN LEVEL " + str([(t is not None) for t in self.block_info[-1]]))

        if self.top_layer_filled():
            new_block = random_block(L=self.height(), i=i, pos_std=blk_pos_std, angle_std=blk_angle_std)
            return Tower(self.block_info + [[(new_block if eye == i else None) for eye in range(3)]],
                         pos_std=self.pos_std,
                         angle_std=self.angle_std,
                         )
        else:
            new_block = random_block(L=self.height() - 1, i=i, pos_std=blk_pos_std, angle_std=blk_angle_std)
            return Tower(
                [
                    [(new_block if eye == i and L == self.height() - 1 else block) for (eye, block) in enumerate(level)]
                    for (L, level) in enumerate(self.block_info)],
                pos_std=self.pos_std,
                angle_std=self.angle_std,
            )

    def __str__(self):
        """
        returns string representation of tower
        binary encoding of each level
        i.e. full layer is a 7, layer with one block is a 1, 2, or 4 depending on which block
        blocks are indexed from -x to +x (even layers) or -y to +y (odd layers)
        """
        s = ''
        for L in self.block_info:
            s_L = 0
            for (i, t) in enumerate(L):
                if t is not None:
                    s_L += int(2**i)
            s += str(s_L)
        return s


if __name__ == "__main__":
    t = Tower(pos_std=0, angle_std=0)
    print(t)
    t = t.remove_block(16, 2)
    t = t.add_block(0)
    t = t.add_block(1)
    t = t.add_block(2)

    print(t.com())
    print(t.top_layer_filled())
    print(t.valid_place_blocks())
