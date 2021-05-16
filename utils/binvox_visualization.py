# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from mpl_toolkits.mplot3d import Axes3D


def get_volume_views(volume, save_dir, n_itr):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    volume = volume.squeeze().__ge__(0.5)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    ax.voxels(volume, edgecolor="k")

    save_path = os.path.join(save_dir, 'voxels-%06d.png' % n_itr)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return cv2.imread(save_path)

def get_RGBvolume_views(volume, save_dir, n_itr):
    # prepare some coordinates
    # x, y, z = np.indices((8, 8, 8))

    # # draw cuboids in the top left and bottom right corners, and a link between them
    # cube1 = (x < 3) & (y < 3) & (z < 3)
    # cube2 = (x >= 5) & (y >= 5) & (z >= 5)
    # link = abs(x - y) + abs(y - z) + abs(z - x) <= 2

    # # combine the objects into a single boolean array
    # voxels = cube1 | cube2 | link

    # # set the colors of each object
    # colors = np.empty(voxels.shape, dtype=object)
    # colors[link] = 'red'
    # colors[cube1] = 'blue'
    # colors[cube2] = 'green'

    # # and plot everything
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.voxels(voxels, facecolors=colors, edgecolor='k')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    volume = volume.squeeze()
    voxels = volume[0,:,:].__ge__(0.5)
    # sphere = volume[0,:-1,:-1,:-1].__ge__(-0.5)
    # idx = volume.transpose(1,2,3,0) < 0
    # volume.transpose(1,2,3,0)[idx]=0
    colors = volume.transpose(1,2,3,0)
    # colors[..., 0] = volume[0,:,:,:]
    # colors[..., 1] = volume[1,:,:,:]
    # colors[..., 2] = volume[2,:,:,:]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    ax.voxels(voxels,
          facecolors=colors,)
        #   edgecolors='k' )
    plt.show()
    save_path = os.path.join(save_dir, 'voxels-%06d.png' % n_itr)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return cv2.imread(save_path)