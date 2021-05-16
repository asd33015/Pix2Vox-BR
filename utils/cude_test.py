import skimage
from skimage import measure
import binvox_rw
import numpy as np
import CommonUtil as util
import ObjIO


with open('./GT_23.binvox', 'rb') as f:
        volume = binvox_rw.read_as_3d_array(f)
        volume = volume.data.astype(np.float32)#.transpose(0,1,)

vertices, simplices, normals, _ = measure.marching_cubes_lewiner(volume, 0.5)
vertices = vertices*2.0
mesh = dict()
mesh['v'] = vertices
mesh['f'] = simplices
mesh['f'] = mesh['f'][:, (1, 0, 2)]
mesh['vn'] = util.calc_normal(mesh)
print('mesh[v] =', type(mesh['v']), mesh['v'].shape)
print('mesh[vn] =', type(mesh['vn']), mesh['vn'].shape)
print('mesh[f] =', type(mesh['f']), mesh['f'].shape)

mesh_ = dict()      # extract_hd_mesh(vol)
mesh_['v'] = np.copy(mesh['v'])
mesh_['f'] = np.copy(mesh['f'])
mesh_['vn'] = np.copy(mesh['vn'])
ObjIO.save_obj_data_binary(mesh_, 'GT23_out.obj')