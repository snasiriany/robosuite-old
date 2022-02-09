import numpy as np
import trimesh
import os
import itertools

# input args
input_mesh_path = '/Users/soroushnasiriany/research/robosuite/robosuite/models/assets/objects/meshes/blender.stl'
show = False

# load input mesh
resolver = trimesh.resolvers.FilePathResolver(os.path.dirname(input_mesh_path))
input_mesh = trimesh.load(input_mesh_path, resolver=resolver)

# generate and export convex hull and bounding box
conv_hull = input_mesh.convex_hull
bb = conv_hull.bounding_box_oriented
conv_hull.export(input_mesh_path[:-4] + '_collision.stl')
bb.export(input_mesh_path[:-4] + '_bb.stl')

# get bounding box information
ext = np.array(bb.primitive.extents)
transform = np.array(bb.primitive.transform)

center = transform[:-1,3]
rot = transform[:3,:3]

# scaling of mesh
scaling = 0.04

print("center:", center * scaling)

# get the axes
print()
axes = []
for i in range(3):
    axes.append(rot[:,i] * ext[i] / 2)
    print("axis {}:".format(i+1), (axes[i] + center) * scaling)

# get the bounding box corners
print()
for i, (m0, m1, m2) in enumerate(itertools.product(*[[-1, 1], [-1, 1], [-1, 1]])):
    p = center + (axes[0] * m0) + (axes[1] * m1) + (axes[2] * m2)
    print("corner {}:".format(i+1), p * scaling)

# display the visual mesh, convex hull mesh, and bounding box
if show:
    input_mesh.show()
    conv_hull.show()
    (conv_hull + bb).show()