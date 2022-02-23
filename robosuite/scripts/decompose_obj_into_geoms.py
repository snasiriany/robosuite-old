import trimesh
import tempfile
import os
import numpy as np
from robosuite.utils.transform_utils import mat2quat

model_path = '/Users/soroushnasiriany/research/robosuite/robosuite/models/assets/objects/meshes/test/test.obj'
show = False

f = open(model_path,'r')
lines = f.readlines()
f.close()

objs = []

obj_start_inds = [i for (i, line) in enumerate(lines) if line.startswith('o ')]

for start, end in zip(obj_start_inds, obj_start_inds[1:] + [len(lines)]):
    obj_lines = lines[start:end]
    
    # filter lines corresponding to faces
    obj_lines = [line for line in obj_lines if not line.startswith('f ')]
    
    tmp = tempfile.NamedTemporaryFile(suffix='.obj')

    # Open the file for writing.
    with open(tmp.name, 'w') as f:
        [f.write(line) for line in obj_lines] # where `stuff` is, y'know... stuff to write (a string)
    
    mesh = trimesh.load(tmp.name)
    
    if show:
        (mesh.convex_hull).show()
        
    obj_type_substr = obj_lines[0][2:]
    if obj_type_substr.startswith('Cube'):
        obj_type = 'box'
    elif obj_type_substr.startswith('Cylinder'):
        obj_type = 'cylinder'
    elif obj_type_substr.startswith('Sphere'):
        obj_type = 'ellipsoid'
    else:
        raise ValueError
        
    bb_mesh = mesh.bounding_box_oriented
    ext = np.array(bb_mesh.primitive.extents)
    transform = np.array(bb_mesh.primitive.transform)

    center = transform[:-1,3]
    rot = transform[:3,:3]
    
    obj_info = dict(
        obj_type=obj_type,
        pos=center,
        quat=mat2quat(rot),
        size=ext,
    )
    print(obj_info)
