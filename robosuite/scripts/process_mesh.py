import numpy as np
import trimesh
import os
import itertools
import robosuite

import xml.etree.ElementTree as ET

from robosuite.scripts.convert_obj_to_msh import generate_msh_file

def generate_meshes_and_bb(model_path, scaling, show_meshes=False):
    _, type = get_model_name_and_type(model_path)
    assert type in ['obj', 'stl']
    
    # load input mesh
    resolver = trimesh.resolvers.FilePathResolver(os.path.dirname(model_path))
    model = trimesh.load(model_path, resolver=resolver)
    
    # set mesh paths
    assert model_path
    vis_mesh_path = model_path[:-len(type)-1] + '_vis.msh'
    coll_mesh_path = model_path[:-len(type)-1] + '_coll.stl'
    bb_mesh_path = model_path[:-len(type)-1] + '_bb.stl'
    
    # generate meshes
    vis_mesh = generate_msh_file(model_path, vis_mesh_path)
    coll_mesh = model.convex_hull
    bb_mesh = coll_mesh.bounding_box_oriented
    
    # save meshes
    # vis_mesh.export(vis_mesh_path)
    coll_mesh.export(coll_mesh_path)
    bb_mesh.export(bb_mesh_path)

    # get bounding box information
    ext = np.array(bb_mesh.primitive.extents)
    transform = np.array(bb_mesh.primitive.transform)

    center = transform[:-1,3]
    rot = transform[:3,:3]

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
    if show_meshes:
        vis_mesh.show()
        coll_mesh.show()
        (coll_mesh + bb_mesh).show()
    
    info = dict(
        vis_mesh_path=vis_mesh_path,
        coll_mesh_path=coll_mesh_path,
        bb_mesh_path=bb_mesh_path,
        scaling=scaling,
        center=center,
        axes=axes,
    )
    
    return info
    
def get_model_name_and_type(model_path):
    split_name = os.path.basename(model_path).split(".")
    assert len(split_name) == 2
    return split_name[0], split_name[1]
    

def generate_object_xml(model_path, info, show_all_meshes_in_xml):
    base_path = os.path.abspath(os.path.join(os.path.dirname(robosuite.__file__), os.pardir))
    xml_base_path = os.path.join(base_path, 'robosuite/models/assets/objects')
    
    name, _ = get_model_name_and_type(model_path)
    
    # load template xml
    tree = ET.parse(os.path.join(xml_base_path, 'template.xml'))
    root = tree.getroot()
    
    # set model name
    root.attrib["model"] = name
    
    # set mesh file, name, and scale
    asset = root.find('asset')
    for mesh in asset.iter('mesh'):
        for k in ['name', 'file']:
            mesh.attrib[k] = mesh.attrib[k].replace('template', name)
        mesh.attrib['scale'] = '{sc} {sc} {sc}'.format(sc=info['scaling'])
        
    worldbody = root.find('worldbody')
    body = worldbody.find('body').find('body')
    bb_geom = None
    for geom in body.iter('geom'):
        for k in ['mesh', 'name']:
            geom.attrib[k] = geom.attrib[k].replace('template', name)
            
        if geom.attrib['name'].endswith('_boundingbox'):
            bb_geom = geom
            
        if show_all_meshes_in_xml:
            if geom.attrib['name'].endswith('_collision'):
                geom.attrib['rgba'] = "0.8 0.8 0.8 0.2"
            elif geom.attrib['name'].endswith('_boundingbox'):
                geom.attrib['rgba'] = "0.8 0.8 0.8 0.05"
    
    # delete bounding box mesh, not needed
    if not show_all_meshes_in_xml:
        body.remove(bb_geom)
    
    # save xml for new model
    tree.write(
        os.path.join(xml_base_path, '{}.xml'.format(name)),
        encoding="utf-8"
    )
    

# input args
# model_path = '/Users/soroushnasiriany/research/robosuite/robosuite/models/assets/objects/meshes/blender/blender.stl'

# obj_name = 'blender'; scaling = 0.04
# obj_name = 'mug'; scaling = 0.40
obj_name = 'spoon'; scaling = 0.04

show_meshes = False
show_all_meshes_in_xml = False

base_path = os.path.abspath(os.path.join(os.path.dirname(robosuite.__file__), os.pardir))
mesh_base_path = os.path.join(base_path, 'robosuite/models/assets/objects/meshes')
model_path = os.path.join(mesh_base_path, obj_name, '{}.obj'.format(obj_name))

info = generate_meshes_and_bb(model_path, scaling, show_meshes)
generate_object_xml(model_path, info, show_all_meshes_in_xml)
