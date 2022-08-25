import numpy as np
import trimesh
import os
import itertools
import robosuite
import argparse
import shutil
import xml.etree.ElementTree as ET

from robosuite.scripts.convert_obj_to_msh import generate_msh_file

def generate_meshes_and_bb(model_path, scaling, show_meshes=False, verbose=False):
    name, type = get_model_name_and_type(model_path)
    assert type in ['obj', 'stl']
        
    cwd = os.getcwd()
    home_robosuite_dir = cwd[:cwd.rfind("/")]
    new_mesh_dir = home_robosuite_dir + '/models/assets/objects/meshes/' + name

    # copy obj/stl and file to new file path
    if model_path[:model_path.rfind("/")] != new_mesh_dir:
        if not os.path.isdir(new_mesh_dir):
            os.mkdir(new_mesh_dir)

        shutil.copy(model_path, new_mesh_dir)
        if type == 'obj':
            shutil.copy(model_path[:model_path.rfind(".")] + '.mtl', new_mesh_dir)

    model_path = new_mesh_dir + '/' + name + '.' + type

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

    if verbose:
        print("center:", center * scaling)

    # get the axes
    if verbose:
        print()
    axes = []
    for i in range(3):
        axes.append(rot[:,i] * ext[i] / 2)
        if verbose:
            print("axis {}:".format(i+1), (axes[i] + center) * scaling)

    # get the bounding box corners
    if verbose:
        print()
    for i, (m0, m1, m2) in enumerate(itertools.product(*[[-1, 1], [-1, 1], [-1, 1]])):
        p = center + (axes[0] * m0) + (axes[1] * m1) + (axes[2] * m2)
        if verbose:
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
    

def generate_object_xml(model_path, texture_file, info, show_all_meshes_in_xml, verbose=False):
    base_path = os.path.abspath(os.path.join(os.path.dirname(robosuite.__file__), os.pardir))
    xml_base_path = os.path.join(base_path, 'robosuite/models/assets/objects')
    mtl_base_path = os.path.join(base_path, 'robosuite/models/assets/objects/meshes')
    
    name, _ = get_model_name_and_type(model_path)

    texture_from_mtl = False
    if texture_file == None:
        # get the texture used in the mtl file
        mtl_file = os.path.join(mtl_base_path, name+'/'+name+'.mtl')
        texture_dict = {}
        with open(mtl_file, 'r') as f:
            for i, line in enumerate(f.readlines()):
                key, value = line.split(" ", 1)
                texture_dict[key] = value.strip()

        if 'map_Kd' in texture_dict:
            texture_file = texture_dict['map_Kd']
            texture_from_mtl = True
            
    # ------   pass in custom texture file or we read it directly from mtl -------
    # potential texture_file values
        # custom texture_file that was inputted by user (texture_file != None and texture_from_mtl = False)
        # texture_file from mtl (texture_file != None and texture_from_mtl = True)
        # none if not custom texture file and nothing from mtl (otherwise)

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

    texture = asset.find('texture')
    for k in ['name', 'file']:
        if k == "file":
            if texture_file != None and texture_from_mtl == False:
                texture.attrib[k] = texture.attrib[k].replace('template.png', texture_file)
            elif texture_file != None and texture_from_mtl == True:
                texture.attrib[k] = texture_file
            else:
                texture.attrib[k] = texture.attrib[k].replace('template.png', 'ceramic.png')
        else:
            texture.attrib[k] = texture.attrib[k].replace('template', name)

    material = asset.find('material')
    for k in ['name', 'texture']:
        material.attrib[k] = material.attrib[k].replace('template', name)
        
    worldbody = root.find('worldbody')
    body = worldbody.find('body').find('body')
    bb_geom = None
    for geom in body.iter('geom'):
        for k in ['mesh', 'name', 'material']:
            if k in geom.attrib:
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
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--texture_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--show_meshes",
        action="store_true",
    )
    parser.add_argument(
        "--show_all_meshes_in_xml",
        action="store_true",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    
    args = parser.parse_args()
    
    

    # obj_name = 'blender'; scaling = 0.04
    # obj_name = 'mug'; scaling = 0.40
    # obj_name = 'spoon'; scaling = 0.04

    # show_meshes = False
    # show_all_meshes_in_xml = False

    # base_path = os.path.abspath(os.path.join(os.path.dirname(robosuite.__file__), os.pardir))
    # mesh_base_path = os.path.join(base_path, 'robosuite/models/assets/objects/meshes')
    # model_path = os.path.join(mesh_base_path, obj_name, '{}.obj'.format(obj_name))

    info = generate_meshes_and_bb(args.model_path, args.scale, args.show_meshes, args.verbose)
    generate_object_xml(args.model_path, args.texture_file, info, args.show_all_meshes_in_xml, args.verbose)
