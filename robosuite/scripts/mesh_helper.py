"""
Convenience script to help import object pieces based on STL meshes into mujoco.
Assumes that the meshes have already been centered.
"""

import argparse
import numpy as np
import stl

def bounding_box_from_mesh(mesh, scale):
    """
    Helper function to retrieve bounding box for a mesh.
    The bounding box corresponds to box half-sizes for a geom
    that would contain the mesh.
    """
    return scale * np.maximum(
        np.maximum(mesh.v0.max(axis=0), -mesh.v0.min(axis=0)),
        np.maximum(mesh.v1.max(axis=0), -mesh.v1.min(axis=0)),
        np.maximum(mesh.v2.max(axis=0), -mesh.v2.min(axis=0)),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path to stl to import
    parser.add_argument(
        "--stl",
        type=str,
    )
    # scale factor to apply to the mesh
    parser.add_argument(
        "--scale",
        type=float,
    )
    args = parser.parse_args()

    stl_file = args.stl
    scale = args.scale

    # load the mesh
    mesh = stl.mesh.Mesh.from_file(args.stl)

    # compute bounding box for the mesh
    bbox = bounding_box_from_mesh(mesh=mesh, scale=scale)
    print("\nbounding box: {}\n".format(bbox))
