import numpy as np
import robosuite.utils.transform_utils as T
from contextlib import contextmanager
from robosuite.models.objects import MujocoObject


def bodyid2geomids(sim, body_id):
    """Finds the list of geom ids given a body id"""
    geoms = []
    for i, bid in enumerate(sim.model.geom_bodyid):
        if bid == body_id:
            geoms.append(i)
    return geoms


def set_body_pose(sim, body_name, pos, quat=None, joint_id=0):
    """
    Sets an body to a target pose
    :param sim: Mujoco simulation
    :param body_name: name of the body
    :param pos: target position
    :param quat: target quaternion
    :return: the object pose before the function call.
    """
    sim_state = sim.get_state()
    joint_name = '{}_{}'.format(body_name, joint_id)
    qpos_addr = sim.model.get_joint_qpos_addr(joint_name)
    pos_addr = qpos_addr[0]
    old_pose = np.array(sim_state.qpos[pos_addr: pos_addr + 7])
    sim_state.qpos[pos_addr: pos_addr + 3] = np.array(pos)
    if quat is not None:
        sim_state.qpos[pos_addr + 3: pos_addr + 7] = T.convert_quat(quat, to="wxyz")
    sim.set_state(sim_state)
    return old_pose[:3], old_pose[3:7]


@contextmanager
def world_saved(sim):
    """
    Context scope for saved world state.
    Simulation state gets saved when entering the scope and gets reset when leaving the scope.
    """
    world_state = sim.get_state().flatten()
    yield
    sim.set_state_from_flattened(world_state)
    sim.forward()


def all_contacting_geom_ids(sim, geom_id):
    """
    Returns a list of geom ids that are in contact with the target geom id
    """
    contact_gids = []
    for contact in sim.data.contact[:sim.data.ncon]:
        if contact.geom1 == geom_id:
            contact_gids.append(contact.geom2)
        elif contact.geom2 == geom_id:
            contact_gids.append(contact.geom1)
    return contact_gids


def all_contacting_body_ids(sim, body_id):
    """
    Returns a list of body ids that are in contact with the target body id
    """
    contact_gids = []
    for gid in bodyid2geomids(sim, body_id):
        contact_gids += all_contacting_geom_ids(sim, gid)
    contact_bids = list(set([sim.model.geom_bodyid[gid] for gid in contact_gids]))
    return contact_bids


def sample_stable_placement(
        sim,
        top_object,
        top_bodyid,
        surface_object,
        surface_bodyid,
        rotation_range=None,
        max_radius=None,
        num_attempts=1000
):
    """
    Sample a stable pose on top of another object
    Args:
        sim: mujoco sim object
        top_object (MujocoObject): the object to sample pose for
        surface_object (MujocoObject): supporting surface
        rotation_range (int tuple): pair of rotation sampling range.
        max_radius (float): max radius to sample object wrt the center of the surface object, set to None to
            automatically determine the range by the size of the surface object. set to 0 to get center placement
        num_attempts (int): numbers of attempt to sample the placing pose
    Return:
        pos, rot: pose for placing the top object
    """
    assert isinstance(top_object, MujocoObject)
    assert isinstance(surface_object, MujocoObject)

    assert rotation_range is None  # TODO: support this
    assert max_radius == 0  # TODO: support random placement

    suf_pos = sim.data.body_xpos[surface_bodyid]
    bot_quat = sim.data.body_xquat[top_bodyid]
    suf_top = suf_pos + surface_object.get_top_offset()

    top_new_pos = suf_top - top_object.get_bottom_offset()
    return top_new_pos, bot_quat


def is_stable_placement(sim, top_object, top_bodyid, surface_object, surface_bodyid, z_tol=0.01, radius_tol=0.01):
    """
    Check if the top object is placed stably on top of the surface object
    Args:
        top_object (MujocoObject): object on top
        surface_object (MujocoObject): surface object
    Returns:
        (boolean) is stable placement or not
    """
    assert isinstance(top_object, MujocoObject)
    assert isinstance(surface_object, MujocoObject)

    top_pos = sim.data.body_xpos[top_bodyid]
    suf_pos = sim.data.body_xpos[surface_bodyid]

    top_bottom = top_pos + top_object.get_bottom_offset()
    suf_top = suf_pos + surface_object.get_top_offset()

    # check surface distance on z axis
    if top_bottom[-1] - suf_top[-1] < -z_tol or top_bottom[-1] - suf_top[-1] > z_tol:
        return False

    # stable placement if the center of the top object is within the radius of the surface object
    diff = np.linalg.norm(top_pos[:2] - suf_pos[:2])
    suf_radius = surface_object.get_horizontal_radius()
    if diff - suf_radius >= -radius_tol:
        return False

    return True

