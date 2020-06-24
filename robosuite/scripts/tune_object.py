"""
Convenience script to tune a camera view in a mujoco environment.
Allows keyboard presses to move a camera around in the viewer, and
then prints the final position and quaternion you should set
for your camera in the mujoco XML file.
"""

"""
TODOs:
- Document keys
- Fast or slow pos / rot toggle?
"""

import os
import time
import h5py
import argparse
import glfw
import threading
import xml.etree.ElementTree as ET
import numpy as np
from collections import OrderedDict

import robosuite
import robosuite.utils.transform_utils as T

from robosuite.environments.sawyer import SawyerEnv
from robosuite.controllers import load_controller_config
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import TableTopMergedTask, SequentialCompositeSampler
from robosuite.models.objects import BoxObject, CylinderObject, MujocoGeneratedObject, MujocoXMLObject, TestXMLObject
from robosuite.models.base import MujocoXML
from robosuite.utils.mjcf_utils import postprocess_model_xml, new_body, new_geom, \
    new_site, new_joint, array_to_string


# some settings
DELTA_POS_KEY_PRESS = 0.01 # delta position per key press
DELTA_ROT_KEY_PRESS = 5 # delta angle per key press

FINETUNE_POS_KEY_PRESS = 0.001
FINETUNE_ROT_KEY_PRESS = 1

DELTA_POS_CAMERA_KEY_PRESS = 0.05 # delta position per key press
DELTA_ROT_CAMERA_KEY_PRESS = 1 # delta angle per key press

def modify_xml_for_camera_movement(xml_str, camera_name):
    """
    Cameras in mujoco are 'fixed', so they can't be moved by default.
    Although it's possible to hack position movement, rotation movement
    does not work. An alternative is to attach a camera to a mocap body,
    and move the mocap body.
    This function modifies the camera with name @camera_name in the xml
    by attaching it to a mocap body that can move around freely. In this
    way, we can move the camera by moving the mocap body.
    See http://www.mujoco.org/forum/index.php?threads/move-camera.2201/ for
    further details.
    """
    tree = ET.fromstring(xml_str)
    wb = tree.find("worldbody")

    # find the correct camera
    camera_elem = None
    cameras = wb.findall("camera")
    for camera in cameras:
        if camera.get("name") == camera_name:
            camera_elem = camera
            break
    if camera_elem is None:
        print("*" * 50)
        print("WARNING: could not modify camera named '{}' for movement".format(camera_name))
        print("*" * 50)
        return xml_str

    # add mocap body
    mocap = ET.SubElement(wb, "body")
    mocap.set("name", "cameramover_{}".format(camera_name))
    mocap.set("mocap", "true")
    mocap.set("pos", camera_elem.get("pos"))
    mocap.set("quat", camera_elem.get("quat"))
    new_camera = ET.SubElement(mocap, "camera")
    new_camera.set("mode", "fixed")
    new_camera.set("name", camera.get("name"))
    new_camera.set("pos", "0 0 0")

    # remove old camera element
    wb.remove(camera_elem)

    return ET.tostring(tree, encoding="utf8").decode("utf8")

def move_camera(env, direction, scale, camera_id):
    """
    Move the camera view along a direction (in the camera frame).
    :param direction: a 3-dim numpy array for where to move camera in camera frame
    :param scale: a float for how much to move along that direction
    :param camera_id: which camera to modify
    """
    camera_name = env.sim.model.camera_id2name(camera_id)
    mocap_name = "cameramover_{}".format(camera_name)

    # current camera pose
    camera_pos = np.array(env.sim.data.get_mocap_pos(mocap_name))
    camera_rot = T.quat2mat(T.convert_quat(env.sim.data.get_mocap_quat(mocap_name), to='xyzw'))

    # move along camera frame axis and set new position
    camera_pos += scale * camera_rot.dot(direction) 
    env.sim.data.set_mocap_pos(mocap_name, camera_pos)
    env.sim.forward()

def rotate_camera(env, direction, angle, camera_id):
    """
    Rotate the camera view about a direction (in the camera frame).
    :param direction: a 3-dim numpy array for where to move camera in camera frame
    :param angle: a float for how much to rotate about that direction
    :param camera_id: which camera to modify
    """
    camera_name = env.sim.model.camera_id2name(camera_id)
    mocap_name = "cameramover_{}".format(camera_name)

    # current camera rotation
    camera_rot = T.quat2mat(T.convert_quat(env.sim.data.get_mocap_quat(mocap_name), to='xyzw'))

    # rotate by angle and direction to get new camera rotation
    rad = np.pi * angle / 180.0
    R = T.rotation_matrix(rad, direction, point=None)
    camera_rot = camera_rot.dot(R[:3, :3])

    # set new rotation
    env.sim.data.set_mocap_quat(mocap_name, T.convert_quat(T.mat2quat(camera_rot), to='wxyz'))
    env.sim.forward()


class CompositeMujocoObject(MujocoGeneratedObject):
    """
    An object constructed out of basic objects to make more intricate shapes.
    """

    def __init__(
        self,
        mujoco_objects,
        object_poses,
        total_size,
    ):
        """
        Args:
            objects (OrderedDict): ordered dict of (name, MujocoGeneratedObject / MujocoXMLObject instance)

            total_size (list): half-size in each dimension for the bounding box for
                this CompositeBody object

            object_poses (OrderedDict): ordered dict of (name, (pos, quat)) per object relative
                to the center of the bounding box for this composite object
        """
        super().__init__(joint=[], rgba=None)

        assert np.all([
            (isinstance(mujoco_objects[k], MujocoGeneratedObject) or isinstance(mujoco_objects[k], MujocoXMLObject)) 
            for k in mujoco_objects])
        self.mujoco_objects = OrderedDict(mujoco_objects)
        self.total_size = np.array(total_size)

        self.object_poses = OrderedDict(object_poses)
        assert len(self.object_poses) == len(self.mujoco_objects)

    def get_bottom_offset(self):
        return np.array([0., 0., -self.total_size[2]])

    def get_top_offset(self):
        return np.array([0., 0., self.total_size[2]])

    def get_horizontal_radius(self):
        return np.linalg.norm(self.total_size[:2], 2)

    def _get_body(self, name=None, site=None, visual=False):
        main_body = new_body()
        if name is not None:
            main_body.set("name", name)

        # give main body a small mass in order to have a free joint
        inertial = ET.Element("inertial")
        inertial.set("pos", "0 0 0")
        inertial.set("mass", "0.0001")
        inertial.set("diaginertia", "0.0001 0.0001 0.0001")
        main_body.append(inertial)

        for i, k in enumerate(self.mujoco_objects):

            # body name
            if visual:
                body_name = k + "_visual"
            else:
                body_name = k

            # add object body
            if visual:
                obj_body = self.mujoco_objects[k].get_visual(
                    name=body_name,
                    site=site,
                )
            else:
                obj_body = self.mujoco_objects[k].get_collision(
                    name=body_name,
                    site=site,
                )

            # set body pose
            pos, quat = self.object_poses[k]
            quat = T.convert_quat(quat, to="wxyz")
            obj_body.set("pos", array_to_string(pos))
            obj_body.set("quat", array_to_string(quat))

            # # add object joints
            # for j, joint in enumerate(self.mujoco_objects[k].joint):
            #     obj_body.append(new_joint(name="{}_{}".format(body_name, j), **joint))
            main_body.append(obj_body)

        # add site if requested
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            template["rgba"] = "1 0 0 0"
            if name is not None:
                template["name"] = name
            main_body.append(ET.Element("site", attrib=template))

        return main_body

    def get_collision(self, name=None, site=None):
        return self._get_body(name=name, site=site, visual=False)

    def get_visual(self, name=None, site=None):
        return self._get_body(name=name, site=site, visual=True)


class SawyerWorkspace(SawyerEnv):
    """
    This is a helper environment to play with object placements.
    """

    def __init__(
        self,
        mujoco_objects,
        ckpt_path,
        object_xml_path,
        controller_config=None,
        gripper_type="TwoFingerGripper",
        table_full_size=(0.8, 0.8, 0.8),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        gripper_visualization=False,
        use_indicator_object=False,
        indicator_num=1,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=1000,
        ignore_done=False,
        camera_name="frontview",
        camera_height=256,
        camera_width=256,
        camera_depth=False,
        camera_real_depth=False,
        camera_segmentation=False,
    ):
        """
        NOTE: the first item in @mujoco_objects is treated as the reference object and is not
              included in the dumped object xml
        """

        # Load the default controller if none is specified
        if controller_config is None:
            controller_path = os.path.join(os.path.dirname(__file__), '..', 'controllers/config/default_sawyer.json')
            controller_config = load_controller_config(custom_fpath=controller_path)

        # Assert that the controller config is a dict file
        assert type(controller_config) == dict, \
            "Inputted controller config must be a dict! Instead, got type: {}".format(type(controller_config))

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        # mujoco objects to load
        self.mujoco_objects = OrderedDict(mujoco_objects)
        self.object_id_to_name = list(self.mujoco_objects.keys())
        self.object_name_to_id = {}
        for i, obj_name in enumerate(self.mujoco_objects):
            self.object_name_to_id[obj_name] = i

        # partition into reference object and other objects
        ref_object_name = self.object_id_to_name[0]
        self.ref_object = OrderedDict({ ref_object_name : self.mujoco_objects[ref_object_name] })
        self.other_objects = OrderedDict({ k : self.mujoco_objects[k] for k in self.object_id_to_name[1:]})

        # merge assets and compute relative poses
        objects_to_add = { k : self.mujoco_objects[k] for k in self.object_id_to_name[1:]}

        # filepaths for saving workspace checkpoints and object xmls
        self.ckpt_path = ckpt_path
        self.object_xml_path = object_xml_path

        super().__init__(
            controller_config=controller_config,
            gripper_type=gripper_type,
            gripper_visualization=gripper_visualization,
            use_indicator_object=use_indicator_object,
            indicator_num=indicator_num,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            use_camera_obs=use_camera_obs,
            camera_name=camera_name,
            camera_height=camera_height,
            camera_width=camera_width,
            camera_depth=camera_depth,
            camera_real_depth=camera_real_depth,
            camera_segmentation=camera_segmentation,
            eval_mode=False,
            perturb_evals=False,
        )

    def _get_default_initializer(self):
        initializer = SequentialCompositeSampler()

        # reference object should be at center of table
        ref_object_name = self.object_id_to_name[0]
        initializer.sample_on_top(
            ref_object_name,
            surface_name="table",
            x_range=(0.0, 0.0),
            y_range=(0.0, 0.0),
            z_rotation=(0.0, 0.0),
            ensure_object_boundary_in_range=False,
        )

        # initialize other objects randomly on tabletop
        for obj_name in self.object_id_to_name[1:]:
            initializer.sample_on_top(
                obj_name,
                surface_name="table",
                x_range=(-self.table_full_size[0] / 2., self.table_full_size[0] / 2.),
                y_range=(-self.table_full_size[1] / 2., self.table_full_size[1] / 2.),
                z_rotation=(0.0, 0.0),
                ensure_object_boundary_in_range=False,
            )
        return initializer

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        # load model for table top workspace
        self.mujoco_arena = TableArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator(self.indicator_num)

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_full_size[0] / 2, 0, 0])

        # reset initial joint positions (gets reset in sim during super() call in _reset_internal)
        self.init_qpos = np.array([0.00, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])

        # ensure that all objects have free joints
        for obj_name in self.mujoco_objects:
            self.mujoco_objects[obj_name].joint = [{'type': 'free'}]

        # object placement initializer
        self.placement_initializer = self._get_default_initializer()

        # task includes arena, robot, and objects of interest
        self.model = TableTopMergedTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robot=self.mujoco_robot,
            mujoco_objects=self.other_objects,
            visual_objects=self.ref_object,
            initializer=self.placement_initializer,
        )
        self.model.place_objects()

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()
        self.object_body_ids = { k : self.sim.model.body_name2id(k) for k in self.mujoco_objects }
        self.object_qpos_addrs = { k : self.sim.model.get_joint_qpos_addr(k + '_0') for k in self.mujoco_objects }
        self.object_qvel_addrs = { k : self.sim.model.get_joint_qvel_addr(k + '_0') for k in self.mujoco_objects }

    def _pre_action(self, action, policy_step=None):
        result = super()._pre_action(action, policy_step=policy_step)

        # gravity compensation for all objects
        for obj_name in self.object_qvel_addrs:
            self.sim.data.qfrc_applied[
                    self.object_qvel_addrs[obj_name][0] : self.object_qvel_addrs[obj_name][1]
                ] = self.sim.data.qfrc_bias[
                    self.object_qvel_addrs[obj_name][0] : self.object_qvel_addrs[obj_name][1]
                ]

            # set object velocities to 0
            joint_id = self.object_qvel_addrs[obj_name]
            self.sim.data.qvel[joint_id[0] : joint_id[1]] = 0.

        return result

    def save_object_xml(self):
        """
        Uses current object location to return an xml corresponding to a new
        composite object composed of all objects. The first item of 
        @mujoco_objects is used as the reference object - all object poses
        are relative to this object, and the object is not included in the
        composite.
        """

        # make a temporary xml to instantiate MujocoXML object
        f = open(self.object_xml_path, "w")
        tmp = ET.Element("mujoco")
        tmp.set("model", "object")
        f.write(ET.tostring(tmp, encoding="utf8").decode("utf8"))
        f.close()

        # initialize composite object
        composite = MujocoXML(self.object_xml_path)

        # reference object used to compute relative poses
        ref_object_name = self.object_id_to_name[0]
        ref_object = self.mujoco_objects[ref_object_name]
        ref_bbox = ref_object.get_bounding_box_size()

        # merge assets and compute relative poses
        relative_object_poses = OrderedDict()
        for obj_name in self.other_objects:
            obj_mjcf = self.mujoco_objects[obj_name]
            composite.merge_asset(obj_mjcf)

            # relative pose to reference frame will become poses for body components
            rel_pos, rel_rot = get_relative_object_pose(env=env, object_name=obj_name, ref_object_name=ref_object_name)
            rel_quat = T.mat2quat(rel_rot)
            relative_object_poses[obj_name] = (rel_pos, rel_quat)

        # create main body
        main_body = new_body()
        main_body.set("name", "object")

        # this is used to get the collision and visual bodies
        comp_obj = CompositeMujocoObject(
            mujoco_objects=self.other_objects,
            object_poses=relative_object_poses,
            total_size=ref_bbox,
        )

        # add collision and visual bodies
        main_body.append(comp_obj.get_collision(name="collision"))
        main_body.append(comp_obj.get_visual(name="visual"))

        # add sites (used by placement initializer)
        # reference object bounding box used to determine sites
        main_body.append(
            new_site(name="bottom_site", 
                rgba=(0, 0, 0, 0), 
                pos=(0, 0, -ref_bbox[2]), 
                size=(0.005,),
            )
        )
        main_body.append(
            new_site(name="top_site", 
                rgba=(0, 0, 0, 0), 
                pos=(0, 0, ref_bbox[2]), 
                size=(0.005,),
            )
        )
        main_body.append(
            new_site(name="horizontal_radius_site", 
                rgba=(0, 0, 0, 0), 
                pos=(ref_bbox[0], ref_bbox[1], 0), 
                size=(0.005,),
            )
        )

        # merge in main body
        composite.worldbody.append(main_body)

        # save model xml
        composite.save_model(self.object_xml_path, pretty=True)

    def save_ckpt(self):
        """
        Helper method to save the state of the workspace to a file.
        """
        f = h5py.File(self.ckpt_path, "w")
        grp = f.create_group("data")

        # save mujoco state
        state = self.sim.get_state().flatten()
        grp.create_dataset("state", data=np.array(state))

        # save model xml
        xml_str = self.model.get_xml()
        grp.attrs["xml"] = xml_str

        f.close()

    def restore_from_ckpt(self, ckpt):
        """
        Helper method to restore the state of the workspace from a file.
        """
        f = h5py.File(ckpt, "r")
        xml = postprocess_model_xml(f["data"].attrs["xml"])
        state = np.array(f["data/state"][()])
        f.close()

        self.reset_from_xml_string(xml)
        self.sim.reset()
        self.sim.set_state_from_flattened(state)
        self.sim.forward()


def get_object_pose(env, object_name=None, object_id=None):
    """
    Helper function to get object pos and rot from environment.
    """
    assert (object_name is not None) or (object_id is not None)
    if object_name is None:
        object_name = env.object_id_to_name[object_id]
    body_id = env.object_body_ids[object_name]
    obj_pos = np.array(env.sim.data.body_xpos[body_id])
    obj_rot = T.quat2mat(T.convert_quat(env.sim.data.body_xquat[body_id], to='xyzw'))
    return (obj_pos, obj_rot)

def set_object_pose(env, object_name=None, object_id=None, pos=None, rot=None):
    """
    Helper function to set object pos and rot in environment.
    """
    assert (object_name is not None) or (object_id is not None)
    if object_name is None:
        object_name = env.object_id_to_name[object_id]
    joint_id = env.object_qpos_addrs[object_name][0]

    assert (pos is not None) or (rot is not None)
    if pos is not None:
        env.sim.data.qpos[joint_id: joint_id + 3] = np.array(pos)
    if rot is not None:
        env.sim.data.qpos[joint_id + 3: joint_id + 7] = T.convert_quat(T.mat2quat(rot), to='wxyz')
    env.sim.forward()

def get_relative_object_pose(env, object_name=None, object_id=None, ref_object_name=None, ref_object_id=None):
    """
    Helper function to get relative pose of one object with respect to another object.
    """
    pos1, rot1 = get_object_pose(env=env, object_name=object_name, object_id=object_id)
    pos2, rot2 = get_object_pose(env=env, object_name=ref_object_name, object_id=ref_object_id)
    return relative_pose(env=env, pos1=pos1, rot1=rot1, pos2=pos2, rot2=rot2)

def relative_pose(env, pos1, rot1, pos2, rot2):
    """
    Helper function to compute relative pose of (pos1, rot1) with respect to (pos2, rot2).
    """
    ref_pose = T.make_pose(pos2, rot2)
    inv_ref_pose = T.pose_inv(ref_pose)
    pose = T.make_pose(pos1, rot1)
    rel_pose = T.pose_in_A_to_pose_in_B(pose, inv_ref_pose)
    return rel_pose[:3, 3], rel_pose[:3, :3]

def move_object(env, direction, scale, object_id):
    """
    Move object position in world frame.

    Args:
        direction: a 3-dim numpy array for where to move object
        scale: a float for how much to move along that direction
        object_id: which object to modify
    """
    obj_pos, _ = get_object_pose(env=env, object_id=object_id)
    obj_pos += scale * np.array(direction)
    set_object_pose(env=env, object_id=object_id, pos=obj_pos)

def rotate_object(env, direction, angle, object_id):
    """
    Rotate object about a direction in object frame.

    Args:
        direction: a 3-dim numpy array for where to rotate object in object frame
        angle: a float for how much to rotate about that direction
        object_id: which camera to modify
    """
    _, obj_rot = get_object_pose(env=env, object_id=object_id)

    # compute new rotation
    rad = np.pi * angle / 180.0
    R = T.rotation_matrix(rad, np.array(direction), point=None)
    obj_rot = R[:3, :3].T.dot(obj_rot)

    set_object_pose(env=env, object_id=object_id, rot=obj_rot)

def snap_object_to_center(env, object_id, other_object_id):
    """
    Helper function to snap object to the center of another object.
    """
    # set new object pos to center of other object
    other_obj_pos, _ = get_object_pose(env=env, object_id=other_object_id)
    set_object_pose(env=env, object_id=object_id, pos=other_obj_pos)

def snap_object_to_table(env, object_id):
    """
    Helper function to snap an object to the table surface.
    """

    # use object id to get name, object, and position
    obj_name = env.object_id_to_name[object_id]
    obj = env.mujoco_objects[obj_name]
    obj_pos, _ = get_object_pose(env=env, object_name=obj_name)

    # change object z-position to lie directly on top of table
    table_body_id = env.sim.model.body_name2id("table")
    table_pos = np.array(env.sim.data.body_xpos[table_body_id])
    offset = (env.table_full_size[2] / 2.) - obj.get_bottom_offset()[2]
    obj_pos[2] = table_pos[2] + offset

    # set new object pos
    set_object_pose(env=env, object_name=obj_name, pos=obj_pos)

def snap_object_to_boundary(env, object_id, axis):
    """
    Depending on axis input, this function will move to align the boundary of the object
    with the boundary of the other object in that axis. The other object is determined
    to be the one closest to the current object, along the relevant axis.
    """
    # use object id to get name, object, and position
    obj_name = env.object_id_to_name[object_id]
    obj = env.mujoco_objects[obj_name]
    obj_pos, _ = get_object_pose(env=env, object_name=obj_name)

    best_dist = np.inf
    best_obj_pos = None
    for other_obj_name in env.mujoco_objects:
        if other_obj_name == obj_name:
            continue
        other_obj = env.mujoco_objects[other_obj_name]
        other_body_id = env.object_body_ids[other_obj_name]
        other_obj_pos = np.array(env.sim.data.body_xpos[other_body_id])
        new_obj_pos = np.array(obj_pos)

        if axis == 'x':
            # move object to left / right of x-boundary depending on distance
            offset = obj.get_horizontal_radius() + other_obj.get_horizontal_radius()
            if np.abs(obj_pos[0] - other_obj_pos[0] - offset) < np.abs(obj_pos[0] - other_obj_pos[0] + offset):
                dist = np.abs(obj_pos[0] - other_obj_pos[0] - offset)
                new_obj_pos[0] = other_obj_pos[0] + offset
            else:
                dist = np.abs(obj_pos[0] - other_obj_pos[0] + offset)
                new_obj_pos[0] = other_obj_pos[0] - offset
        elif axis == 'y':
            # move object to left / right of y-boundary depending on distance
            offset = obj.get_horizontal_radius() + other_obj.get_horizontal_radius()
            if np.abs(obj_pos[1] - other_obj_pos[1] - offset) < np.abs(obj_pos[1] - other_obj_pos[1] + offset):
                dist = np.abs(obj_pos[1] - other_obj_pos[1] - offset)
                new_obj_pos[1] = other_obj_pos[1] + offset
            else:
                dist = np.abs(obj_pos[1] - other_obj_pos[1] + offset)
                new_obj_pos[1] = other_obj_pos[1] - offset
        elif axis == 'z':
            offset = other_obj.get_top_offset()[2] - obj.get_bottom_offset()[2]
            dist = np.abs(obj_pos[2] - other_obj_pos[2] - offset)
            new_obj_pos[2] = other_obj_pos[2] + offset
        else:
            raise Exception("Invalid axis.")

        if dist < best_dist:
            best_dist = dist
            best_obj_pos = new_obj_pos

    # set new object pos
    set_object_pose(env=env, object_name=obj_name, pos=best_obj_pos)

def snap_object_to_grid(env, object_id, rotation_set):
    """
    Helper function to snap object to rotation grid.
    """
    # get closest rotation and set it
    _, obj_rot = get_object_pose(env=env, object_id=object_id)
    obj_rot = get_closest_rotation(obj_rot, rotation_set)
    set_object_pose(env=env, object_id=object_id, rot=obj_rot)

def get_axis_aligned_rotations():
    """
    Helper function to get a set of axis-aligned rotation matrices, used to enable
    projecting an object's rotation to one of these matrices.
    """
    # angle spacing of 10 degrees in each dimension
    spacing = 10
    num = (360 // 10) + 1
    angles = np.linspace(0, 360, num)[:-1]
    all_rotations = []
    for i in range(len(angles)):
        rad = np.pi * angles[i] / 180.0
        all_rotations.append(T.rotation_matrix(rad, [1., 0., 0.], point=None)[:3, :3])
        all_rotations.append(T.rotation_matrix(rad, [0., 1., 0.], point=None)[:3, :3])
        all_rotations.append(T.rotation_matrix(rad, [0., 0., 1.], point=None)[:3, :3])
    return all_rotations

def get_closest_rotation(rotation, rotation_set):
    """
    Projects @rotation onto the set of rotations in @rotation_set by finding
    the rotation in the set that minimizes rotation distance (as measured
    by the absolute angle of the axis-angle representaion of the delta
    rotation).
    """
    dists = []
    for i in range(len(rotation_set)):
        delta_rotation = rotation_set[i].T.dot(rotation)
        _, angle = T.quat2axisangle(T.mat2quat(delta_rotation))
        dists.append(np.abs(angle))
    min_ind = np.argmin(dists)
    return rotation_set[min_ind]

class KeyboardHandler:
    def __init__(self, env):
        """
        Store internal state here.
        """
        self.env = env
        self.object_id = 0
        self.num_objects = len(self.env.object_id_to_name)
        self.camera_id = self.env.sim.model.camera_name2id("frontview")
        self.num_cameras = len(self.env.sim.model.camera_names)
        self.camera_mode = False
        self.finetune_mode = False
        self.measurement_mode = False
        self.measurement_pose = [None, None]
        self.object_pos_scale = DELTA_POS_KEY_PRESS
        self.object_rot_scale = DELTA_ROT_KEY_PRESS

        # used for snapping to rotation grid
        self._rotation_grid = get_axis_aligned_rotations()

        # store pressed keys
        self._keys = []
        self._key_lock = threading.Lock()

        # register callbacks to handle key presses in the viewer
        self.env.viewer.add_keypress_callback("any", self.on_press)
        self.env.viewer.add_keyup_callback("any", self.on_release)
        self.env.viewer.add_keyrepeat_callback("any", self.on_press)

    def dequeue(self):
        ret = None
        with self._key_lock:
            if len(self._keys) > 0:
                ret = self._keys.pop(0)
        return ret

    def handle_key(self, key):
        with self._key_lock:
            # toggle object movement mode and camera movement mode
            if key == glfw.KEY_Q:
                self.camera_mode = not self.camera_mode

            # finetune mode for less sensitive movement
            if key == glfw.KEY_1:
                if self.camera_mode:
                    pass
                else:
                    self.finetune_mode = not self.finetune_mode
                    if self.finetune_mode:
                        self.object_pos_scale = FINETUNE_POS_KEY_PRESS
                        self.object_rot_scale = FINETUNE_ROT_KEY_PRESS
                    else:
                        self.object_pos_scale = DELTA_POS_KEY_PRESS
                        self.object_rot_scale = DELTA_ROT_KEY_PRESS

            # make a measurement
            if key == glfw.KEY_Z:
                if self.camera_mode:
                    pass
                else:
                    if self.measurement_mode:
                        # print relative pose
                        new_pos, new_rot = get_object_pose(env=self.env, object_id=self.object_id)
                        rel_pose = relative_pose(
                            env=self.env, 
                            pos1=new_pos, 
                            rot1=new_rot, 
                            pos2=self.measurement_pose[0], 
                            rot2=self.measurement_pose[1],
                        )
                        rel_quat = T.convert_quat(T.mat2quat(rel_pose[1]), to="wxyz")
                        print("measured relative position: {}".format(rel_pose[0]))
                        print("measured relative quaternion: {}\n".format(rel_quat))
                        # clear recorded pose
                        self.measurement_pose = (None, None)
                    else:
                        # record current object pose as reference
                        print("\nstarting measurement...")
                        self.measurement_pose = get_object_pose(env=self.env, object_id=self.object_id)

                    # toggle mode to indicate completion
                    self.measurement_mode = not self.measurement_mode

            # switch camera / object
            if key == glfw.KEY_TAB:
                if self.camera_mode:
                    self.camera_id = (self.camera_id + 1) % self.num_cameras
                    self.env.viewer.set_camera(camera_id=self.camera_id)
                else:
                    self.object_id = (self.object_id + 1) % self.num_objects


            # save model
            if key == glfw.KEY_SPACE:
                if self.camera_mode:
                    pass
                else:
                    print("Saving current model")
                    env.save_ckpt()
                    env.save_object_xml()


            # snap current rotation to grid
            if key == glfw.KEY_E:
                if self.camera_mode:
                    pass
                else:
                    snap_object_to_grid(env=self.env, object_id=self.object_id, rotation_set=self._rotation_grid)


            # snap object to reference object center
            if key == glfw.KEY_G:
                if self.camera_mode:
                    pass
                else:
                    snap_object_to_center(env=self.env, object_id=self.object_id, other_object_id=0)


            # snap object to tabletop
            if key == glfw.KEY_H:
                if self.camera_mode:
                    pass
                else:
                    snap_object_to_table(env=self.env, object_id=self.object_id)


            # snap object to closest object boundary
            if key == glfw.KEY_T:
                if self.camera_mode:
                    pass
                else:
                    snap_object_to_boundary(env=self.env, object_id=self.object_id, axis='x')

            if key == glfw.KEY_Y:
                if self.camera_mode:
                    pass
                else:
                    snap_object_to_boundary(env=self.env, object_id=self.object_id, axis='y')

            if key == glfw.KEY_U:
                if self.camera_mode:
                    pass
                else:
                    snap_object_to_boundary(env=self.env, object_id=self.object_id, axis='z')


            # controls for moving position
            if key == glfw.KEY_W:
                if self.camera_mode:
                    # move camera forward
                    move_camera(env=self.env, direction=[0., 0., -1.], scale=DELTA_POS_CAMERA_KEY_PRESS, camera_id=self.camera_id)
                else:
                    # move -x
                    move_object(env=self.env, direction=[-1., 0., 0.], scale=self.object_pos_scale, object_id=self.object_id)
            elif key == glfw.KEY_S:
                if self.camera_mode:
                    # move camera backward
                    move_camera(env=self.env, direction=[0., 0., 1.], scale=DELTA_POS_CAMERA_KEY_PRESS, camera_id=self.camera_id)
                else:
                    # move x
                    move_object(env=self.env, direction=[1., 0., 0.], scale=self.object_pos_scale, object_id=self.object_id)
            elif key == glfw.KEY_A:
                if self.camera_mode:
                    # move camera left
                    move_camera(env=self.env, direction=[-1., 0., 0.], scale=DELTA_POS_CAMERA_KEY_PRESS, camera_id=self.camera_id)
                else:
                    # move -y
                    move_object(env=self.env, direction=[0., -1., 0.], scale=self.object_pos_scale, object_id=self.object_id)
            elif key == glfw.KEY_D:
                if self.camera_mode:
                    # move camera right
                    move_camera(env=self.env, direction=[1., 0., 0.], scale=DELTA_POS_CAMERA_KEY_PRESS, camera_id=self.camera_id)
                else:
                    # move y
                    move_object(env=self.env, direction=[0., 1., 0.], scale=self.object_pos_scale, object_id=self.object_id)
            elif key == glfw.KEY_R:
                if self.camera_mode:
                    # move camera up
                    move_camera(env=self.env, direction=[0., 1., 0.], scale=DELTA_POS_CAMERA_KEY_PRESS, camera_id=self.camera_id)
                else:
                    # move z
                    move_object(env=self.env, direction=[0., 0., 1.], scale=self.object_pos_scale, object_id=self.object_id)
            elif key == glfw.KEY_F:
                if self.camera_mode:
                    # move camera down
                    move_camera(env=self.env, direction=[0., -1., 0.], scale=DELTA_POS_CAMERA_KEY_PRESS, camera_id=self.camera_id)
                else:
                    # move -z
                    move_object(env=self.env, direction=[0., 0., -1.], scale=self.object_pos_scale, object_id=self.object_id)


            # controls for moving rotation
            elif key == glfw.KEY_UP:
                if self.camera_mode:
                    # rotate camera up
                    rotate_camera(env=self.env, direction=[1., 0., 0.], angle=DELTA_ROT_CAMERA_KEY_PRESS, camera_id=self.camera_id)
                else:
                    # rotate y
                    rotate_object(env=self.env, direction=[0., 1., 0.], angle=self.object_rot_scale, object_id=self.object_id)
            elif key == glfw.KEY_DOWN:
                if self.camera_mode:
                    # rotate camera down
                    rotate_camera(env=self.env, direction=[-1., 0., 0.], angle=DELTA_ROT_CAMERA_KEY_PRESS, camera_id=self.camera_id)
                else:
                    # rotate y
                    rotate_object(env=self.env, direction=[0., -1., 0.], angle=self.object_rot_scale, object_id=self.object_id)
            elif key == glfw.KEY_LEFT:
                if self.camera_mode:
                    # rotate camera left
                    rotate_camera(env=self.env, direction=[0., 1., 0.], angle=DELTA_ROT_CAMERA_KEY_PRESS, camera_id=self.camera_id)
                else:
                    # rotate x
                    rotate_object(env=self.env, direction=[-1., 0., 0.], angle=self.object_rot_scale, object_id=self.object_id)
            elif key == glfw.KEY_RIGHT:
                if self.camera_mode:
                    # rotate camera right
                    rotate_camera(env=self.env, direction=[0., -1., 0.], angle=DELTA_ROT_CAMERA_KEY_PRESS, camera_id=self.camera_id)
                else:
                    # rotate x
                    rotate_object(env=self.env, direction=[1., 0., 0.], angle=self.object_rot_scale, object_id=self.object_id)
            elif key == glfw.KEY_PERIOD:
                if self.camera_mode:
                    # rotate camera counterclockwise
                    rotate_camera(env=self.env, direction=[0., 0., 1.], angle=DELTA_ROT_CAMERA_KEY_PRESS, camera_id=self.camera_id)
                else:
                    # rotate z
                    rotate_object(env=self.env, direction=[0., 0., -1.], angle=self.object_rot_scale, object_id=self.object_id)
            elif key == glfw.KEY_SLASH:
                if self.camera_mode:
                    # rotate camera clockwise
                    rotate_camera(env=self.env, direction=[0., 0., -1.], angle=DELTA_ROT_CAMERA_KEY_PRESS, camera_id=self.camera_id)
                else:
                    # rotate z
                    rotate_object(env=self.env, direction=[0., 0., 1.], angle=self.object_rot_scale, object_id=self.object_id)


    def on_press(self, window, key, scancode, action, mods):
        """
        Key handler for key presses.
        """
        with self._key_lock:
            self._keys.append(key)

    def on_release(self, window, key, scancode, action, mods):
        """
        Key handler for key releases.
        """
        pass

def print_command(char, info):
    char += " " * (10 - len(char))
    print("{}\t{}".format(char, info))

def reset_env(env, has_reset=False):
    """
    Helper function to modify the xml to control cameras and add viewer callbacks.
    """
    if not has_reset:
        env.reset()
        initial_mjstate = env.sim.get_state().flatten()
        xml_str = env.model.get_xml()

        # add mocap body to all cameras to be able to move it around
        for camera_name in env.sim.model.camera_names:
            xml_str = modify_xml_for_camera_movement(xml_str, camera_name=camera_name)
        env.reset_from_xml_string(xml_str)
        env.sim.reset()
        env.sim.set_state_from_flattened(initial_mjstate)
        env.sim.forward()

    camera_id = env.sim.model.camera_name2id("frontview")
    env.viewer.set_camera(camera_id=camera_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # load from ckpt
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
    )
    # path to checkpoint file to save
    parser.add_argument(
        "--save_ckpt",
        type=str,
        default="workspace.hdf5",
    )
    # path to object xml to save
    parser.add_argument(
        "--xml",
        type=str,
        default="object.xml",
    )
    args = parser.parse_args()

    print("\nWelcome to the object creation script! You will be able to make a composite object")
    print("by moving individual pieces around using your keyboard. The controls are below.")
    print("You can control the objects, or move cameras around to get a better view.")
    print("The first object is treated as the workspace for crafting the composite object.")

    print("")
    print("General Controls\n")
    print_command("q", "toggle camera mode on/off")
    print_command("SPACE", "save current workspace and composite object to hdf5 and xml")

    print("\nObject Controls\n")
    print_command("Keys", "Command")
    print_command("TAB", "switch active object")
    print_command("1", "toggle precise movement mode")
    print_command("w-s", "move object along x-axis")
    print_command("a-d", "move object along y-axis")
    print_command("r-f", "move object along z-axis")
    print_command("arrow keys", "rotate object about x and y axis")
    print_command(".-/", "rotate object about z-axis")
    print_command("t-y-u", "snap object to closest object boundary in x, y, and z")
    print_command("e", "snap object rotation to closest axis-aligned rotation")
    print_command("g", "snap object to center of reference workspace")
    print_command("h", "snap object to tabletop surface")
    print_command("z", "make a measurement: press once to record initial pose and again to print relative pose")

    print("\nCamera Controls\n")
    print_command("Keys", "Command")
    print_command("TAB", "switch active camera")
    print_command("w-s", "zoom the camera in/out")
    print_command("a-d", "pan the camera left/right")
    print_command("r-f", "pan the camera up/down")
    print_command("arrow keys", "rotate the camera to change view direction")
    print_command(".-/", "rotate the camera view without changing view direction")
    print("")

    # mujoco objects
    mujoco_objects = OrderedDict()

    # first object is the bounding box workspace for creating the compositional object
    mujoco_objects["ref"] = BoxObject(
        size=[0.1, 0.15, 0.125],
        rgba=[0, 1, 0, 0.1],
    )

    # from robosuite.models.objects import CoffeeMachineBodyObject, CoffeeMachineLidObject, CoffeeMachineBaseObject, CoffeeMachinePodObject
    # mujoco_objects["coffee_body"] = CoffeeMachineBodyObject()
    # mujoco_objects["coffee_lid"] = CoffeeMachineLidObject()
    # mujoco_objects["coffee_base"] = CoffeeMachineBaseObject()

    # from robosuite.models.objects import CupObject
    # mujoco_objects["pod_holder"] = CupObject(
    #     outer_cup_radius=0.03,
    #     inner_cup_radius=0.025,
    #     cup_height=0.025,
    #     cup_ngeoms=64,#8,
    #     cup_base_height=0.005,
    #     cup_base_offset=0.005,
    #     add_handle=False,
    #     rgba=[1, 0, 0, 1],
    #     density=100.,
    # )

    # mujoco_objects["pod_holder_holder"] = BoxObject(
    #     size=[0.01, 0.02, 0.005],
    #     rgba=[0.514, 0.286, 0.204, 1], # brown
    # )

    from robosuite.models.objects import CoffeeMachineObject2
    mujoco_objects["coffee_machine"] = CoffeeMachineObject2()


    # next are the individual pieces that should be rearranged in the workspace
    # mujoco_objects["cube"] = BoxObject(
    #     size=[0.02, 0.02, 0.02],
    #     rgba=[1, 0, 0, 1],
    # )
    # mujoco_objects["cylinder"] = CylinderObject(
    #     size=[0.02, 0.02],
    #     rgba=[0, 0, 1, 1],
    # )

    # from robosuite.models.objects import CupObject
    # mujoco_objects["cup"] = CupObject(
    #     outer_cup_radius=0.02,
    #     inner_cup_radius=0.015,
    #     cup_height=0.02,
    #     cup_ngeoms=64,#8,
    #     cup_base_height=0.005,
    #     cup_base_offset=0.005,
    #     add_handle=True,
    #     handle_outer_radius=0.0115,
    #     handle_inner_radius=0.009,
    #     handle_thickness=0.0025,
    #     handle_ngeoms=64,
    #     rgba=[1, 0, 0, 1],
    #     density=100.,
    # )

    # from robosuite.models.objects import CoffeePodObject
    # mujoco_objects["pod"] = CoffeePodObject(
    #     lid_radius=0.02,
    #     lid_height=0.001,
    #     lid_rgba=[1, 0, 0, 1],
    #     pod_radius=0.015,
    #     pod_height=0.015,
    #     pod_rgba=[1, 0, 0, 1],
    #     density=100.,
    # )

    # from robosuite.models.objects import CoffeeMachineObject
    # mujoco_objects["coffee"] = CoffeeMachineObject()

    # mujoco_objects["test12"] = TestXMLObject()

    # make the environment and key handler
    env = robosuite.make(
        "SawyerWorkspace",
        mujoco_objects=mujoco_objects,
        ckpt_path=args.save_ckpt,
        object_xml_path=args.xml,
        has_renderer=True,
        ignore_done=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=100,
    )

    # load ckpt if necessary
    load_ckpt = (args.ckpt is not None)
    if load_ckpt:
        env.restore_from_ckpt(args.ckpt)

    # prepare env for keyboard interaction
    reset_env(env=env, has_reset=load_ckpt)

    # make key handler
    key_handler = KeyboardHandler(env=env)

    # just spin to let user interact with glfw window
    spin_count = 0
    while True:
        action = np.zeros(env.dof)
        obs, reward, done, _ = env.step(action)
        env.render()

        # handle key presses
        key = key_handler.dequeue()
        if key is not None:
            key_handler.handle_key(key)

        spin_count += 1
        if spin_count % 500 == 0:
            pass


