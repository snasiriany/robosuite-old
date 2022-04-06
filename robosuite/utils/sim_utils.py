"""
Collection of useful simulation utilities
"""
import os
import numpy as np
from tempfile import TemporaryDirectory

import mujoco

from robosuite.models.base import MujocoModel


def check_contact(sim, geoms_1, geoms_2=None):
    """
    Finds contact between two geom groups.

    Args:
        sim (MjSim): Current simulation object
        geoms_1 (str or list of str or MujocoModel): an individual geom name or list of geom names or a model. If
            a MujocoModel is specified, the geoms checked will be its contact_geoms
        geoms_2 (str or list of str or MujocoModel or None): another individual geom name or list of geom names.
            If a MujocoModel is specified, the geoms checked will be its contact_geoms. If None, will check
            any collision with @geoms_1 to any other geom in the environment

    Returns:
        bool: True if any geom in @geoms_1 is in contact with any geom in @geoms_2.
    """
    # Check if either geoms_1 or geoms_2 is a string, convert to list if so
    if type(geoms_1) is str:
        geoms_1 = [geoms_1]
    elif isinstance(geoms_1, MujocoModel):
        geoms_1 = geoms_1.contact_geoms
    if type(geoms_2) is str:
        geoms_2 = [geoms_2]
    elif isinstance(geoms_2, MujocoModel):
        geoms_2 = geoms_2.contact_geoms
    for contact in sim.data.contact[: sim.data.ncon]:
        # check contact geom in geoms
        c1_in_g1 = sim.model.geom_id2name(contact.geom1) in geoms_1
        c2_in_g2 = sim.model.geom_id2name(contact.geom2) in geoms_2 if geoms_2 is not None else True
        # check contact geom in geoms (flipped)
        c2_in_g1 = sim.model.geom_id2name(contact.geom2) in geoms_1
        c1_in_g2 = sim.model.geom_id2name(contact.geom1) in geoms_2 if geoms_2 is not None else True
        if (c1_in_g1 and c2_in_g2) or (c1_in_g2 and c2_in_g1):
            return True
    return False


def get_contacts(sim, model):
    """
    Checks for any contacts with @model (as defined by @model's contact_geoms) and returns the set of
    geom names currently in contact with that model (excluding the geoms that are part of the model itself).

    Args:
        sim (MjSim): Current simulation model
        model (MujocoModel): Model to check contacts for.

    Returns:
        set: Unique geoms that are actively in contact with this model.

    Raises:
        AssertionError: [Invalid input type]
    """
    # Make sure model is MujocoModel type
    assert isinstance(model, MujocoModel), \
        "Inputted model must be of type MujocoModel; got type {} instead!".format(type(model))
    contact_set = set()
    for contact in sim.data.contact[: sim.data.ncon]:
        # check contact geom in geoms; add to contact set if match is found
        g1, g2 = sim.model.geom_id2name(contact.geom1), sim.model.geom_id2name(contact.geom2)
        if g1 in model.contact_geoms and g2 not in model.contact_geoms:
            contact_set.add(g2)
        elif g2 in model.contact_geoms and g1 not in model.contact_geoms:
            contact_set.add(g1)
    return contact_set


class MjSimState:
    """
    A mujoco simulation state.
    """
    def __init__(self, time, qpos, qvel):
        self.time = time
        self.qpos = qpos
        self.qvel = qvel

    @classmethod
    def from_flattened(cls, array, sim):
        """
        Takes flat mjstate array and MjSim instance and 
        returns MjSimState.
        """
        idx_time = 0
        idx_qpos = idx_time + 1
        idx_qvel = idx_qpos + sim.model.nq

        time = array[idx_time]
        qpos = array[idx_qpos:idx_qpos + sim.model.nq]
        qvel = array[idx_qvel:idx_qvel + sim.model.nv]
        assert sim.model.na == 0

        return cls(time=time, qpos=qpos, qvel=qvel)

    def flatten(self):
        return np.concatenate([[self.time], self.qpos, self.qvel], axis=0)


class MjSim:
    """
    Meant to somewhat replicate functionality in mujoco-py's MjSim object
    (see https://github.com/openai/mujoco-py/blob/master/mujoco_py/mjsim.pyx).
    """
    def __init__(self, model):
        """
        Args:
            model: should be an MjModel instance created via a factory function
                such as mujoco.MjModel.from_xml_string(xml)
        """
        self.model = model
        self.data = mujoco.MjData(model)

        # make useful mappings such as _body_name2id and _body_id2name
        self.make_mappings()

    @classmethod
    def from_xml_string(cls, xml):
        model = mujoco.MjModel.from_xml_string(xml)
        return cls(model)

    @classmethod
    def from_xml_file(cls, xml_file):
        f = open(xml_file, "r")
        xml = f.read()
        f.close()
        return cls.from_xml_string(xml)

    def _extract_mj_names(self, name_adr, num_obj, obj_type):
        """
        See https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L1127
        """

        ### TODO: fix this to use @name_adr like mujoco-py - more robust than assuming IDs are continuous ###

        # objects don't need to be named in the XML, so name might be None
        id2name = { i: None for i in range(num_obj) }
        name2id = {}
        for i in range(num_obj):
            name = mujoco.mj_id2name(self.model, obj_type, i)
            name2id[name] = i
            id2name[i] = name

        # # objects don't need to be named in the XML, so name might be None
        # id2name = { i: None for i in range(num_obj) }
        # name2id = {}
        # for i in range(num_obj):
        #     name = self.model.names[name_adr[i]]
        #     decoded_name = name.decode()
        #     if decoded_name:
        #         obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        #         assert (0 <= obj_id < num_obj) and (id2name[obj_id] is None)
        #         name2id[decoded_name] = obj_id
        #         id2name[obj_id] = decoded_name

         # sort names by increasing id to keep order deterministic
        return tuple(id2name[nid] for nid in sorted(name2id.values())), name2id, id2name

    def make_mappings(self):
        """
        Make some useful internal mappings that mujoco-py supported.
        """
        p = self.model
        self.body_names, self._body_name2id, self._body_id2name = self._extract_mj_names(p.name_bodyadr, p.nbody, mujoco.mjtObj.mjOBJ_BODY)
        self.joint_names, self._joint_name2id, self._joint_id2name = self._extract_mj_names(p.name_jntadr, p.njnt, mujoco.mjtObj.mjOBJ_JOINT)
        self.geom_names, self._geom_name2id, self._geom_id2name = self._extract_mj_names(p.name_geomadr, p.ngeom, mujoco.mjtObj.mjOBJ_GEOM)
        self.site_names, self._site_name2id, self._site_id2name = self._extract_mj_names(p.name_siteadr, p.nsite, mujoco.mjtObj.mjOBJ_SITE)
        self.light_names, self._light_name2id, self._light_id2name = self._extract_mj_names(p.name_lightadr, p.nlight, mujoco.mjtObj.mjOBJ_LIGHT)
        self.camera_names, self._camera_name2id, self._camera_id2name = self._extract_mj_names(p.name_camadr, p.ncam, mujoco.mjtObj.mjOBJ_CAMERA)
        self.actuator_names, self._actuator_name2id, self._actuator_id2name = self._extract_mj_names(p.name_actuatoradr, p.nu, mujoco.mjtObj.mjOBJ_ACTUATOR)
        self.sensor_names, self._sensor_name2id, self._sensor_id2name = self._extract_mj_names(p.name_sensoradr, p.nsensor, mujoco.mjtObj.mjOBJ_SENSOR)
        self.tendon_names, self._tendon_name2id, self._tendon_id2name = self._extract_mj_names(p.name_tendonadr, p.ntendon, mujoco.mjtObj.mjOBJ_TENDON)
        self.mesh_names, self._mesh_name2id, self._mesh_id2name = self._extract_mj_names(p.name_meshadr, p.nmesh, mujoco.mjtObj.mjOBJ_MESH)

    def reset(self):
        """Reset simulation."""
        mujoco.mj_resetData(self.model, self.data)

    def forward(self):
        """Forward call to synchronize derived quantities."""
        mujoco.mj_forward(self.model, self.data)

    def step(self, with_udd=True):
        """Step simulation."""
        mujoco.mj_step(self.model, self.data)

    def get_state(self):
        """Return MjSimState instance for current state."""
        return MjSimState(
            time=self.data.time,
            qpos=np.copy(self.data.qpos),
            qvel=np.copy(self.data.qvel),
        )

    def set_state(self, value):
        """
        Set internal state from MjSimState instance. Should
        call @forward afterwards to synchronize derived quantities.
        """
        self.data.time = value.time
        self.data.qpos[:] = np.copy(value.qpos)
        self.data.qvel[:] = np.copy(value.qvel)

    def set_state_from_flattened(self, value):
        """
        Set internal mujoco state using flat mjstate array. Should
        call @forward afterwards to synchronize derived quantities.

        See https://github.com/openai/mujoco-py/blob/4830435a169c1f3e3b5f9b58a7c3d9c39bdf4acb/mujoco_py/mjsimstate.pyx#L54
        """
        state = MjSimState.from_flattened(value, self)

        # do this instead of @set_state to avoid extra copy of qpos and qvel
        self.data.time = state.time
        self.data.qpos[:] = state.qpos
        self.data.qvel[:] = state.qvel

    """
    Some methods supported by sim.model in mujoco-py.
    Copied from https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L2611
    """
    def body_id2name(self, id):
        if id not in self._body_id2name:
            raise ValueError("No body with id %d exists." % id)
        return self._body_id2name[id]

    def body_name2id(self, name):
        if name not in self._body_name2id:
            raise ValueError("No \"body\" with name %s exists. Available \"body\" names = %s." % (name, self.body_names))
        return self._body_name2id[name]

    def body_name2id(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def joint_id2name(self, id):
        if id not in self._joint_id2name:
            raise ValueError("No joint with id %d exists." % id)
        return self._joint_id2name[id]

    def joint_name2id(self, name):
        if name not in self._joint_name2id:
            raise ValueError("No \"joint\" with name %s exists. Available \"joint\" names = %s." % (name, self.joint_names))
        return self._joint_name2id[name]

    def geom_id2name(self, id):
        if id not in self._geom_id2name:
            raise ValueError("No geom with id %d exists." % id)
        return self._geom_id2name[id]

    def geom_name2id(self, name):
        if name not in self._geom_name2id:
            raise ValueError("No \"geom\" with name %s exists. Available \"geom\" names = %s." % (name, self.geom_names))
        return self._geom_name2id[name]

    def site_id2name(self, id):
        if id not in self._site_id2name:
            raise ValueError("No site with id %d exists." % id)
        return self._site_id2name[id]

    def site_name2id(self, name):
        if name not in self._site_name2id:
            raise ValueError("No \"site\" with name %s exists. Available \"site\" names = %s." % (name, self.site_names))
        return self._site_name2id[name]

    def light_id2name(self, id):
        if id not in self._light_id2name:
            raise ValueError("No light with id %d exists." % id)
        return self._light_id2name[id]

    def light_name2id(self, name):
        if name not in self._light_name2id:
            raise ValueError("No \"light\" with name %s exists. Available \"light\" names = %s." % (name, self.light_names))
        return self._light_name2id[name]

    def camera_id2name(self, id):
        if id not in self._camera_id2name:
            raise ValueError("No camera with id %d exists." % id)
        return self._camera_id2name[id]

    def camera_name2id(self, name):
        if name not in self._camera_name2id:
            raise ValueError("No \"camera\" with name %s exists. Available \"camera\" names = %s." % (name, self.camera_names))
        return self._camera_name2id[name]

    def actuator_id2name(self, id):
        if id not in self._actuator_id2name:
            raise ValueError("No actuator with id %d exists." % id)
        return self._actuator_id2name[id]

    def actuator_name2id(self, name):
        if name not in self._actuator_name2id:
            raise ValueError("No \"actuator\" with name %s exists. Available \"actuator\" names = %s." % (name, self.actuator_names))
        return self._actuator_name2id[name]

    def sensor_id2name(self, id):
        if id not in self._sensor_id2name:
            raise ValueError("No sensor with id %d exists." % id)
        return self._sensor_id2name[id]

    def sensor_name2id(self, name):
        if name not in self._sensor_name2id:
            raise ValueError("No \"sensor\" with name %s exists. Available \"sensor\" names = %s." % (name, self.sensor_names))
        return self._sensor_name2id[name]

    def tendon_id2name(self, id):
        if id not in self._tendon_id2name:
            raise ValueError("No tendon with id %d exists." % id)
        return self._tendon_id2name[id]

    def tendon_name2id(self, name):
        if name not in self._tendon_name2id:
            raise ValueError("No \"tendon\" with name %s exists. Available \"tendon\" names = %s." % (name, self.tendon_names))
        return self._tendon_name2id[name]

    def mesh_id2name(self, id):
        if id not in self._mesh_id2name:
            raise ValueError("No mesh with id %d exists." % id)
        return self._mesh_id2name[id]

    def mesh_name2id(self, name):
        if name not in self._mesh_name2id:
            raise ValueError("No \"mesh\" with name %s exists. Available \"mesh\" names = %s." % (name, self.mesh_names))
        return self._mesh_name2id[name]

    # def userdata_id2name(self, id):
    #     if id not in self._userdata_id2name:
    #         raise ValueError("No userdata with id %d exists." % id)
    #     return self._userdata_id2name[id]

    # def userdata_name2id(self, name):
    #     if name not in self._userdata_name2id:
    #         raise ValueError("No \"userdata\" with name %s exists. Available \"userdata\" names = %s." % (name, self.userdata_names))
    #     return self._userdata_name2id[name]

    def get_xml(self):
        with TemporaryDirectory() as td:
            filename = os.path.join(td, 'model.xml')
            ret = mujoco.mj_saveLastXML(filename.encode(), self.model)
            return open(filename).read()

    def get_joint_qpos_addr(self, name):
        """
        See https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L1178

        Returns the qpos address for given joint.
        Returns:
        - address (int, tuple): returns int address if 1-dim joint, otherwise
            returns the a (start, end) tuple for pos[start:end] access.
        """
        joint_id = self.joint_name2id(name)
        joint_type = self.model.jnt_type[joint_id]
        joint_addr = self.model.jnt_qposadr[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            ndim = 7
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            ndim = 4
        else:
            assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
            ndim = 1

        if ndim == 1:
            return joint_addr
        else:
            return (joint_addr, joint_addr + ndim)

    def get_joint_qvel_addr(self, name):
        """
        See https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L1202

        Returns the qvel address for given joint.
        Returns:
        - address (int, tuple): returns int address if 1-dim joint, otherwise
            returns the a (start, end) tuple for vel[start:end] access.
        """
        joint_id = self.joint_name2id(name)
        joint_type = self.model.jnt_type[joint_id]
        joint_addr = self.model.jnt_dofadr[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            ndim = 6
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            ndim = 3
        else:
            assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
            ndim = 1

        if ndim == 1:
            return joint_addr
        else:
            return (joint_addr, joint_addr + ndim)

    """
    Some methods supported by sim.data in mujoco-py.
    Copied from https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L2611
    """
    def get_body_xpos(self, name):
        bid = self.body_name2id(name)
        return self.data.xpos[bid]

    def get_body_xquat(self, name):
        bid = self.body_name2id(name)
        return self.data.xquat[bid]

    def get_body_xmat(self, name):
        bid = self.body_name2id(name)
        return self.data.xmat[bid].reshape((3, 3))

    def get_body_jacp(self, name):
        bid = self.body_name2id(name)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, None, bid)
        return jacp

    def get_body_jacr(self, name):
        bid = self.body_name2id(name)
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, None, jacr, bid)
        return jacr

    def get_body_xvelp(self, name):
        jacp = self.get_body_jacp(name)
        xvelp = np.dot(jacp, self.data.qvel)
        return xvelp

    def get_body_xvelr(self, name):
        jacr = self.get_body_jacr(name)
        xvelr = np.dot(jacr, self.data.qvel)
        return xvelr

    def get_geom_xpos(self, name):
        gid = self.geom_name2id(name)
        return self.data.geom_xpos[gid]

    def get_geom_xmat(self, name):
        gid = self.geom_name2id(name)
        return self.data.geom_xmat[gid].reshape((3, 3))

    def get_geom_jacp(self, name):
        gid = self._model.geom_name2id(name)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacGeom(self.model, self.data, jacp, None, gid)
        return jacp

    def get_geom_jacr(self, name):
        gid = self._model.geom_name2id(name)
        jacv = np.zeros((3, self.model.nv))
        mujoco.mj_jacGeom(self.model, self.data, None, jacv, gid)
        return jacr

    def get_geom_xvelp(self, name):
        jacp = self.get_geom_jacp(name)
        xvelp = np.dot(jacp, self.data.qvel)
        return xvelp

    def get_geom_xvelr(self, name):
        jacr = self.get_geom_jacr(name)
        xvelr = np.dot(jacr, self.data.qvel)
        return xvelr

    def get_site_xpos(self, name):
        sid = self.site_name2id(name)
        return self.data.site_xpos[sid]

    def get_site_xmat(self, name):
        sid = self.site_name2id(name)
        return self.data.site_xmat[sid].reshape((3, 3))

    def get_site_jacp(self, name):
        sid = self.site_name2id(name)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, sid)
        return jacp

    def get_site_jacr(self, name):
        sid = self.site_name2id(name)
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, None, jacr, sid)
        return jacr

    def get_site_xvelp(self, name):
        jacp = self.get_site_jacp(name)
        xvelp = np.dot(jacp, self.data.qvel)
        return xvelp

    def get_site_xvelr(self, name):
        jacr = self.get_site_jacr(name)
        xvelr = np.dot(jacr, self.data.qvel)
        return xvelr

    def get_camera_xpos(self, name):
        cid = self.camera_name2id(name)
        return self._cam_xpos[cid]

    def get_camera_xmat(self, name):
        cid = self.camera_name2id(name)
        return self.data.cam_xmat[cid].reshape((3, 3))

    def get_light_xpos(self, name):
        lid = self.light_name2id(name)
        return self.data.light_xpos[lid]

    def get_light_xdir(self, name):
        lid = self.light_name2id(name)
        return self.data.light_xdir[lid]

    def get_sensor(self, name):
        sid = self._model.sensor_name2id(name)
        return self.data.sensordata[sid]

    def get_mocap_pos(self, name):
        body_id = self.body_name2id(name)
        mocap_id = self.model.body_mocapid[body_id]
        return self.data.mocap_pos[mocap_id]

    def set_mocap_pos(self, name, value):
        body_id = self.body_name2id(name)
        mocap_id = self.model.body_mocapid[body_id]
        self.data.mocap_pos[mocap_id] = value

    def get_mocap_quat(self, name):
        body_id = self.body_name2id(name)
        mocap_id = self.model.body_mocapid[body_id]
        return self.data.mocap_quat[mocap_id]

    def set_mocap_quat(self, name, value):
        body_id = self.body_name2id(name)
        mocap_id = self.model.body_mocapid[body_id]
        self.data.mocap_quat[mocap_id] = value

    def get_joint_qpos(self, name):
        addr = self.get_joint_qpos_addr(name)
        if isinstance(addr, (int, np.int32, np.int64)):
            return self.data.qpos[addr]
        else:
            start_i, end_i = addr
            return self.data.qpos[start_i:end_i]

    def set_joint_qpos(self, name, value):
        """
        See https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L2821
        """
        addr = self.get_joint_qpos_addr(name)
        if isinstance(addr, (int, np.int32, np.int64)):
            self.data.qpos[addr] = value
        else:
            start_i, end_i = addr
            value = np.array(value)
            assert value.shape == (end_i - start_i,), (
                "Value has incorrect shape %s: %s" % (name, value))
            self.data.qpos[start_i:end_i] = value

    def get_joint_qvel(self, name):
        addr = self.get_joint_qvel_addr(name)
        if isinstance(addr, (int, np.int32, np.int64)):
            return self.data.qvel[addr]
        else:
            start_i, end_i = addr
            return self.data.qvel[start_i:end_i]

    def set_joint_qvel(self, name, value):
        addr = self.get_joint_qvel_addr(name)
        if isinstance(addr, (int, np.int32, np.int64)):
            self.data.qvel[addr] = value
        else:
            start_i, end_i = addr
            value = np.array(value)
            assert value.shape == (end_i - start_i,), (
                "Value has incorrect shape %s: %s" % (name, value))
            self.data.qvel[start_i:end_i] = value
