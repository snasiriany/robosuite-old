from collections import OrderedDict
import numpy as np
from copy import deepcopy

from robosuite.utils.mjcf_utils import xml_path_completion, bounds_to_grid
import robosuite.utils.transform_utils as T
import robosuite.utils.env_utils as EU
from robosuite.environments.sawyer import SawyerEnv

from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CompositeBodyObject, HingeStackObject, CoffeeMachineObject
from robosuite.models.robots import Sawyer
from robosuite.models.tasks import TableTopMergedTask, UniformRandomSampler, SequentialCompositeSampler, RoundRobinSampler
from robosuite.controllers import load_controller_config
import os


class SawyerCoffee(SawyerEnv):
    """
    This class corresponds to the coffee task for the Sawyer robot arm.
    """

    def __init__(
        self,
        controller_config=None,
        gripper_type="TwoFingerGripper",
        table_full_size=(0.8, 0.8, 0.8),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_shaping=False,
        placement_initializer=None,
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
        eval_mode=False,
        perturb_evals=False,
    ):
        """
        Args:
            controller_config (dict): If set, contains relevant controller parameters for creating a custom controller.
                Else, uses the default controller for this specific task

            gripper_type (str): type of gripper, used to instantiate
                gripper models from gripper factory.

            table_full_size (3-tuple): x, y, and z dimensions of the table.

            table_friction (3-tuple): the three mujoco friction parameters for
                the table.

            use_camera_obs (bool): if True, every observation includes a
                rendered image.

            use_object_obs (bool): if True, include object (cube) information in
                the observation.

            reward_shaping (bool): if True, use dense rewards.

            placement_initializer (ObjectPositionSampler instance): if provided, will
                be used to place objects on every reset, else a UniformRandomSampler
                is used by default.

            gripper_visualization (bool): True if using gripper visualization.
                Useful for teleoperation.

            use_indicator_object (bool): if True, sets up an indicator object that
                is useful for debugging.

            indicator_num (int): number of indicator objects to add to the environment.
                Only used if @use_indicator_object is True.

            has_renderer (bool): If true, render the simulation state in
                a viewer instead of headless mode.

            has_offscreen_renderer (bool): True if using off-screen rendering.

            render_collision_mesh (bool): True if rendering collision meshes
                in camera. False otherwise.

            render_visual_mesh (bool): True if rendering visual meshes
                in camera. False otherwise.

            control_freq (float): how many control signals to receive
                in every second. This sets the amount of simulation time
                that passes between every action input.

            horizon (int): Every episode lasts for exactly @horizon timesteps.

            ignore_done (bool): True if never terminating the environment (ignore @horizon).

            camera_name (str): name of camera to be rendered. Must be
                set if @use_camera_obs is True.

            camera_height (int): height of camera frame.

            camera_width (int): width of camera frame.

            camera_depth (bool): True if rendering RGB-D, and RGB otherwise.

            camera_real_depth (bool): True if convert depth to real depth in meters

            camera_segmentation (bool): True if also return semantic segmentation of the camera view
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

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # reward configuration
        self.reward_shaping = reward_shaping

        # object placement initializer
        if placement_initializer is not None:
            self.placement_initializer = placement_initializer
        else:
            self.placement_initializer = self._get_default_initializer()

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
            eval_mode=eval_mode,
            perturb_evals=perturb_evals, 
        )

    def _get_default_initializer(self):
        initializer = SequentialCompositeSampler()
        initializer.sample_on_top(
            "coffee_machine",
            surface_name="table",
            x_range=(0.0, 0.0),
            y_range=(-0.1, -0.1),
            z_rotation=(-np.pi / 6., -np.pi / 6.),
            ensure_object_boundary_in_range=False
        )
        initializer.sample_on_top(
            "coffee_pod",
            surface_name="table",
            x_range=(-0.13, -0.07),
            y_range=(0.17, 0.23),
            z_rotation=(0.0, 0.0),
            ensure_object_boundary_in_range=False
        )
        return initializer

    def _get_placement_initializer_for_eval_mode(self):
        """
        Sets a placement initializer that is used to initialize the
        environment into a fixed set of known task instances.
        This is for reproducibility in policy evaluation.
        """

        assert(self.eval_mode)

        ordered_object_names = ["coffee_machine", "coffee_pod"]
        bounds = self._grid_bounds_for_eval_mode()
        initializer = SequentialCompositeSampler(round_robin_all_pairs=True)

        for name in ordered_object_names:
            if self.perturb_evals:
                # perturbation sizes should be half the grid spacing
                perturb_sizes = [((b[1] - b[0]) / b[2]) / 2. for b in bounds[name][:3]]
            else:
                perturb_sizes = [None for b in bounds[name][:3]]

            grid = bounds_to_grid(bounds[name][:3])
            sampler = RoundRobinSampler(
                x_range=grid[0],
                y_range=grid[1],
                ensure_object_boundary_in_range=False,
                z_rotation=grid[2],
                x_perturb=perturb_sizes[0],
                y_perturb=perturb_sizes[1],
                z_rotation_perturb=perturb_sizes[2],
                z_offset=bounds[name][3],
            )
            initializer.append_sampler(name, sampler)

        self.placement_initializer = initializer
        return initializer

    def _grid_bounds_for_eval_mode(self):
        """
        Helper function to get grid bounds of x positions, y positions, 
        and z-rotations for reproducible evaluations, and number of points
        per dimension.
        """
        ret = {}

        # (low, high, number of grid points for each dimension)
        ret["coffee_machine"] = [
            (0., 0., 1), 
            (-0.1, -0.1, 1), 
            (-np.pi / 6., -np.pi / 6., 1), 
            0.,
        ]

        ret["coffee_pod"] = [
            (-0.13, -0.07, 3), 
            (0.17, 0.23, 3), 
            (0., 0., 1), 
            0.,
        ]

        return ret

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

        # initialize objects of interest
        # self.coffee_machine = HingeStackObject()
        # self.coffee_machine = CoffeeMachineObject()
        # self.coffee_machine = CoffeeMachineXMLObject()

        from robosuite.models.objects import CoffeeMachineBodyObject, CoffeeMachineLidObject, CoffeeMachineBaseObject, CoffeeMachinePodObject, CylinderObject
        self.coffee_pod = CoffeeMachinePodObject()
        # self.coffee_pod = CylinderObject(
        #     size=[0.0225, 0.02],
        #     rgba=[1, 0, 0, 1],
        #     density=100.,
        #     solref=[0.02, 1.],
        #     solimp=[0.998, 0.998, 0.001],
        # )

        # from robosuite.models.objects import TestXMLObject
        # self.coffee_machine = TestXMLObject()
        from robosuite.models.objects import CoffeeMachineObject2
        self.coffee_machine = CoffeeMachineObject2(add_cup=True)
        # from robosuite.models.objects import CupObject
        # self.coffee_machine = CupObject(
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

        self.mujoco_objects = OrderedDict([
            ("coffee_machine", self.coffee_machine), 
            ("coffee_pod", self.coffee_pod),
        ])

        # reset initial joint positions (gets reset in sim during super() call in _reset_internal)
        # self.init_qpos = np.array([-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
        self.init_qpos = np.array([0.00, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])
        self.init_qpos += np.random.randn(self.init_qpos.shape[0]) * 0.02

        # task includes arena, robot, and objects of interest
        self.model = TableTopMergedTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
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
        self.object_body_ids = {}
        self.object_body_ids["coffee_pod_holder"] = self.sim.model.body_name2id("coffee_machine_4")
        self.object_body_ids["coffee_pod"] = self.sim.model.body_name2id("coffee_pod")
        self.hinge_qpos_addr = self.sim.model.get_joint_qpos_addr("coffee_machine_1_0")

        # size of bounding box for pod holder
        self.pod_holder_size = self.mujoco_objects["coffee_machine"].pod_holder_size

        # size of bounding box for pod
        self.pod_size = self.mujoco_objects["coffee_pod"].get_bounding_box_size()

        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.left_finger_geoms
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(x) for x in self.gripper.right_finger_geoms
        ]

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.sim.data.qpos[self.hinge_qpos_addr] = 2. * np.pi / 3.
        self.sim.forward()

    def reward(self, action=None):
        """
        Reward function for the task.

        The dense reward has three components.

            Reaching: in [0, 1], to encourage the arm to reach the cube
            Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            Lifting: in {0, 1}, non-zero if arm has lifted the cube

        The sparse reward only consists of the lifting component.

        Args:
            action (np array): unused for this task

        Returns:
            reward (float): the reward
        """
        reward = 0.

        # sparse completion reward
        if self._check_success()["task"]:
            reward = 1.0

        # use a shaping reward
        if self.reward_shaping:
            pass

        return reward

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:
            robot-state: contains robot-centric information.
            object-state: requires @self.use_object_obs to be True.
                contains object-centric information.
            image: requires @self.use_camera_obs to be True.
                contains a rendered frame from the simulation.
            depth: requires @self.use_camera_obs and @self.camera_depth to be True.
                contains a rendered depth map from the simulation
        """
        di = super()._get_observation()

        # low-level object information
        if self.use_object_obs:

            # remember the keys to collect into object info
            object_state_keys = []

            # for conversion to relative gripper frame
            gripper_pose = T.pose2mat((di["eef_pos"], di["eef_quat"]))
            world_pose_in_gripper = T.pose_inv(gripper_pose)

            # add pose and relative poses of relevant bodies
            for k in self.object_body_ids:
                # position and rotation of the relevant bodies
                body_id = self.object_body_ids[k]
                body_pos = np.array(self.sim.data.body_xpos[body_id])
                body_quat = T.convert_quat(
                    np.array(self.sim.data.body_xquat[body_id]), to="xyzw"
                )
                di["{}_pos".format(k)] = body_pos
                di["{}_quat".format(k)] = body_quat

                # get relative pose of object in gripper frame
                body_pose = T.pose2mat((body_pos, body_quat))
                rel_pose = T.pose_in_A_to_pose_in_B(body_pose, world_pose_in_gripper)
                rel_pos, rel_quat = T.mat2pose(rel_pose)
                di["{}_to_eef_pos".format(k)] = rel_pos
                di["{}_to_eef_quat".format(k)] = rel_quat

                object_state_keys.append("{}_pos".format(k))
                object_state_keys.append("{}_quat".format(k))
                object_state_keys.append("{}_to_eef_pos".format(k))
                object_state_keys.append("{}_to_eef_quat".format(k))

            # add hinge angle of lid
            di["hinge_angle"] = np.array([self.sim.data.qpos[self.hinge_qpos_addr]])
            object_state_keys.append("hinge_angle")

            di["object-state"] = np.concatenate([di[k] for k in object_state_keys])

        return di

    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = False
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            if (
                self.sim.model.geom_id2name(contact.geom1)
                in self.gripper.contact_geoms()
                or self.sim.model.geom_id2name(contact.geom2)
                in self.gripper.contact_geoms()
            ):
                collision = True
                break
        return collision

    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        success = {}

        # lid should be closed (angle should be less than 5 degrees)
        hinge_tolerance = 15. * np.pi / 180. 
        hinge_angle = self.sim.data.qpos[self.hinge_qpos_addr]
        lid_check = (hinge_angle < hinge_tolerance)

        # pod should be in pod holder
        pod_holder_pos = np.array(self.sim.data.body_xpos[self.object_body_ids["coffee_pod_holder"]])
        pod_pos = np.array(self.sim.data.body_xpos[self.object_body_ids["coffee_pod"]])
        pod_check = True
        pod_horz_check = True

        # center of pod cannot be more than the difference of radii away from the center of pod holder
        r_diff = self.pod_holder_size[0] - self.pod_size[0]
        if np.linalg.norm(pod_pos[:2] - pod_holder_pos[:2]) > r_diff:
            pod_check = False
            pod_horz_check = False

        # make sure vertical pod dimension is above pod holder lower bound and below the lid lower bound
        lid_body_id = self.sim.model.body_name2id("coffee_machine_1")
        lid_pos = np.array(self.sim.data.body_xpos[lid_body_id])
        z_lim_low = pod_holder_pos[2] - self.pod_holder_size[2]
        z_lim_high = lid_pos[2] - self.mujoco_objects["coffee_machine"].lid_size[2]
        if (pod_pos[2] - self.pod_size[2] < z_lim_low) or (pod_pos[2] + self.pod_size[2] > z_lim_high):
            pod_check = False

        success["task"] = lid_check and pod_check

        # partial task metrics below

        # for pod insertion check, just check that bottom of pod is within some tolerance of bottom of container
        pod_insertion_z_tolerance = 0.02
        pod_z_check = (pod_pos[2] - self.pod_size[2] > z_lim_low) and (pod_pos[2] - self.pod_size[2] < z_lim_low + pod_insertion_z_tolerance)
        success["insertion"] = pod_horz_check and pod_z_check

        # pod grasp check
        touch_left_finger = False
        touch_right_finger = False
        pod_geom_id = self.sim.model.geom_name2id("coffee_pod")
        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            if c.geom1 in self.l_finger_geom_ids and c.geom2 == pod_geom_id:
                touch_left_finger = True
            if c.geom1 == pod_geom_id and c.geom2 in self.l_finger_geom_ids:
                touch_left_finger = True
            if c.geom1 in self.r_finger_geom_ids and c.geom2 == pod_geom_id:
                touch_right_finger = True
            if c.geom1 == pod_geom_id and c.geom2 in self.r_finger_geom_ids:
                touch_right_finger = True
        success["grasp"] = (touch_left_finger and touch_right_finger)

        # check is True if the pod is on / near the rim of the pod holder
        rim_horz_tolerance = 0.03
        rim_horz_check = (np.linalg.norm(pod_pos[:2] - pod_holder_pos[:2]) < rim_horz_tolerance)

        rim_vert_tolerance = 0.026
        rim_vert_length = pod_pos[2] - pod_holder_pos[2] - self.pod_holder_size[2]
        rim_vert_check = (rim_vert_length < rim_vert_tolerance) and (rim_vert_length > 0.)
        success["rim"] = rim_horz_check and rim_vert_check

        return success

    def _gripper_visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """

        # color the gripper site appropriately based on distance to block
        if self.gripper_visualization:
            # get distance to cube
            block_site_id = self.sim.model.site_name2id("coffee_machine")
            dist = np.sum(
                np.square(
                    self.sim.data.site_xpos[block_site_id]
                    - self.sim.data.get_site_xpos("grip_site")
                )
            )

            # set RGBA for the EEF site here
            max_dist = 0.1
            scaled = (1.0 - min(dist / max_dist, 1.)) ** 15
            rgba = np.zeros(4)
            rgba[0] = 1 - scaled
            rgba[1] = scaled
            rgba[3] = 0.5

            self.sim.model.site_rgba[self.eef_site_id] = rgba


class SawyerCoffeeFT(SawyerCoffee):
    """
    Variant of Sawyer Coffee task that is equipped with FT sensors on the gripper
    and in the environment for improved observations.
    """
    def __init__(
        self,
        **kwargs
    ):
        # use FT gripper
        assert "gripper_type" not in kwargs
        kwargs["gripper_type"] = "TwoFingerGripperWithFT"
        super().__init__(**kwargs)

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        SawyerEnv._load_model(self)
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        # load model for table top workspace
        self.mujoco_arena = TableArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator(self.indicator_num)

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_full_size[0] / 2, 0, 0])

        from robosuite.models.objects import CoffeeMachineBodyObject, CoffeeMachineLidObject, CoffeeMachineBaseObject, CoffeeMachinePodObject, CylinderObject
        self.coffee_pod = CoffeeMachinePodObject()

        from robosuite.models.objects import CoffeeMachineObject2
        # pod_holder_friction = [1.0, 5e-3, 1e-4] # low friction on holder surface
        self.coffee_machine = CoffeeMachineObject2(
            add_cup=True, 
            # pod_holder_friction=pod_holder_friction,
        )

        self.mujoco_objects = OrderedDict([
            ("coffee_machine", self.coffee_machine), 
            ("coffee_pod", self.coffee_pod),
        ])

        # reset initial joint positions (gets reset in sim during super() call in _reset_internal)
        self.init_qpos = np.array([0.00, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])

        # task includes arena, robot, and objects of interest
        self.model = TableTopMergedTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            initializer=self.placement_initializer,
        )
        self.model.place_objects()

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:
            robot-state: contains robot-centric information.
            object-state: requires @self.use_object_obs to be True.
                contains object-centric information.
            image: requires @self.use_camera_obs to be True.
                contains a rendered frame from the simulation.
            depth: requires @self.use_camera_obs and @self.camera_depth to be True.
                contains a rendered depth map from the simulation
        """
        di = super()._get_observation()
        if self.use_object_obs:
            # add in sensor measurement
            di["object-state"] = np.concatenate([
                di["object-state"],
                self.get_sensor_measurement("force_ee"),
                self.get_sensor_measurement("torque_ee"),
            ])
        return di
