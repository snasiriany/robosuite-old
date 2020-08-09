from collections import OrderedDict
import numpy as np
from copy import deepcopy

import robosuite.utils.env_utils as EU
from robosuite.environments.sawyer import SawyerEnv

from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.objects.interactive_objects import MomentaryButtonObject, MaintainedButtonObject
from robosuite.models.tasks import TableTopMergedTask, SequentialCompositeSampler
from robosuite.controllers import load_controller_config
import os


class SawyerPT(SawyerEnv):
    """
    Sawyer PlayTable
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
        position_only=True,
        eval_mode_perturb_range=0.0,
    ):

        # Load the default controller if none is specified
        if controller_config is None:
            controller_path = os.path.join(os.path.dirname(__file__), '..', 'controllers/config/default_sawyer.json')
            controller_config = load_controller_config(custom_fpath=controller_path)

        # Assert that the controller config is a dict file
        assert type(controller_config) == dict, \
            "Inputted controller config must be a dict! Instead, got type: {}".format(type(controller_config))

        if controller_config["type"] == "EE_POS_ORI" and position_only:
            controller_config["type"] = "EE_POS"

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # reward configuration
        self.reward_shaping = reward_shaping

        self.interactive_objects = OrderedDict()
        self._has_interaction = False
        self._goal_dict = None  # observation for goal state

        self._eval_mode_perturb_range = eval_mode_perturb_range

        if eval_mode:
            self.placement_initializer = self._get_placement_initializer_for_eval_mode()
            self.placement_initializer = self._get_default_placement_initializer()
        else:
            self.placement_initializer = self._get_default_placement_initializer()

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

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].
        """
        di = super()._get_observation()

        gripper_site_pos = np.array(self.sim.data.site_xpos[self.eef_site_id])

        object_state = []
        object_only_state = []  # no gripper-object pos

        for obj_name in self.task_object_names:
            pos = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id(obj_name)])
            quat = np.array(self.sim.data.body_xquat[self.sim.model.body_name2id(obj_name)])
            rel_pos = gripper_site_pos - pos
            object_state.extend([pos, quat, rel_pos])
            object_only_state.extend([pos])

        object_state = np.concatenate(object_state, axis=0)
        object_only_state = np.concatenate(object_only_state, axis=0)
        ostate = [o.flat_state for o in self.interactive_objects.values()]

        di["object-state"] = np.concatenate([object_state] + ostate)
        di["object-goal-state"] = np.concatenate([object_only_state] + ostate)
        di["task_id"] = np.array([self.task_id])
        return di

    def set_task_objects_visual_position(self, object_states):
        """Set positions of the visual objects"""
        assert len(object_states) == (len(self.task_object_names) * 3)  # [x, y, z] for each task object
        object_states = object_states.reshape((-1, 3))
        curr_pos = []
        for i, obj_name in enumerate(self.task_object_names):
            curr_pos.append(np.array(self.sim.data.body_xpos[self.sim.model.body_name2id(obj_name + "_visual")]))
            EU.set_body_pose(sim=self.sim, body_name=obj_name + "_visual", pos=object_states[i])
            self.sim.forward()
        return np.concatenate(curr_pos, axis=0)

    def render_subgoal(self, subgoal_state, camera_name=None, width=None, height=None):
        """
        Visualize subgoal in an environment
        """
        prev_state = self.set_task_objects_visual_position(subgoal_state)
        camera_name = camera_name if camera_name is not None else self.camera_name
        width = width if width is not None else self.camera_width
        height = height if height is not None else self.camera_height
        im = self.sim.render(camera_name=camera_name, width=width, height=height)
        im = im[::-1]  # flip
        self.set_task_objects_visual_position(prev_state)
        return im

    @property
    def task_id(self):
        return 0.

    def _load_model(self):
        SawyerEnv._load_model(self)

        # setup robot and arena
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        # load model for table top workspace
        self.mujoco_arena = TableArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator(self.indicator_num)

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.16 + self.table_full_size[0] / 2, 0, 0])

        mujoco_objects, visual_objects = self._load_objects()
        # task includes arena, robot, and objects of interest
        self.model = TableTopMergedTask(
            self.mujoco_arena,
            self.mujoco_robot,
            mujoco_objects=mujoco_objects,
            visual_objects=visual_objects,
            initializer=self.placement_initializer,
        )
        self.model.place_objects()

    def reward(self, action=None):
        return 0.

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        self._has_interaction = False
        self._goal_dict = None

        super()._reset_internal()

        # reset joint positions
        init_pos = np.array([-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
        # init_pos += np.random.randn(init_pos.shape[0]) * 0.02
        self.sim.data.qpos[self._ref_joint_pos_indexes] = np.array(init_pos)

    def step(self, action):
        if not self._has_interaction:
            # this is the first step call of the episode
            self.placement_initializer.increment_counter()
        self._has_interaction = True
        for _, o in self.interactive_objects.items():
            o.step(sim_step=self.timestep)
        return super().step(action)

    def set_goal(self, _):
        # do nothing
        pass

    def _set_goal_rendering(self, _):
        pass

    def _get_goal(self):
        """
        Get goal observation by moving object to the target, get obs, and move back.
        :return: observation dict with goal
        """
        # avoid generating goal obs every time
        if self._goal_dict is not None:
            return self._goal_dict

        with EU.world_saved(self.sim):
            self._set_state_to_goal()
            self._goal_dict = deepcopy(self._get_observation())

        return self._goal_dict

    def _get_placement_initializer_for_eval_mode(self):
        pass

    def _get_default_placement_initializer(self):
        raise NotImplementedError

    @property
    def task_object_names(self):
        raise NotImplementedError

    def _load_objects(self):
        raise NotImplementedError

    def _set_state_to_goal(self):
        """Set the environment to a goal state"""
        raise NotImplementedError

    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        raise NotImplementedError


class SawyerPTStack(SawyerPT):
    def _get_default_placement_initializer(self):
        initializer = SequentialCompositeSampler()
        range = 0.1
        initializer.sample_on_top(
            "cube1", "table",
            x_range=(-range, range), y_range=(-range, range), z_rotation=(0.0, 0.0), ensure_object_boundary_in_range=False
        )
        initializer.sample_on_top(
            "cube2", "table",
            x_range=(-range, range), y_range=(-range, range), z_rotation=(0.0, 0.0), ensure_object_boundary_in_range=False
        )
        initializer.sample_on_top(
            "plate", "table",
            x_range=(-range, range), y_range=(-range, range), z_rotation=(0.0, 0.0),  ensure_object_boundary_in_range=False
        )

        initializer.hide("cube1_visual")
        initializer.hide("cube2_visual")
        initializer.hide("plate_visual")
        return initializer

    def _get_placement_initializer_for_eval_mode(self):
        initializer = SequentialCompositeSampler()
        initializer.sample_on_top_square_grid(
            "cube1", "table",
            x_bound=(-0.1, 0.1, 5), y_bound=(-0.1, 0.1, 5), z_bound=(0.0, 0.0, 1), ensure_object_boundary_in_range=False
        )
        initializer.sample_on_top_square_grid(
            "cube2", "table",
            x_bound=(-0.1, 0.1, 5), y_bound=(-0.1, 0.1, 5), z_bound=(0.0, 0.0, 1), ensure_object_boundary_in_range=False
        )
        initializer.sample_on_top_square_grid(
            "plate", "table",
            x_bound=(-0.1, 0.1, 5), y_bound=(-0.1, 0.1, 5), z_bound=(0.0, 0.0, 1), ensure_object_boundary_in_range=False
        )
        return initializer

    def _load_objects(self):
        # setup objects and initializers
        mujoco_objects = OrderedDict()
        visual_objects = OrderedDict()

        mujoco_objects["cube1"] = BoxObject(size=(0.02, 0.02, 0.02), rgba=(1, 0, 0, 1))
        mujoco_objects["cube2"] = BoxObject(size=(0.02, 0.02, 0.02), rgba=(0, 0, 1, 1))
        mujoco_objects["plate"] = BoxObject(size=(0.03, 0.03, 0.01), rgba=(0, 1, 0, 1))

        visual_objects["cube1_visual"] = BoxObject(size=(0.02, 0.02, 0.02), rgba=(1, 0, 0, 0.3))
        visual_objects["cube2_visual"] = BoxObject(size=(0.02, 0.02, 0.02), rgba=(0, 0, 1, 0.3))
        visual_objects["plate_visual"] = BoxObject(size=(0.03, 0.03, 0.01), rgba=(0, 1, 0, 0.3))
        # target visual object
        return mujoco_objects, visual_objects

    @property
    def task_object_names(self):
        return ["cube1", "cube2", "plate"]

    @property
    def task_id(self):
        return 0.

    def _set_state_to_goal(self):
        """Set the environment to a goal state"""
        new_pos, new_quat = EU.sample_stable_placement(
            self.sim,
            self.model.mujoco_objects["cube1"],
            self.sim.model.body_name2id("cube1"),
            self.model.mujoco_objects["plate"],
            self.sim.model.body_name2id("plate"),
            max_radius=0.  # center placement
        )
        EU.set_body_pose(self.sim, "cube1", pos=new_pos, quat=new_quat)
        self.sim.forward()

    def _check_success(self):
        return EU.is_stable_placement(
            self.sim,
            self.model.mujoco_objects["cube1"],
            self.sim.model.body_name2id("cube1"),
            self.model.mujoco_objects["plate"],
            self.sim.model.body_name2id("plate"),
        )


class SawyerPTStackCubes(SawyerPTStack):
    @property
    def task_id(self):
        return 1.

    def _set_state_to_goal(self):
        """Set the environment to a goal state"""
        new_pos, new_quat = EU.sample_stable_placement(
            self.sim,
            self.model.mujoco_objects["cube2"],
            self.sim.model.body_name2id("cube2"),
            self.model.mujoco_objects["cube1"],
            self.sim.model.body_name2id("cube1"),
            max_radius=0.  # center placement
        )
        EU.set_body_pose(self.sim, "cube2", pos=new_pos, quat=new_quat)
        self.sim.forward()

    def _check_success(self):
        return EU.is_stable_placement(
            self.sim,
            self.model.mujoco_objects["cube2"],
            self.sim.model.body_name2id("cube2"),
            self.model.mujoco_objects["cube1"],
            self.sim.model.body_name2id("cube1"),
        )


class SawyerPTStackAll(SawyerPTStack):
    @property
    def task_id(self):
        return 2.

    def _set_state_to_goal(self):
        """Set the environment to a goal state"""
        new_pos, new_quat = EU.sample_stable_placement(
            self.sim,
            self.model.mujoco_objects["cube1"],
            self.sim.model.body_name2id("cube1"),
            self.model.mujoco_objects["plate"],
            self.sim.model.body_name2id("plate"),
            max_radius=0.  # center placement
        )
        EU.set_body_pose(self.sim, "cube1", pos=new_pos, quat=new_quat)
        self.sim.forward()

        new_pos, new_quat = EU.sample_stable_placement(
            self.sim,
            self.model.mujoco_objects["cube2"],
            self.sim.model.body_name2id("cube2"),
            self.model.mujoco_objects["cube1"],
            self.sim.model.body_name2id("cube1"),
            max_radius=0.  # center placement
        )
        EU.set_body_pose(self.sim, "cube2", pos=new_pos, quat=new_quat)
        self.sim.forward()

    def _check_success(self):
        cube_stack = EU.is_stable_placement(
            self.sim,
            self.model.mujoco_objects["cube2"],
            self.sim.model.body_name2id("cube2"),
            self.model.mujoco_objects["cube1"],
            self.sim.model.body_name2id("cube1"),
        )

        cube_on_plate = EU.is_stable_placement(
            self.sim,
            self.model.mujoco_objects["cube1"],
            self.sim.model.body_name2id("cube1"),
            self.model.mujoco_objects["plate"],
            self.sim.model.body_name2id("plate"),
        )
        return cube_stack and cube_on_plate