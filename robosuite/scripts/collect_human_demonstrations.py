"""
A script to collect a batch of human demonstrations that can be used
to generate a learning curriculum (see `demo_learning_curriculum.py`).

The demonstrations can be played back using the `playback_demonstrations_from_pkl.py`
script.
"""

import os
import shutil
import time
import argparse
import datetime
import h5py
from glob import glob
import numpy as np

import robosuite
import robosuite.utils.transform_utils as T
# from robosuite.wrappers import IKWrapper
from robosuite.wrappers import DataCollectionWrapper

from robosuite.devices import *

import json

from robosuite.controllers import load_controller_config


def collect_human_trajectory(env, device):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env: environment to control
        device (instance of Device class): to receive controls from the device
    """

    obs = env.reset()

    # # rotate the gripper so we can see it easily
    # env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])

    env.viewer.set_camera(camera_id=2)
    env.render()

    is_first = True

    # episode terminates on a spacenav reset input or if task is completed
    reset = False
    task_completion_hold_count = -1 # counter to collect 10 timesteps after reaching goal
    device.start_control()
    while not reset:
        state = device.get_controller_state()
        dpos, rotation, raw_drotation, grasp, reset = (
            state["dpos"],
            state["rotation"],
            state["raw_drotation"],
            state["grasp"],
            state["reset"],
        )

        # # convert into a suitable end effector action for the environment
        # current = env._right_hand_orn
        # drotation = current.T.dot(rotation)  # relative rotation of desired from current
        # dquat = T.mat2quat(drotation)
        # grasp = grasp - 1.  # map 0 to -1 (open) and 1 to 0 (closed halfway)
        # action = np.concatenate([dpos, dquat, [grasp]])

        gripper_dof = 1
        drotation = raw_drotation[[1, 0, 2]]

        # Flip z
        drotation[2] = -drotation[2]
        # Scale rotation for teleoperation (tuned for OSC) -- gains tuned for each device
        drotation = drotation * 1.5 if isinstance(device, Keyboard) else drotation * 50
        dpos = dpos * 75 if isinstance(device, Keyboard) else dpos * 125

        if grasp == 0:
            grasp = -1

        action = np.concatenate([dpos, drotation, [grasp] * gripper_dof])

        obs, reward, done, info = env.step(action)

        # if is_first:
        #     is_first = False
        #
        #     # We grab the initial model xml and state and reload from those so that
        #     # we can support deterministic playback of actions from our demonstrations.
        #     # This is necessary due to rounding issues with the model xml and with
        #     # env.sim.forward(). We also have to do this after the first action is
        #     # applied because the data collector wrapper only starts recording
        #     # after the first action has been played.
        #     initial_mjstate = env.sim.get_state().flatten()
        #     xml_str = env.model.get_xml()
        #     env.reset_from_xml_string(xml_str)
        #     env.sim.reset()
        #     env.sim.set_state_from_flattened(initial_mjstate)
        #     env.sim.forward()
        #     env.viewer.set_camera(camera_id=2)

        env.render()

        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1 # latched state, decrement count
            else:
                task_completion_hold_count = 10 # reset count on first success timestep
        else:
            task_completion_hold_count = -1 # null the counter if there's no success

    print()
    # cleanup for end of data collection episodes
    env.close()


def gather_demonstrations_as_hdf5(directory, out_dir, env_info, excluded_episodes=None, meta_data=None):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    print("Saving hdf5 to", hdf5_path)
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):
        # print("Processing {} ...".format(ep_directory))
        if (excluded_episodes is not None) and (ep_directory in excluded_episodes):
            # print("\tExcluding this episode!")
            continue

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        action_modes = []

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
                action_modes.append(ai["action_modes"])

        if len(states) == 0:
            continue

        # Delete the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action,
        # so we end up with an extra state at the end.
        del states[-1]
        assert len(states) == len(actions)

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))
        ep_data_grp.create_dataset("action_modes", data=np.array(action_modes))

    if num_eps == 0:
        f.close()
        return

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = robosuite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    if meta_data is not None:
        for (k, v) in meta_data.items():
            grp.attrs[k] = v

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(robosuite.models.assets_root, "demonstrations"),
    )
    parser.add_argument("--environment", type=str, default="SawyerLift")
    parser.add_argument("--device", type=str, default="keyboard")
    args = parser.parse_args()

    config = {
        "env_name": args.environment,
    }

    # create original environment
    env = robosuite.make(
        **config,
        ignore_done=True,
        use_camera_obs=False,
        has_renderer=True,
        control_freq=50, #100,
        gripper_visualization=True,
    )

    # # Wrap this with visualization wrapper
    # env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard()
        env.viewer.add_keypress_callback("any", device.on_press)
        env.viewer.add_keyup_callback("any", device.on_release)
        env.viewer.add_keyrepeat_callback("any", device.on_press)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse()
    else:
        raise Exception(
            "Invalid device choice: choose either 'keyboard' or 'spacemouse'."
        )

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)

    # collect demonstrations
    while True:
        collect_human_trajectory(env, device)
        gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)
