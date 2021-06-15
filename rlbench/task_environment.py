import logging
from typing import List, Callable

import numpy as np
from pyquaternion import Quaternion
from pyrep import PyRep
from pyrep.errors import IKError
from pyrep.objects import Dummy, Object

from rlbench import utils
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.backend.exceptions import BoundaryError, WaypointError
from rlbench.backend.observation import Observation
from rlbench.backend.robot import Robot
from rlbench.backend.scene import Scene
from rlbench.backend.task import Task
from rlbench.demo import Demo
from rlbench.observation_config import ObservationConfig

_TORQUE_MAX_VEL = 9999
_DT = 0.05
_MAX_RESET_ATTEMPTS = 40
_MAX_DEMO_ATTEMPTS = 10


class InvalidActionError(Exception):
    pass


class TaskEnvironmentError(Exception):
    pass


class TaskEnvironment(object):

    def __init__(self, pyrep: PyRep, robot: Robot, scene: Scene, task: Task,
                 action_mode: ActionMode, dataset_root: str,
                 obs_config: ObservationConfig,
                 static_positions: bool = False,
                 attach_grasped_objects: bool = True):
        self._pyrep = pyrep
        self._robot = robot
        self._scene = scene
        self._task = task
        self._variation_number = 0
        self._action_mode = action_mode
        self._dataset_root = dataset_root
        self._obs_config = obs_config
        self._static_positions = static_positions
        self._attach_grasped_objects = attach_grasped_objects
        self._reset_called = False
        self._prev_ee_velocity = None
        self._enable_path_observations = False

        self._scene.load(self._task)
        self._pyrep.start()
        self._target_workspace_check = Dummy.create()

        self._last_e = None


    def get_name(self) -> str:
        return self._task.get_name()

    def sample_variation(self) -> int:
        self._variation_number = np.random.randint(
            0, self._task.variation_count())
        return self._variation_number

    def set_variation(self, v: int) -> None:
        if v >= self.variation_count():
            raise TaskEnvironmentError(
                'Requested variation %d, but there are only %d variations.' % (
                    v, self.variation_count()))
        self._variation_number = v

    def variation_count(self) -> int:
        return self._task.variation_count()

    def reset(self) -> (List[str], Observation):
        self._scene.reset()
        try:
            desc = self._scene.init_episode(
                self._variation_number, max_attempts=_MAX_RESET_ATTEMPTS,
                randomly_place=not self._static_positions)
        except (BoundaryError, WaypointError) as e:
            raise TaskEnvironmentError(
                'Could not place the task %s in the scene. This should not '
                'happen, please raise an issues on this task.'
                % self._task.get_name()) from e

        self._reset_called = True

        # redundancy resolution
        self._last_e = None

        # Returns a list of descriptions and the first observation
        return desc, self._scene.get_observation()

    def get_observation(self) -> Observation:
        return self._scene.get_observation()

    def get_joint_upper_velocity_limits(self):
        return self._robot.arm.get_joint_upper_velocity_limits()

    def get_all_graspable_objects(self):
        return self._task.get_graspable_objects()

    def get_robot_visuals(self):
        return self._robot.arm.get_visuals()

    def get_all_graspable_object_positions(self, relative_to_cameras=False):
        """ returns the positions of all graspable object relative to all enabled cameras """
        objects = self._task.get_graspable_objects()
        positions = []
        for ob in objects:
            if relative_to_camera:
                positions.append(self._scene.get_object_position_relative_to_cameras(ob))
            else:
                positions.append({"left_shoulder_camera": ob.get_position(),
                                  "right_shoulder_camera": ob.get_position(),
                                  "front_camera": ob.get_position(),
                                  "wrist_camera": ob.get_position()})
        return positions

    def get_all_graspable_object_poses(self, relative_to_cameras=False):
        """ returns the pose of all graspable object relative to all enabled cameras """
        objects = self._task.get_graspable_objects()
        poses = []
        for ob in objects:
            if relative_to_cameras:
                poses.append(self._scene.get_object_pose_relative_to_cameras(ob))
            else:
                poses.append({"left_shoulder_camera": ob.get_pose(),
                              "right_shoulder_camera": ob.get_pose(),
                              "front_camera": ob.get_pose(),
                              "wrist_camera": ob.get_pose()})
        return poses

    def _assert_action_space(self, action, expected_shape):
        if np.shape(action) != expected_shape:
            raise RuntimeError(
                'Expected the action shape to be: %s, but was shape: %s' % (
                    str(expected_shape), str(np.shape(action))))

    def _assert_unit_quaternion(self, quat):
        if not np.isclose(np.linalg.norm(quat), 1.0):
            raise RuntimeError('Action contained non unit quaternion!')

    def _torque_action(self, action):
        self._robot.arm.set_joint_target_velocities(
            [(_TORQUE_MAX_VEL if t < 0 else -_TORQUE_MAX_VEL)
             for t in action])
        self._robot.arm.set_joint_forces(np.abs(action))

    def _ee_action(self, action, relative_to=None):
        self._assert_unit_quaternion(action[3:])
        try:
            joint_positions = self._robot.arm.solve_ik(
                action[:3], quaternion=action[3:], relative_to=relative_to)
            self._robot.arm.set_joint_target_positions(joint_positions)
        except IKError as e:
            raise InvalidActionError('Could not find a path.') from e
        done = False
        prev_values = None
        # Move until reached target joint positions or until we stop moving
        # (e.g. when we collide wth something)
        while not done:
            self._scene.step()
            cur_positions = self._robot.arm.get_joint_positions()
            reached = np.allclose(cur_positions, joint_positions, atol=0.01)
            not_moving = False
            if prev_values is not None:
                not_moving = np.allclose(
                    cur_positions, prev_values, atol=0.001)
            prev_values = cur_positions
            done = reached or not_moving

    def _path_action(self, action, relative_to=None):
        self._assert_unit_quaternion(action[3:])
        try:

            # Check if the target is in the workspace; if not, then quick reject
            # Only checks position, not rotation
            pos_to_check = action[:3]
            if relative_to is not None:
                self._target_workspace_check.set_position(
                    pos_to_check, relative_to)
                pos_to_check = self._target_workspace_check.get_position()
            valid = self._scene.check_target_in_workspace(pos_to_check)
            if not valid:
                raise InvalidActionError('Target is outside of workspace.')

            path = self._robot.arm.get_path(
                action[:3], quaternion=action[3:], ignore_collisions=True,
                relative_to=relative_to)
            done = False
            observations = []
            while not done:
                done = path.step()
                self._scene.step()
                if self._enable_path_observations:
                    observations.append(self._scene.get_observation())
                success, terminate = self._task.success()
                # If the task succeeds while traversing path, then break early
                if success:
                    break
                observations.append(self._scene.get_observation())
            return observations
        except IKError as e:
            raise InvalidActionError('Could not find a path.') from e

    def step(self, action, camcorder=None) -> (Observation, int, bool):
        # returns observation, reward, done, info
        if not self._reset_called:
            raise RuntimeError(
                "Call 'reset' before calling 'step' on a task.")

        # action should contain 1 extra value for gripper open close state
        arm_action = np.array(action[:-1])
        ee_action = action[-1]

        if 0.0 > ee_action > 1.0:
            raise ValueError('Gripper action expected to be within 0 and 1.')

        # Discretize the gripper action
        current_ee = (1.0 if self._robot.gripper.get_open_amount()[0] > 0.9 else 0.0)

        if ee_action > 0.5:
            ee_action = 1.0
        elif ee_action < 0.5:
            ee_action = 0.0

        if current_ee != ee_action:
            arm_action = np.array([0.0]*7)

        if self._action_mode.arm == ArmActionMode.ABS_JOINT_VELOCITY:

            self._assert_action_space(arm_action,
                                      (len(self._robot.arm.joints),))
            self._robot.arm.set_joint_target_velocities(arm_action)
            self._scene.step()

            # if needed save some images
            if camcorder:
                obs = self._scene.get_observation()
                camcorder.save(obs, self.get_robot_visuals(), self.get_all_graspable_objects())

        elif self._action_mode.arm == ArmActionMode.DELTA_JOINT_VELOCITY:

            self._assert_action_space(arm_action,
                                      (len(self._robot.arm.joints),))
            cur = np.array(self._robot.arm.get_joint_velocities())
            self._robot.arm.set_joint_target_velocities(cur + arm_action)
            self._scene.step()

        elif self._action_mode.arm == ArmActionMode.ABS_JOINT_POSITION:

            self._assert_action_space(arm_action,
                                      (len(self._robot.arm.joints),))
            self._robot.arm.set_joint_target_positions(arm_action)
            self._scene.step()

        elif self._action_mode.arm == ArmActionMode.DELTA_JOINT_POSITION:

            self._assert_action_space(arm_action,
                                      (len(self._robot.arm.joints),))
            cur = np.array(self._robot.arm.get_joint_positions())
            self._robot.arm.set_joint_target_positions(cur + arm_action)
            self._scene.step()

        elif self._action_mode.arm == ArmActionMode.ABS_JOINT_TORQUE:

            self._assert_action_space(
                arm_action, (len(self._robot.arm.joints),))
            self._torque_action(arm_action)
            self._scene.step()

        elif self._action_mode.arm == ArmActionMode.DELTA_JOINT_TORQUE:

            cur = np.array(self._robot.arm.get_joint_forces())
            new_action = cur + arm_action
            self._torque_action(new_action)
            self._scene.step()

        elif self._action_mode.arm == ArmActionMode.ABS_EE_POSE_WORLD_FRAME:

            self._assert_action_space(arm_action, (7,))
            self._ee_action(list(arm_action))

        elif self._action_mode.arm == ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME:

            self._assert_action_space(arm_action, (7,))
            self._path_observations = []
            self._path_observations = self._path_action(list(arm_action))

        elif self._action_mode.arm == ArmActionMode.DELTA_EE_POSE_PLAN_WORLD_FRAME:

            self._assert_action_space(arm_action, (7,))
            a_x, a_y, a_z, a_qx, a_qy, a_qz, a_qw = arm_action
            x, y, z, qx, qy, qz, qw = self._robot.arm.get_tip().get_pose()
            new_rot = Quaternion(a_qw, a_qx, a_qy, a_qz) * Quaternion(qw, qx,
                                                                      qy, qz)
            qw, qx, qy, qz = list(new_rot)
            new_pose = [a_x + x, a_y + y, a_z + z] + [qx, qy, qz, qw]
            self._path_observations = []
            self._path_observations = self._path_action(list(new_pose))

        elif self._action_mode.arm == ArmActionMode.DELTA_EE_POSE_WORLD_FRAME:

            self._assert_action_space(arm_action, (7,))
            a_x, a_y, a_z, a_qx, a_qy, a_qz, a_qw = arm_action
            x, y, z, qx, qy, qz, qw = self._robot.arm.get_tip().get_pose()
            new_rot = Quaternion(a_qw, a_qx, a_qy, a_qz) * Quaternion(
                qw, qx, qy, qz)
            qw, qx, qy, qz = list(new_rot)
            new_pose = [a_x + x, a_y + y, a_z + z] + [qx, qy, qz, qw]
            self._ee_action(list(new_pose))

        elif self._action_mode.arm == ArmActionMode.EE_POSE_EE_FRAME:

            self._assert_action_space(arm_action, (7,))
            self._ee_action(
                list(arm_action), relative_to=self._robot.arm.get_tip())

        elif self._action_mode.arm == ArmActionMode.EE_POSE_PLAN_EE_FRAME:

            self._assert_action_space(arm_action, (7,))
            self._path_observations = []
            self._path_observations = self._path_action(
                list(arm_action), relative_to=self._robot.arm.get_tip())

        else:
            raise RuntimeError('Unrecognised action mode.')

        if current_ee != ee_action:
            done = False
            while not done:
                done = self._robot.gripper.actuate(ee_action, velocity=0.2)
                self._pyrep.step()
                self._task.step()

                # if needed save some images
                if camcorder:
                    obs = self._scene.get_observation()
                    camcorder.save(obs, self.get_robot_visuals(), self.get_all_graspable_objects())

            if ee_action == 0.0 and self._attach_grasped_objects:
                # If gripper close action, the check for grasp.
                for g_obj in self._task.get_graspable_objects():
                    self._robot.gripper.grasp(g_obj)
            else:
                # If gripper open action, the check for ungrasp.
                self._robot.gripper.release()


        success, terminate = self._task.success()
        task_reward = self._task.reward()
        reward = float(success) if task_reward is None else task_reward
        return self._scene.get_observation(), reward, terminate

    def resolve_redundancy_joint_velocities(self, actions, setup):
        """
        Resolves redundant self-motion into the nullspace without changing the gripper tip position
        :param actions:
         Current actions without redundancy resolution.
        :param setup:
         Setup for redundancy resolution defining the mode, weighting etc.
        :return: Array of joint velocities, which move the robot's tip according to the provided actions yet push
         the joint position towards a reference position.
        """
        # get the Jacobian
        J = self._robot.arm.get_jacobian()
        J = np.transpose(J)
        J = np.flip(J)
        J = J[-3:]

        # compute the pseudo inverse
        J_plus = np.linalg.pinv(J)

        # weighting
        if type(setup["W"]) is list:
            W = np.array(setup["W"])
        elif setup["W"] is None:
            # use default weighting later
            W = None
        else:
            raise TypeError("Unsupported type %s for weighting vector." % type(setup["W"]))

        # compute the error
        if setup["mode"] == "reference_position":
            e = self.get_error_reference_position(setup["ref_position"], W)
        elif setup["mode"] == "collision_avoidance":
            e = self.get_error_collision_avoidance(W)

        # compute the joint velocities
        q_dot_redundancy = setup["alpha"] * np.matmul((np.identity(len(self._robot.arm.joints)) - np.matmul(J_plus, J)), e)

        # the provided jacobian seems to be inaccurate resulting in slight movement of the ee. This is why
        # the velocites are set to 0 once the error step changing much.
        if setup["cut-off_error"] is not None:
            if self._last_e is not None:
                e_dot = np.sum(np.abs(e - self._last_e))
            if self._last_e is not None and e_dot < setup["cut-off_error"]:
                q_dot_redundancy = np.array([0.0] * 7)
                self._last_e = e
            else:
                self._last_e = e

        return actions - q_dot_redundancy

    def get_error_reference_position(self, ref_pos,  W):
        """
        Calculates the error for redundancy resoltuion with respect to a reference position.
        :param ref_pos:
         Reference position.
        :param W:
         Weighting vector.
        :return:
         Weighted error vector.
        """
        if W is None:
            # default weighting
            W = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        return (self._robot.arm.get_joint_positions() - ref_pos) * W

    def get_error_collision_avoidance(self, W):
        """
        Calculates the error for redundancy resoltuion with collision avoidance. This only works with tasks with obstacles!
        :param W:
         Weighting vector.
        :return:
         Weighted error vector.
        """

        # get the position of the object
        p_obs = self._task.obstacle.get_position()  + np.array([0, 0, 0.33]) -  self._robot.arm.joints[0].get_position()
        p_obs = np.append(p_obs, [1])

        # get the transformation matrices, their derivatives, and the positions of the links
        A_1, A_2, A_3, A_4, A_5, A_6, A_7 = self._robot.get_transformation_matrices()
        dA_1, dA_2, dA_3, dA_4, dA_5, dA_6, dA_7 = self._robot.get_transformation_matrices_derivatives()
        p_1, p_2, p_3, p_4, p_5, p_6, p_7 = self._robot.get_link_positions_in_ref_frames()


        # we use reciprocal of the distance between each link and an obstacle as our Loss
        # the chain rule delivers: d/dq L = (p_i^0 (q_1,..., q_i) - p_obs)^T * d/dq (p_i^0 (q_1,..., q_i) - p_obs)
        # where p_i^0 = (\prod_{j=1}^i A_j^{j-1}(q_j)) * p_i
        # as the left side of d/dq L is used often, let's calculate it in advance
        d_1_T = np.transpose(A_1.dot(p_1) - p_obs)
        d_2_T = np.transpose(A_1.dot(A_2).dot(p_2) - p_obs)
        d_3_T = np.transpose(A_1.dot(A_2).dot(A_3).dot(p_3) - p_obs)
        d_4_T = np.transpose(A_1.dot(A_2).dot(A_3).dot(A_4).dot(p_4) - p_obs)
        d_5_T = np.transpose(A_1.dot(A_2).dot(A_3).dot(A_4).dot(A_5).dot(p_5) - p_obs)
        d_6_T = np.transpose(A_1.dot(A_2).dot(A_3).dot(A_4).dot(A_5).dot(A_6).dot(p_6) - p_obs)
        d_7_T = np.transpose(A_1.dot(A_2).dot(A_3).dot(A_4).dot(A_5).dot(A_6).dot(A_7).dot(p_7) - p_obs)



        # now we can calculate the error in each dimension
        e_1 = -np.matmul(d_1_T, dA_1.dot(p_1))  + \
              -np.matmul(d_2_T, dA_1.dot(A_2).dot(p_2)) + \
              -np.matmul(d_3_T, dA_1.dot(A_2).dot(A_3).dot(p_3)) + \
              -np.matmul(d_4_T, dA_1.dot(A_2).dot(A_3).dot(A_4).dot(p_4)) + \
              -np.matmul(d_5_T, dA_1.dot(A_2).dot(A_3).dot(A_4).dot(A_5).dot(p_5)) + \
              -np.matmul(d_6_T, dA_1.dot(A_2).dot(A_3).dot(A_4).dot(A_5).dot(A_6).dot(p_6)) + \
              -np.matmul(d_7_T, dA_1.dot(A_2).dot(A_3).dot(A_4).dot(A_5).dot(A_6).dot(A_7).dot(p_7))
        e_2 = -np.matmul(d_2_T, A_1.dot(dA_2).dot(p_2)) + \
              -np.matmul(d_3_T, A_1.dot(dA_2).dot(A_3).dot(p_3)) + \
              -np.matmul(d_4_T, A_1.dot(dA_2).dot(A_3).dot(A_4).dot(p_4)) + \
              -np.matmul(d_5_T, A_1.dot(dA_2).dot(A_3).dot(A_4).dot(A_5).dot(p_5)) + \
              -np.matmul(d_6_T, A_1.dot(dA_2).dot(A_3).dot(A_4).dot(A_5).dot(A_6).dot(p_6)) + \
              -np.matmul(d_7_T, A_1.dot(dA_2).dot(A_3).dot(A_4).dot(A_5).dot(A_6).dot(A_7).dot(p_7))
        e_3 = -np.matmul(d_3_T, A_1.dot(A_2).dot(dA_3).dot(p_3)) + \
              -np.matmul(d_4_T, A_1.dot(A_2).dot(dA_3).dot(A_4).dot(p_4)) + \
              -np.matmul(d_5_T, A_1.dot(A_2).dot(dA_3).dot(A_4).dot(A_5).dot(p_5)) + \
              -np.matmul(d_6_T, A_1.dot(A_2).dot(dA_3).dot(A_4).dot(A_5).dot(A_6).dot(p_6)) + \
              -np.matmul(d_7_T, A_1.dot(A_2).dot(dA_3).dot(A_4).dot(A_5).dot(A_6).dot(A_7).dot(p_7))
        e_4 = -np.matmul(d_4_T, A_1.dot(A_2).dot(A_3).dot(dA_4).dot(p_4)) + \
              -np.matmul(d_5_T, A_1.dot(A_2).dot(A_3).dot(dA_4).dot(A_5).dot(p_5)) + \
              -np.matmul(d_6_T, A_1.dot(A_2).dot(A_3).dot(dA_4).dot(A_5).dot(A_6).dot(p_6)) + \
              -np.matmul(d_7_T, A_1.dot(A_2).dot(A_3).dot(dA_4).dot(A_5).dot(A_6).dot(A_7).dot(p_7))
        e_5 = -np.matmul(d_5_T, A_1.dot(A_2).dot(A_3).dot(A_4).dot(dA_5).dot(p_5)) + \
              -np.matmul(d_6_T, A_1.dot(A_2).dot(A_3).dot(A_4).dot(dA_5).dot(A_6).dot(p_6)) + \
              -np.matmul(d_7_T, A_1.dot(A_2).dot(A_3).dot(A_4).dot(dA_5).dot(A_6).dot(A_7).dot(p_7))
        e_6 = -np.matmul(d_6_T, A_1.dot(A_2).dot(A_3).dot(A_4).dot(A_5).dot(dA_6).dot(p_6)) + \
              -np.matmul(d_7_T, A_1.dot(A_2).dot(A_3).dot(A_4).dot(A_5).dot(dA_6).dot(A_7).dot(p_7))
        e_7 = -np.matmul(d_7_T, A_1.dot(A_2).dot(A_3).dot(A_4).dot(A_5).dot(A_6).dot(dA_7).dot(p_7))

        if W is None:
            # default weighting vector -> based on the reciprocal of the distance. The greater the distance the smaller
            # the weight. That is, it is concentrated on close objects.
            W = np.array([1 / np.sum(np.square(d_1_T)),
                          1 / np.sum(np.square(d_2_T)) ,
                          1 / np.sum(np.square(d_3_T)) ,
                          1 / np.sum(np.square(d_4_T)) ,
                          1 / np.sum(np.square(d_5_T)) ,
                          1 / np.sum(np.square(d_6_T)) ,
                          1 / np.sum(np.square(d_7_T)) ]) * 0.1

        # concatenate errors to error vector and apply weightig
        e = np.array([e_1, e_2, e_3, e_4, e_5, e_6, e_6])*W

        return e

    def enable_path_observations(self, value: bool) -> None:
        if (self._action_mode.arm != ArmActionMode.DELTA_EE_POSE_PLAN_WORLD_FRAME and
                self._action_mode.arm != ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME and
                self._action_mode.arm != ArmActionMode.EE_POSE_PLAN_EE_FRAME):
            raise RuntimeError('Only available in DELTA_EE_POSE_PLAN or '
                               'ABS_EE_POSE_PLAN action mode.')
        self._enable_path_observations = value

    def get_path_observations(self):
        if (self._action_mode.arm != ArmActionMode.DELTA_EE_POSE_PLAN_WORLD_FRAME and
                self._action_mode.arm != ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME and
                self._action_mode.arm != ArmActionMode.EE_POSE_PLAN_EE_FRAME):
            raise RuntimeError('Only available in DELTA_EE_POSE_PLAN or '
                               'ABS_EE_POSE_PLAN action mode.')
        return self._path_observations

    def get_demos(self, amount: int, live_demos: bool = False,
                  image_paths: bool = False,
                  callable_each_step: Callable[[Observation], None] = None,
                  max_attempts: int = _MAX_DEMO_ATTEMPTS,
                  ) -> List[Demo]:
        """Negative means all demos"""

        if not live_demos and (self._dataset_root is None
                       or len(self._dataset_root) == 0):
            raise RuntimeError(
                "Can't ask for a stored demo when no dataset root provided.")

        if not live_demos:
            if self._dataset_root is None or len(self._dataset_root) == 0:
                raise RuntimeError(
                    "Can't ask for stored demo when no dataset root provided.")
            demos = utils.get_stored_demos(
                amount, image_paths, self._dataset_root, self._variation_number,
                self._task.get_name(), self._obs_config)
        else:
            ctr_loop = self._robot.arm.joints[0].is_control_loop_enabled()
            self._robot.arm.set_control_loop_enabled(True)
            demos = self._get_live_demos(
                amount, callable_each_step, max_attempts)
            self._robot.arm.set_control_loop_enabled(ctr_loop)
        return demos

    def _get_live_demos(self, amount: int,
                        callable_each_step: Callable[
                            [Observation], None] = None,
                        max_attempts: int = _MAX_DEMO_ATTEMPTS) -> List[Demo]:
        demos = []
        for i in range(amount):
            attempts = max_attempts
            while attempts > 0:
                random_seed = np.random.get_state()
                self.reset()
                logging.info('Collecting demo %d' % i)
                try:
                    demo = self._scene.get_demo(
                        callable_each_step=callable_each_step)
                    demo.random_seed = random_seed
                    demos.append(demo)
                    break
                except Exception as e:
                    attempts -= 1
                    logging.info('Bad demo. ' + str(e))
            if attempts <= 0:
                raise RuntimeError(
                    'Could not collect demos. Maybe a problem with the task?')
        return demos

    def reset_to_demo(self, demo: Demo) -> (List[str], Observation):
        demo.restore_state()
        return self.reset()
