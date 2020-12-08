import math
import random
from typing import List
import numpy as np
from pyrep.objects.object import Object
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.vision_sensor import VisionSensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, ConditionSet, \
    GraspedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.const import colors


class CustomPickAndLift(Task):

    def init_task(self) -> None:
        self.target_block = Shape('pick_and_lift_target')
        self.target = Shape("success_visual")
        self.target.set_renderable(False)
        self.register_graspable_objects([self.target_block])
        self.boundary = SpawnBoundary([Shape('pick_and_lift_boundary')])
        self.success_detector = ProximitySensor('pick_and_lift_success')
        self.front_camera_exists = Object.exists('cam_front')
        if self.front_camera_exists:
            self.front_camera = VisionSensor('cam_front')
            self.init_front_camera_position = self.front_camera.get_position()
            self.init_front_camera_orientation = self.front_camera.get_orientation()
            self.panda_base =Shape("Panda_link0_visual")

        cond_set = ConditionSet([
            GraspedCondition(self.robot.gripper, self.target_block),
            DetectedCondition(self.target_block, self.success_detector)
        ])
        self.register_success_conditions([cond_set])

    def init_episode(self, index: int) -> List[str]:

        block_color_name, block_rgb = colors[index]
        self.target_block.set_color(block_rgb)

        self.boundary.clear()
        self.boundary.sample(
            self.success_detector, min_rotation=(0.0, 0.0, 0.0),
            max_rotation=(0.0, 0.0, 0.0))
        for block in [self.target_block]:
            # block is symteric -> limit rotation range to pi/4 = 45° to avoid ambiguous poses
            self.boundary.sample(block, min_distance=0.1,
                                 min_rotation=(0, 0, -math.pi/8),
                                 max_rotation=(0,0, math.pi/8))

        if self.front_camera_exists:
            # apply new position to front camera for better generalization
            self.front_camera_new_position()

        return ['pick up the %s block and lift it up to the target' %
                block_color_name,
                'grasp the %s block to the target' % block_color_name,
                'lift the %s block up to the target' % block_color_name]

    def variation_count(self) -> int:
        return len(colors)

    def step(self) -> None:
        if self.front_camera_exists:
            # apply new position to front camera for better generalization
            self.front_camera_new_position()

    def get_low_dim_state(self) -> np.ndarray:
        # get pose from target block
        block_position = self.target_block.get_pose()
        # get x, y, z from target position
        target_position = self.target.get_position()
        return np.concatenate((block_position, target_position))

    def reward(self):
        """ the custom reward consists of two parts:
            1. distance between the gripper and the target_block
            2. distance between the target_block and the target (lift_target)

            the total reward is the sum of both parts"""
        max_precision = 0.01  # 1cm
        max_reward = 1 / max_precision
        scale = 0.1

        # fist part
        gripper_position = self.robot.arm.get_tip().get_position()
        target_block_position = self.target_block.get_position()
        dist = np.sqrt(np.sum(np.square(np.subtract(target_block_position, gripper_position)), axis=0))  # euclidean norm
        reward1 = min((1 / (dist + 0.00001)), max_reward)
        reward1 = scale * reward1

        # second part
        target_position = self.target.get_position()
        dist = np.sqrt(np.sum(np.square(np.subtract(target_position, target_block_position)), axis=0))  # euclidean norm
        reward2 = min((1 / (dist + 0.00001)), max_reward)
        reward2 = scale * reward2

        return reward1 + reward2

    def front_camera_new_position(self):
        """ changes the position of the front camera to a new one, and rotates the camera in a way that it sees the
         target-block and the Panda-base. """

        # reset camera position
        self.front_camera.set_position(self.init_front_camera_position)

        # --- move front camera to new position ---
        # calculate the new (random) position
        new_rel_front_camera_pos = np.random.uniform(low=-0.5, high=0.5, size=(3,))

        # move camera
        self.front_camera.set_position(new_rel_front_camera_pos, relative_to=self.front_camera)

        # --- turn the camera ---
        # -> camera should be turned to look at the new position -> mean of Panda_base and target_block
        # get the coordinates
        tar_block_obj_pos = self.target_block.get_position(relative_to=self.front_camera)
        panda_base_pos = self.panda_base.get_position(relative_to=self.front_camera)

        # calculate the mean to get the position to look at
        look_at = np.mean( np.array([tar_block_obj_pos, panda_base_pos]), axis=0 )

        # rotation arround y
        rot_y = math.atan2(look_at[0], look_at[2])

        # rotation arround x
        rot_x = math.atan2(look_at[1], math.sqrt( math.pow(look_at[0], 2) + math.pow(look_at[2], 2) ) )

        # calculate random rotation around z
        rot_z = random.uniform(-math.pi, math.pi)

        # apply rotation
        old_orientation = self.front_camera.get_orientation(relative_to=self.front_camera)
        self.front_camera.set_orientation([old_orientation[0] - rot_x, old_orientation[1] + rot_y, rot_z],
                                          relative_to=self.front_camera)