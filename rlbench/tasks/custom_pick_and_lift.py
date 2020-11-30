from typing import List
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, ConditionSet, \
    GraspedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.const import colors


class CustomPickAndLift(Task):

    def init_task(self) -> None:
        self.target_block = Shape('pick_and_lift_target')
        self.target = Shape("success_visual")
        self.register_graspable_objects([self.target_block])
        self.boundary = SpawnBoundary([Shape('pick_and_lift_boundary')])
        self.success_detector = ProximitySensor('pick_and_lift_success')

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
            self.boundary.sample(block, min_distance=0.1)

        return ['pick up the %s block and lift it up to the target' %
                block_color_name,
                'grasp the %s block to the target' % block_color_name,
                'lift the %s block up to the target' % block_color_name]

    def variation_count(self) -> int:
        return len(colors)

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