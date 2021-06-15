from typing import List, Tuple
import numpy as np
import math
import random
from pyrep.objects.shape import Shape
from pyrep.objects.object import Object
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import ObjectType, PrimitiveShape
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import DetectedCondition


class ReachTargetWithObstacles(Task):

    def init_task(self) -> None:
        self.target = Shape('target')
        self.distractor0 = Shape('distractor0')
        self.distractor1 = Shape('distractor1')

        self.target.set_renderable(True)
        self.distractor0.set_renderable(False)
        self.distractor1.set_renderable(False)
        self.front_camera_exists = Object.exists('cam_front')
        if self.front_camera_exists:
            self.front_camera = VisionSensor('cam_front')
        #curr_pos = self.front_camera.get_position()
        # new_pos = curr_pos + np.array([-1.25, 1.6 , -0.35])
        # self.front_camera.set_position(new_pos)
        # curr_ori = self.front_camera.get_orientation()
        # new_ori = np.array([1.6, 0, 0])
        # self.front_camera.set_orientation(new_ori)

        # front pos
        #new_pos = curr_pos + np.array([0.0, 0.0 , -0.2])
        #self.front_camera.set_position(new_pos)
        #curr_ori = self.front_camera.get_orientation()
        #new_ori = curr_ori + np.array([0, -0.25, 0])
        #self.front_camera.set_orientation(new_ori)
        self.step_in_episode = 0

        self.obstacle = Shape.create(type=PrimitiveShape.SPHERE,
                                     size=[0.3, 0.3, 0.3],
                                     renderable=True,
                                     respondable=True,
                                     static=True,
                                     position=[0.1, 0.1, 1.4],
                                     color=[0,0,0])


        self.boundaries = Shape('boundary')
        success_sensor = ProximitySensor('success')
        self.register_success_conditions(
            [DetectedCondition(self.robot.arm.get_tip(), success_sensor)])

    def init_episode(self, index: int) -> List[str]:
        color_name, color_rgb = colors[index]
        self.target.set_color(color_rgb)
        color_choices = np.random.choice(
            list(range(index)) + list(range(index + 1, len(colors))),
            size=2, replace=False)
        for ob, i in zip([self.distractor0, self.distractor1], color_choices):
            name, rgb = colors[i]
            ob.set_color(rgb)
        b = SpawnBoundary([self.boundaries])
        for ob in [self.target, self.distractor0, self.distractor1]:
            b.sample(ob, min_distance=0.2,
                     min_rotation=(0, 0, 0), max_rotation=(0, 0, 0))

        # change the position of the robot
        q = self.robot.arm.get_joint_positions()
        new_q = np.array([1.2, 0, 0, 0, 0, 0, 0]) + q
        self.robot.arm.set_joint_positions(new_q)

        self.step_in_episode = 0

        return ['reach the %s target' % color_name,
                'touch the %s ball with the panda gripper' % color_name,
                'reach the %s sphere' %color_name]

    # def step(self) -> None:
    #
    #     if self.step_in_episode == 0:
    #         rand_pos = np.array([np.random.uniform(0, +0.5), np.random.uniform(-0.5, 0.5), np.random.uniform(0.77, 1.5)])
    #         self.obstacle.set_position(rand_pos)
    #     self.step_in_episode += 1

    def variation_count(self) -> int:
        return len(colors)

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    
    def get_low_dim_state(self) -> np.ndarray:
        # One of the few tasks that have a custom low_dim_state function.
        return np.array(self.target.get_position())

    def is_static_workspace(self) -> bool:
        return True

    def reward(self):
        """ Calculates the reward for the ReachTargetWithObstacles Task based on the euclidean distance. """
        max_precision = 0.01  # 1cm
        max_reward = 1 / max_precision
        scale = 0.1
        gripper_position = self.robot.arm.get_tip().get_position()
        target_position = self.target.get_position()
        dist = np.sqrt(np.sum(np.square(np.subtract(target_position, gripper_position)), axis=0))  # euclidean norm
        reward = min((1 / (dist + 0.00001)), max_reward)
        reward = scale * reward
        return reward

    def look_at(self, look_at):

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
