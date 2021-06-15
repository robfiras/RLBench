from pyrep.robots.arms.arm import Arm
from pyrep.robots.end_effectors.gripper import Gripper
import numpy as np

class Robot(object):
    """Simple container for the robot components.
    """

    def __init__(self, arm: Arm, gripper: Gripper):
        self.arm = arm
        self.gripper = gripper

        # DH parameters of the Panda
        # https://frankaemika.github.io/docs/control_parameters.html
        self.d_1 = 0.33
        self.d_3 = 0.316
        self.d_5 = 0.384
        self.d_f = 0.107
        self.a_4 = 0.0825
        self.a_5 = -0.0825
        self.a_7 = 0.088

        # positions of the links in the respective frames (defined for redundancy resolution with collision avoidance)
        # Note: 1 is appended for multiplication with transformation matrix later
        self.p_1 = np.array([0.0, 0.0, -self.d_1/2, 1])
        self.p_2 = np.array([0.0, -self.d_3/3, 0, 1])
        self.p_3 = np.array([0.0, 0.0, -self.d_3/3, 1])
        self.p_4 = np.array([0.0, 0.0, 0, 1])
        self.p_5 = np.array([0.0, 0.0, -self.d_5/2, 1])
        self.p_6 = np.array([0.0, 0.0, 0.0, 1])
        self.p_7 = np.array([0.0, 0.0, 0.0, 1])

    def get_base_coordindates_in_world(self):
        return self.arm.joints[0].get_position() - np.array([0, 0, self.d_1])

    def get_link_positions_in_ref_frames(self):
        return self.p_1, self.p_2, self.p_3, self.p_4, self.p_5, self.p_6, self.p_7

    def get_transformation_matrices(self):

        # get the current joint positions
        q1 = self.arm.joints[0].get_joint_position()
        q2 = self.arm.joints[1].get_joint_position()
        q3 = self.arm.joints[2].get_joint_position()
        q4 = self.arm.joints[3].get_joint_position()
        q5 = self.arm.joints[4].get_joint_position()
        q6 = self.arm.joints[5].get_joint_position()
        q7 = self.arm.joints[6].get_joint_position()

        # transformation matrices generated based on the DH parameters
        # Note: This is following the Craig convention
        A_1 = np.array([[np.cos(q1), -np.sin(q1), 0, 0],
                        [np.sin(q1), np.cos(q1), 0, 0],
                        [0, 0, 1, self.d_1],
                        [0, 0, 0, 1]])

        A_2 = np.array([[np.cos(q2), -np.sin(q2), 0, 0],
                        [0, 0, 1, 0],
                        [-np.sin(q2), -np.cos(q2), 0, 0],
                        [0, 0, 0, 1]])

        A_3 = np.array([[np.cos(q3), -np.sin(q3), 0, 0],
                        [0, 0, -1, -self.d_3],
                        [np.sin(q3), np.cos(q3), 0, 0],
                        [0, 0, 0, 1]])

        A_4 = np.array([[np.cos(q4), -np.sin(q4), 0, self.a_4],
                        [0, 0, -1, 0],
                        [np.sin(q4), np.cos(q4), 0, 0],
                        [0, 0, 0, 1]])

        A_5 = np.array([[np.cos(q5), -np.sin(q5), 0, self.a_5],
                        [0, 0, 1, self.d_5],
                        [-np.sin(q5), -np.cos(q5), 0, 0],
                        [0, 0, 0, 1]])

        A_6 = np.array([[np.cos(q6), -np.sin(q6), 0, 0],
                        [0, 0, -1, 0],
                        [np.sin(q6), np.cos(q6), 0, 0],
                        [0, 0, 0, 1]])

        A_7 = np.array([[np.cos(q7), -np.sin(q7), 0, self.a_7],
                        [0, 0, -1, 0],
                        [np.sin(q7), np.cos(q7), 0, 0],
                        [0, 0, 0, 1]])

        return A_1, A_2, A_3, A_4, A_5, A_6, A_7

    def get_transformation_matrices_derivatives(self):

        # get the current joint positions
        q1 = self.arm.joints[0].get_joint_position()
        q2 = self.arm.joints[1].get_joint_position()
        q3 = self.arm.joints[2].get_joint_position()
        q4 = self.arm.joints[3].get_joint_position()
        q5 = self.arm.joints[4].get_joint_position()
        q6 = self.arm.joints[5].get_joint_position()
        q7 = self.arm.joints[6].get_joint_position()

        # transformation matrices generated based on the DH parameters
        # Note: This is following the Craig convention
        # derivatives of transformation matrices with respect to their joint positions
        dA_1 = np.array([[-np.sin(q1), -np.cos(q1), 0, 0],
                        [np.cos(q1), -np.sin(q1), 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])

        dA_2 = np.array([[-np.sin(q2), -np.cos(q2), 0, 0],
                        [0, 0, 0, 0],
                        [-np.cos(q2), np.sin(q2), 0, 0],
                        [0, 0, 0, 0]])

        dA_3 = np.array([[-np.sin(q3), -np.cos(q3), 0, 0],
                        [0, 0, 0, 0],
                        [np.cos(q3), -np.sin(q3), 0, 0],
                        [0, 0, 0, 0]])

        dA_4 = np.array([[-np.sin(q4), -np.cos(q4), 0, 0],
                        [0, 0, 0, 0],
                        [np.cos(q4), -np.sin(q4), 0, 0],
                        [0, 0, 0, 0]])

        dA_5 = np.array([[-np.sin(q5), -np.cos(q5), 0, 0],
                        [0, 0, 0, 0],
                        [-np.cos(q5), np.sin(q5), 0, 0],
                        [0, 0, 0, 0]])

        dA_6 = np.array([[-np.sin(q6), -np.cos(q6), 0, 0],
                        [0, 0, 0, 0],
                        [np.cos(q6), -np.sin(q6), 0, 0],
                        [0, 0, 0, 0]])

        dA_7 = np.array([[-np.sin(q7), -np.cos(q7), 0, 0],
                        [0, 0, 0, 0],
                        [np.cos(q7), -np.sin(q7), 0, 0],
                        [0, 0, 0, 0]])

        return dA_1, dA_2, dA_3, dA_4, dA_5, dA_6, dA_7


