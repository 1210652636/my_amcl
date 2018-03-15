#!/usr/bin/env python

"""
    This node subscribes to both the ground truth and estimated poses.  It logs
    information about the quality of localization.

    Subscribes to:
          /ground_truth_pose (geometry_msgs/PoseStamped)
          /amcl_pose (geometry_msgs/PoseStamped)

"""
import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped

class EvaluationNode(object):
    """
    """

    def __init__(self):
        """ Set up the node, publishers and subscribers. """
        rospy.init_node('evaluate_node')

        rospy.Subscriber('/amcl_pose', PoseStamped,
                         self.amcl_callback)
        rospy.Subscriber('/ground_truth_pose', PoseStamped,
                         self.ground_truth_callback)

        self.amcl_pose = None
        self.ground_truth_pose = None
        self.distances = []


        self.offset = None
        rospy.spin()

    def ground_truth_callback(self, ground_truth_pose):
        self.ground_truth_pose = ground_truth_pose

    def amcl_callback(self, amcl_pose):
        dx = amcl_pose.pose.position.x - self.ground_truth_pose.pose.position.x
        dy = amcl_pose.pose.position.y - self.ground_truth_pose.pose.position.y
        self.distances.append(np.sqrt(dx**2 + dy**2))
        if len(self.distances) > 100 and len(self.distances) % 10 == 0:
            rospy.loginfo("MAX:     {:.3f}".format(np.max(self.distances[99:])))
            rospy.loginfo("AVERAGE: {:.3f}".format(np.mean(self.distances)))

if __name__ == "__main__":
    EvaluationNode()
