#!/usr/bin/env python
"""This node broadcasts an appropriate tranform from the odom frame to the
map frame based on the the pose estimate published to amcl_pose.

   Subscribes to: 
     amcl_pose  (geometry_msgs/PoseWithCovarianceStamped)

"""
import rospy
import tf
import numpy as np
from geometry_msgs.msg import PoseStamped
import tf_conversions.posemath as posemath

class MCLTf(object):
    """  Node for updating tf based on published pose estimate. 
    """
    def __init__(self):
        """ Initialize the mcl_tf node. """

        rospy.init_node('mcl_tf')
        br = tf.TransformBroadcaster()
        self.tf_listener =  tf.TransformListener()
        
        # Give the listener some time to accumulate transforms... 
        rospy.sleep(1.0) 

        rospy.Subscriber('amcl_pose', PoseStamped, self.pose_callback)

        self.transform_position = np.array([0., 0., 0.])
        self.transform_quaternion = np.array([0., 0., 0., 1.0])
        
        # Broadcast the transform at 10 HZ
        while not rospy.is_shutdown():
            br.sendTransform(self.transform_position,
                             self.transform_quaternion,
                             rospy.Time.now(),
                             "odom",
                             "map")
            rospy.sleep(.1)

    def pose_callback(self, pose):
        # This code is based on:
        # https://github.com/ros-planning/navigation/blob/jade-devel\
        #              /amcl/src/amcl_node.cpp
        
        try:
            self.tf_listener.waitForTransform('map',  # from here
                                              'odom', # to here
                                              pose.header.stamp, 
                                              rospy.Duration(1.0))
            frame = posemath.fromMsg(pose.pose).Inverse()
            pose.pose = posemath.toMsg(frame)
            pose.header.frame_id = 'base_link'

            odom_pose = self.tf_listener.transformPose('odom', 
                                                       pose)
            frame = posemath.fromMsg(odom_pose.pose).Inverse()
            odom_pose.pose = posemath.toMsg(frame)

            self.transform_position[0] = odom_pose.pose.position.x
            self.transform_position[1] = odom_pose.pose.position.y
            self.transform_quaternion[0] = odom_pose.pose.orientation.x
            self.transform_quaternion[1] = odom_pose.pose.orientation.y
            self.transform_quaternion[2] = odom_pose.pose.orientation.z
            self.transform_quaternion[3] = odom_pose.pose.orientation.w
            
        except tf.Exception as e:
            print e
            print "(May not be a big deal.)"

if __name__ == '__main__':
    try:
        td = MCLTf()
    except rospy.ROSInterruptException:
        pass
