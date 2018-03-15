#!/usr/bin/env python

"""This class represents a simple pure-Python particle filter.  It should
serve as a (mostly) drop-in replacement for the default ROS
localization node: http://wiki.ros.org/amcl

    Subscribes to:
          scan           (sensor_msgs/LaserScan)
          map            (nav_msgs/OccupancyGrid)
          odom           (nav_msgs/Odometry)
          initialpose    (geometry_msgs/PoseWithCovarianceStamped)

    Publishes to:
          particlecloud  (geometry_msgs/PoseArray)
          amcl_pose      (geometry_msgs/PoseWithCovarianceStamped)

    ROS Parameters:
      ~num_particles (int, default: 100)

      ~initial_pose_x (double, default: 0.0 meters)
         Initial pose mean (x), used to initialize filter with
         Gaussian distribution.

      ~initial_pose_y (double, default: 0.0 meters)
         Initial pose mean (y), used to initialize filter with
         Gaussian distribution.

      ~initial_pose_a (double, default: 0.0 radians)
         Initial pose mean (yaw), used to initialize filter with
         Gaussian distribution.

      ~initial_cov_xx (double, default: 0.5*0.5 meters)
         Initial pose covariance (x*x), used to initialize filter with
         Gaussian distribution.

      ~initial_cov_yy (double, default: 0.5*0.5 meters)
         Initial pose covariance (y*y), used to initialize filter with
         Gaussian distribution.

      ~initial_cov_aa (double, default: (pi/12)*(pi/12) radian)
         Initial pose covariance (yaw*yaw), used to initialize filter
         with Gaussian distribution.

      The remaining parameters govern the behavior of the particles.

      ~laser_z_hit (double, default: 0.95)
         Mixture weight for the z_hit part of the model.

      ~laser_z_rand (double, default: 0.05)
         Mixture weight for the z_rand part of the model.

      ~laser_sigma_hit (double, default: 0.2 meters)
         Standard deviation for Gaussian model used in z_hit part of
         the model.

      ~odom_var_pos (double, default: 0.1 meters)
         Standard deviation for noise in x and y coordinates of motion
         model.

      ~odom_var_theta (double, default: 0.05)
         Standard deviation for noise in rotation of motion model.

Some of the documentation for this Node is borrowed from
http://wiki.ros.org/amcl
http://creativecommons.org/licenses/by/3.0/
"""
import rospy
import tf
import numpy as np
import particle

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, PoseArray
from geometry_msgs.msg import PoseWithCovarianceStamped

class MCLNode(object):
    """
    This class represents the particle filter.
    """

    def __init__(self):
        """ Set up the node, publishers and subscribers. """
        rospy.init_node('mc_localizer')

        # Get values from ROS parameters:
        self.num_particles = rospy.get_param('~num_particles', 100)
        self.laser_z_hit = rospy.get_param('~laser_z_hit', .1)
        self.laser_z_rand = rospy.get_param('~laser_z_rand', .9)
        self.laser_sigma_hit = rospy.get_param('~laser_sigma_hit', .2)
        self.odom_var_pos = rospy.get_param('~odom_var_pose', .1)
        self.odom_var_theta = rospy.get_param('~odom_var_theta', .05)

        self.map = None
        self.particles = None

        rospy.Subscriber('map', OccupancyGrid, self.map_callback)
        rospy.Subscriber('initialpose', PoseWithCovarianceStamped,
                         self.initial_pose_callback)
	rospy.Subscriber('scan', LaserScan, self.scan_callback)
	rospy.Subscriber('odom', Odometry, self.odom_callback)
	
	
        self.particle_pub = rospy.Publisher('particlecloud', PoseArray)
        self.likelihood_pub = rospy.Publisher('likelihood_field',
                                              OccupancyGrid, latch=True)
	self.pose_pub = rospy.Publisher('amcl_pose', PoseWithCovarianceStamped)
        # We need the map before we can initialize the particles.
        while not rospy.is_shutdown() and self.map is None:
            rospy.loginfo("Waiting for map.")
            rospy.sleep(.5)

        initial_pose_x = rospy.get_param('~initial_pose_x', 0.0)
        initial_pose_y = rospy.get_param('~initial_pose_y', 0.0)
        initial_pose_a = rospy.get_param('~initial_pose_a', 0.0)
        initial_cov_xx = rospy.get_param('~initial_cov_xx', .25)
        initial_cov_yy = rospy.get_param('~initial_cov_yy', .25)
        initial_cov_aa = rospy.get_param('~initial_cov_aa', .07)
        self.initialize_particles(initial_pose_x, initial_pose_y,
                                  initial_pose_a, initial_cov_xx,
                                  initial_cov_yy, initial_cov_aa)

        # Publish the likelihood field.  This is just for debugging
        # purposes.
        self.likelihood_pub.publish(\
            particle.Particle.likelihood_field.to_message())

        # Now enter an infinite loop. Execution will be callbacks.
        rospy.spin()


    def initial_pose_callback(self, initial_pose):
        """This will be called when a new initial pose is provided through
        RViz.  We re-initialize the particle set based on the provided
        pose.

        """
        x = initial_pose.pose.pose.position.x
        y = initial_pose.pose.pose.position.y
        theta = theta_from_pose(initial_pose.pose.pose)
        x_var = initial_pose.pose.covariance[0]
        y_var = initial_pose.pose.covariance[7]
        theta_var = initial_pose.pose.covariance[35]
        self.initialize_particles(x, y, theta, x_var, y_var, theta_var)

    def initialize_particles(self, init_x, init_y, init_theta, x_var,
                             y_var, theta_var):
        """Create an initial set of particles.  The positions and rotations of
        each particle will be randomly drawn from normal distributions
        based on the input arguments.
        """
        self.particles = []
        for _ in range(self.num_particles):
            x = init_x + np.random.randn() * x_var
            y = init_y + np.random.randn() * y_var
            theta = init_theta + np.random.randn() * theta_var
            new_particle = particle.Particle(x, y, theta,
                                             1.0 / self.num_particles,
                                             self.laser_z_hit,
                                             self.laser_z_rand,
                                             self.laser_sigma_hit,
                                             self.odom_var_pos,
                                             self.odom_var_theta,
                                             self.map)
            self.particles.append(new_particle)

        # Publish the newly created particles.
        pose_array = self.create_pose_array_msg()
        self.particle_pub.publish(pose_array)



    def create_pose_array_msg(self):
        """Create a PoseArray object including the poses of all particles.

        Returns: geometry_msgs/PoseArray
        """
        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = 'map'
        pose_array.poses = []
        for p in self.particles:
            pose_array.poses.append(p.pose)
        return pose_array

    def map_callback(self, map_msg):
        """ Store the map message in an instance variable. """
        self.map = map_msg

   def scan_callback():
	scan_markers()

   def odom_callback():



def theta_from_pose(pose):
    """ Utility method to extract the theta/yaw from a Pose object.

    Arguments:
       pose: geometry_msgs/Pose

    Returns: Theta in radians as a float.
    """

    quat = np.array([pose.orientation.x,
                     pose.orientation.y,
                     pose.orientation.z,
                     pose.orientation.w])
    euler = tf.transformations.euler_from_quaternion(quat)
    return euler[2]

if __name__ == "__main__":
    MCLNode()
