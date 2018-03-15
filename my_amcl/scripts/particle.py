#!/usr/bin/env python

"""Particle class to support Monte-Carlo localization.
"""
import rospy
import tf
import map_utils
import numpy as np
import copy

from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker, MarkerArray
import tf_conversions.posemath as posemath

class Particle(object):
    """Particles are essentially wrappers around geometry_msgs/Pose
       objects.
    """

    # This is a class-variable because it makes sense for all particles
    # to share the same map.
    likelihood_field = None

    def __init__(self, x, y, theta, weight, laser_z_hit, laser_z_rand,
                 laser_sigma_hit, odom_var_pos, odom_var_theta,
                 map_msg=None):
        """ Initialize the particle.  """

        self.weight = weight
        self.laser_z_hit = laser_z_hit
        self.laser_z_rand = laser_z_rand
        self.laser_sigma_hit = laser_sigma_hit
        self.odom_var_pos = odom_var_pos
        self.odom_var_theta = odom_var_theta
        self._pose = Pose()

        # The next three lines are calling the property setters.
        self.x = x
        self.y = y
        self.theta = theta

        if Particle.likelihood_field is None:
            self.update_likelihood_field(map_msg, self.laser_sigma_hit)

    def update_likelihood_field(self, map_msg, laser_sigma):
        """The likelihood field is essentially a map indicating which
        locations are likely to result in a laser hit.  Points that
        are occupied in map_msg have the highest probability, points
        that are farther away from occupied regions have a lower
        probability.  The laser_sigma argument controls how quickly
        the probability falls off.  Ideally, it should be related to
        the precision of the laser scanner being used.

        """

        rospy.loginfo('building Likelihood map...')
        world_map = map_utils.Map(map_msg)

        rospy.loginfo('building KDTree')
        from sklearn.neighbors import KDTree
        occupied_points = []
        all_positions = []
        for i in range(world_map.grid.shape[0]):
            for j in range(world_map.grid.shape[1]):
                all_positions.append(world_map.cell_position(i, j))
                if world_map.grid[i, j] > .9:
                    occupied_points.append(world_map.cell_position(i, j))

        kdt = KDTree(occupied_points)

        rospy.loginfo('Constructing likelihood field from KDTree.')
        likelihood_field = map_utils.Map(world_map.to_message())
        dists = kdt.query(all_positions, k=1)[0][:]
        probs = np.exp(-(dists**2) / (2 * laser_sigma**2))

        likelihood_field.grid = probs.reshape(likelihood_field.grid.shape)

        rospy.loginfo('Done building likelihood field')
        Particle.likelihood_field = likelihood_field


    def copy(self):
        """Return a deep copy of this particle.  This needs to be used when
        resampling to ensure that we don't end up with aliased
        particles in the particle set.

        """
        return copy.copy(self)

    @property
    def x(self):
        """ x position in meters"""
        return self._pose.position.x

    @x.setter
    def x(self, x):
        self._pose.position.x = x

    @property
    def y(self):
        """ y position in meters"""
        return self._pose.position.y

    @y.setter
    def y(self, y):
        self._pose.position.y = y

    @property
    def theta(self):
        """ Orientation in radians. """
        quat = np.array([self._pose.orientation.x,
                         self._pose.orientation.y,
                         self._pose.orientation.z,
                         self._pose.orientation.w])
        euler = tf.transformations.euler_from_quaternion(quat)
        return euler[2]

    @theta.setter
    def theta(self, theta):
        quat = tf.transformations.quaternion_from_euler(0, 0, theta)
        self._pose.orientation.x = quat[0]
        self._pose.orientation.y = quat[1]
        self._pose.orientation.z = quat[2]
        self._pose.orientation.w = quat[3]

    @property
    def pose(self):
        """ Pose of this particle as a geometry_msgs/Pose object """
        return self._pose

    @pose.setter
    def pose(self, pose):
        self._pose = copy.copy(pose)

        # Now normalize the quaternion.
        quat = np.array([pose.orientation.x,
                         pose.orientation.y,
                         pose.orientation.z,
                         pose.orientation.w])
        quat = tf.transformations.unit_vector(quat)
        self._pose.orientation.x = quat[0]
        self._pose.orientation.y = quat[1]
        self._pose.orientation.z = quat[2]
        self._pose.orientation.w = quat[3]


    def sense(self, scan_msg):
        """Update the weight of this particle based on a LaserScan message.
        The new weight will be relatively high if the pose of this
        particle corresponds will with the scan, it will be relatively
        low if the pose does not correspond to this scan.

        The algorithm used here is loosely based on the Algorithm in
        Table 6.3 of Probabilistic Robotics Thrun et. al. 2005

        Arguments:
           scan_msg - sensor_msgs/LaserScan object

        Returns:
           None

        """

        xs, ys = self._scan_to_endpoints(scan_msg)
        total_prob = 0
        for i in range(0, len(xs), 10):
            likelihood = self.likelihood_field.get_cell(xs[i], ys[i])
            if np.isnan(likelihood):
                likelihood = 0
            total_prob += np.log(self.laser_z_hit * likelihood +
                                 self.laser_z_rand)

        self.weight *= np.exp(total_prob)


    def move(self, odom1, odom2):
        """Cause this particle to experience the same translation and
        rotation that relates odom1 to odom2, then add noise to the
        resulting position and orientation.  The particle weight will
        not be modified.

        Arguments: 
           odom1, odom2 - nav_msgs/Odometry objects

        Returns: 
           None

        """

        delta_t = odom2.header.stamp.to_sec() - odom1.header.stamp.to_sec()
        twist = odoms_to_twist(odom1, odom2, delta_t)
        particle_f = posemath.fromMsg(self.pose)
        # Rotate the twist to align it with the current particle.
        twist = particle_f.M * twist

        # randomize the twist a bit to model error in the motion model
        twist.vel[0] += np.random.randn() * .1
        twist.vel[1] += np.random.randn() * .1
        twist.rot[2] += np.random.randn() * .05

        new_particle_f = posemath.addDelta(particle_f, twist, delta_t)

        self.pose = posemath.toMsg(new_particle_f)



    def _scan_to_endpoints(self, scan_msg):
        """Helper method used to convert convert range values into x, y
        coordinates in the map coordinate frame.  Based on
        probabilistic robotics equation 6.32

        """
        theta_beam = np.arange(scan_msg.angle_min, scan_msg.angle_max,
                               scan_msg.angle_increment)
        ranges = np.array(scan_msg.ranges)
        xs = (self.x + ranges * np.cos(self.theta + theta_beam))
        ys = (self.y + ranges * np.sin(self.theta + theta_beam))

        # Clear out nan entries:
        xs = xs[np.logical_not(np.isnan(xs))]
        ys = ys[np.logical_not(np.isnan(ys))]

        # Clear out inf entries:
        xs = xs[np.logical_not(np.isinf(xs))]
        ys = ys[np.logical_not(np.isinf(ys))]
        return xs, ys

    def scan_markers(self, scan_msg, color=(0, 1.0, 0)):
        """Returns a MarkerArray message displaying what the scan message
        would look like from the perspective of this particle.  Just
        for debugging.

        Returns:
           visualization_msgs/MarkerArray

        """

        xs, ys = self._scan_to_endpoints(scan_msg)
        marker_array = MarkerArray()
        header = Marker().header
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'
        for i in range(len(xs)):
            marker = Marker()
            marker.header = header
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = xs[i]
            marker.pose.position.y = ys[i]
            marker.pose.position.z = .3
            marker.pose.orientation.w = 1.0
            marker.id = np.array([(id(self) * 13 *  + i*17) % 2**32],
                                 dtype='int32')[0]
            marker.scale.x = .02
            marker.scale.y = .02
            marker.scale.z = .02

            marker.color.a = 1.0
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]

            marker.lifetime = rospy.Duration(5.0)
            marker_array.markers.append(marker)

        return marker_array


# Utility methods below this point.--------------------

def odoms_to_twist(odom1, odom2, delta_t):
    """Create a twist that would move odom1 to odom2 in time delta_t.

    Returns:
      geometry_msgs/Twist

    """
    f_0 = posemath.Frame()
    f_w_1 = posemath.fromMsg(odom1.pose.pose)
    f_w_2 = posemath.fromMsg(odom2.pose.pose)
    f_1_2 = f_w_1.Inverse() * f_w_2
    return posemath.diff(f_0, f_1_2, delta_t)

