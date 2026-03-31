import cv2
import time
import numpy as np
from typing import Tuple

from scipy.spatial.transform import Rotation as R

from lib.a_star_3D import AStar3D
# from lib.mapping_and_planning_examples import trajectory_tracking

# The available ground truth state measurements can be accessed by calling sensor_data[item]. All values of "item" are provided as defined in main.py within the function read_sensors.
# The "item" values that you may later retrieve for the hardware project are:
# "x_global": Global X position
# "y_global": Global Y position
# "z_global": Global Z position
# 'v_x": Global X velocity
# "v_y": Global Y velocity
# "v_z": Global Z velocity
# "ax_global": Global X acceleration
# "ay_global": Global Y acceleration
# "az_global": Global Z acceleration (With gravtiational acceleration subtracted)
# "roll": Roll angle (rad)
# "pitch": Pitch angle (rad)
# "yaw": Yaw angle (rad)
# "q_x": X Quaternion value
# "q_y": Y Quaternion value
# "q_z": Z Quaternion value
# "q_w": W Quaternion value

# A link to further information on how to access the sensor data on the Crazyflie hardware for the hardware practical can be found here: https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/api/logs/#stateestimate


class MyAssignment:
    SEGMENT_LOCATION_GATES = [2, 4, 6, 8, 10]
    INIT_POS = [1, 4, 0.5, np.deg2rad(45)]
    GOAL_TOL = 0.25
    
    MP_GRID_SIZE = 0.25
    MP_BOUNDS = (0, 8, 0, 8, 0, 3)  # (x_min, x_max, y_min, y_max, z_min, z_max)

    def __init__(self):
        # ---- INITIALISE YOUR VARIABLES HERE ----
        self.R_yaw90 = R.from_euler('xyz', [0, 0, np.pi / 2]).as_matrix()

        self.mode = 0 # 0: searching gates, 1: go to segment 0, 2: fast mode, 3: finish
        self.trajectory_mode = 0 
        # 0: waiting for waypoints
        # 1: computing trajectory from start and goal
        # 2: compute trajectory from waypoints
        # 3: running trajectory
        
        self.pos_gates = []
        self.trajectory = []

        self.obstacles = []
        self.start = (0.0, 0.0, 0.5)
        self.goal = (5, 1, 1)
        self.mp = MotionPlanner3D()

        self.tracker = Tracker()

    def run_motion_planing (self, start, goal, obstacles):
        self.mp.run_motion_planner(start, obstacles, self.MP_BOUNDS, self.MP_GRID_SIZE, goal)

    def run_planner (self, trajectories, erase_obstacles=False):
        if erase_obstacles: self.mp.obstacles = []
        
        # Compute trajectory
        self.mp.init_params(trajectories)
        self.mp.run_planner(trajectories)
        self.trajectory_setpoints, self.time_setpoints = self.mp.trajectory_setpoints, self.mp.time_setpoints

    def compute_command(self, sensor_data, camera_data, dt):

        # NOTE: Displaying the camera image with cv2.imshow() will throw an error because GUI operations should be performed in the main thread.
        # If you want to display the camera image you can call it in main.py.

        # Take off example
        if sensor_data['z_global'] < 0.49:
            control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]
            return control_command

        # ---- YOUR CODE HERE ----
        
        control_command = self.get_next_waypoint (sensor_data, dt)
        if self.trajectory_mode != 0: # TODO: Trajectory 
            return control_command

        if self.mode == 0: # TODO: Searching
            control_command = self.search_gates (sensor_data, camera_data)
        elif self.mode == 1: # TODO: Go to segment 0
            control_command = self.INIT_POS
        elif self.mode == 2: # TODO: Fast mode
            control_command = self.fast_mode (sensor_data, camera_data)
        else: # TODO: Finish
            control_command = self.dancing (sensor_data)

        pos = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']])
        actual_segment = self._get_actual_segment(pos)
        self.update_mode(actual_segment, pos)

        control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, np.deg2rad(45)]
        return control_command # Ordered as array with: [pos_x_cmd, pos_y_cmd, pos_z_cmd, yaw_cmd] in meters and radians
    
    def search_gates (self, sensor_data, camera_data):
        # Method 1
        # Detect gates
        self._detect_gates (camera_data)

        # If you dont see: turn
        # If you see one: go to the gate to take a better position
        # If you see one in a good distance: save it in obstacles and in gate positions and go for other one

        # Method 2
        # Go to segment 1: detect gate in segment 2
        # Go to segment 3: detect gate in segment 4
        # Go to segment 5: detect gate in segment 6
        # Go to segment 7: detect gate in segment 8
        # Go to segment 9: detect gate in segment 10



    def fast_mode (self, sensor_data, camera_data):
        # Method 1: Start far form start and gete velocity
        pass

    def get_next_waypoint (self, sensor_data, dt) -> Tuple[bool, np.ndarray]:
        # Follow the trajectory until finish
        control_command = [
            sensor_data['x_global'], sensor_data['y_global'], 
            sensor_data['z_global'], sensor_data['yaw']
        ]

        if self.trajectory_mode == 0:
            return False, control_command
        elif self.trajectory_mode == 1:
            # Compute trajectory
            self.run_motion_planing(self.start, self.goal, self.obstacles)

            # Change mode
            self.trajectory_mode = 3
            return False, control_command
        elif self.trajectory_mode == 2:
            # Compute trajectory form waypoints
            self.run_planner(self.trajectory, True)

            # Change mode
            self.trajectory_mode = 3
            return False, control_command
        elif self.trajectory_mode == 3:
            control_command, done = self.tracker.trajectory_tracking (
                sensor_data, dt, self.time_setpoints, self.trajectory_setpoints, self.GOAL_TOL)
            
            if done: self.trajectory_mode = 0
            return True, control_command

    def dancing (self, sensor_data): # TODO
        pass

    def update_mode (self, actual_segment, pos):
        if self.mode == 0:
            # Verify if we have all the gates
            if len(self.pos_gates) == len(self.SEGMENT_LOCATION_GATES):
                location_gates = []

                for p_gate in self.pos_gates:
                    segment = self._get_actual_segment(p_gate)
                    location_gates.append(segment)

                if sorted(location_gates) == sorted(self.SEGMENT_LOCATION_GATES):
                    # We have all the gates, we can run
                    self.mode += 1
        elif self.mode == 1:
            # Check if we are in segment 0
            if np.linalg.norm(pos - self.INIT_POS) < 1e-2:
                self.mode += 1
        elif self.mode == 2:
            # Check if we pass through all the gates and went to zero
            # TODO: check each gate

            if actual_segment == 0:
                self.mode += 1
        else:
            # Finish or something else
            pass

    def _get_actual_segment (self, global_pos):
        # Fixed coordinates
        centered = (global_pos - np.array([4, 4, 0])) * np.array([1, 1, 1])
        rotated = self.R_yaw90 @ (centered)

        # Get theta
        theta = np.arctan2 (rotated[1], rotated[0])
        dtheta = np.rad2deg (theta)
        
        # TODO: make beauty
        if dtheta < -165 or dtheta > 165:   segment = 9
        elif -165 <= dtheta < -135:         segment = 10
        elif -135 <= dtheta < -105:         segment = 11
        elif -105 <= dtheta <  -75:         segment = 0
        elif  -75 <= dtheta <  -45:         segment = 1
        elif  -45 <= dtheta <  -15:         segment = 2
        elif  -15 <= dtheta <   15:         segment = 3
        elif   15 <= dtheta <   45:         segment = 4
        elif   45 <= dtheta <   75:         segment = 5
        elif   75 <= dtheta <  105:         segment = 6
        elif  105 <= dtheta <  135:         segment = 7
        elif  135 <= dtheta <  165:         segment = 8

        return segment

    def _detect_gates (self, camera_data):
        pass

class GatesDetector:

    CAM_FIELD_OF_VIEW = 1.5
    CAM_WIDTH = 300
    CAM_HEIGHT = 300

    CAM_FOCAL_DIST = CAM_WIDTH / (1 * np.tan(CAM_FIELD_OF_VIEW / 2))

    GATE_HEIGHT = 0.4 # m

    def __init__ (self):
        pass

    def pink_mask (self, camera_data):
        # https://stackoverflow.com/questions/70071741/opencv-python-how-recognize-pink-wood-in-the-image
        img = camera_data.copy()
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # PINK limits
        # H: 140–170, S: 50–255, V: 80–255
        COLOR_MIN = np.array([140, 50, 80], np.uint8) 
        COLOR_MAX = np.array([170, 255, 255], np.uint8)

        frame_threshed = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)     # Thresholding image
        ret, thresh = cv2.threshold(frame_threshed, 127, 255, 0)

        img_points, gates = self.find_boxes (frame_threshed, img)

        return img_points

    def find_boxes (self, mask, img):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        gates = []
        
        for cnt in contours:
            if cv2.contourArea(cnt) < 50:
                continue

            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx  = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) != 4:
                continue

            corners = approx.reshape(4, 2)
            s    = corners.sum(axis=1)
            diff = np.diff(corners, axis=1).flatten()

            ordered = np.array([
                corners[np.argmin(s)],      # top left
                corners[np.argmin(diff)],   # top right
                corners[np.argmax(s)],      # bottom right
                corners[np.argmax(diff)],   # bottom left
            ])

            gates.append(ordered)

            # Draw each corner as a dot  
            for pt in ordered:
                cv2.circle(img, tuple(pt), 4, (255, 0, 0), -1)

        return img, gates
    
    def compute_gates_position (self, gates_corners):
        positions = []
        for gate in gates_corners:
            left_dist_px = np.linalg.norm (gate[0] - gate[3])
            right_dist_px = np.linalg.norm (gate[1] - gate[2])

            left_dist = self.get_distance(left_dist_px, self.GATE_HEIGHT)
            right_dist = self.get_distance(right_dist_px, self.GATE_HEIGHT)

            # TODO: compute gate positions
            # mean point, of the points

    def get_distance (self, dist_px, size_object):
        return size_object * self.CAM_FOCAL_DIST / dist_px # meters
    
class Tracker:
    def __init__ (self):
        self.index_current_setpoint = None
        self.timer = None
        self.timer_done = None

    def trajectory_tracking(self, sensor_data, dt, timepoints, setpoints, tol, repeat = False):

        start_point = setpoints[0]
        end_point = setpoints[-1]

        if self.timer is None:
            # Begin timer and start trajectory
            self.timer = 0
            print("Trajectory tracking started")
            self.index_current_setpoint = 1
        else:
            self.timer += dt

        # Determine the current setpoint based on the time
        if self.timer is not None:
            if self.index_current_setpoint < len(timepoints) - 1:
                # Update new setpoint
                if self.timer >= timepoints[self.index_current_setpoint]:
                    self.index_current_setpoint += 1
                current_setpoint = setpoints[self.index_current_setpoint,:]
            else:
                # Hover at the final setpoint
                current_setpoint = end_point
                distance_to_goal = np.linalg.norm([
                    sensor_data['x_global'] - end_point[0], 
                    sensor_data['y_global'] - end_point[1], 
                    sensor_data['z_global'] - end_point[2]
                ])

                if self.timer_done is None and distance_to_goal < tol:
                    self.timer_done = True
                    print("Trajectory took " + str(np.round(self.timer, 1)) + " [s]")
                    if repeat:
                        self.timer_done = None
                        self.timer = None
                    
                    return current_setpoint, True
        return current_setpoint, False

class MotionPlanner3D():
    
    def __init__(self):
        self.trajectory_setpoints = None
        self.obstacles = []

    def run_motion_planner (self, start, goal, grid_size, obstacles, bounds):
        # Inputs:
        # - start: The sequence of input path waypoints provided by the path-planner, including the start and final goal position: Vector of m waypoints, consisting of a tuple with three reference positions each as provided by AStar 
        # - obstacles: 2D array with obstacle locations and obstacle widths [x, y, z, dx, dy, dz]*n_obs
        # - bounds: The bounds of the environment [x_min, x_max, y_min, y_max, z_min, z_max]
        # - grid_size: The grid size of the environment (scalar)
        # - goal: The final goal position of the drone (tuple of 3) 

        self.ast = AStar3D(start, goal, grid_size, obstacles, bounds)
        self.path = self.ast.find_path()

        self.trajectory_setpoints = None
        self.obstacles = obstacles

        self.run_planner(self.path)

    def run_planner(self, path_waypoints):    
        # Run the subsequent functions to compute the polynomial coefficients and extract and visualize the trajectory setpoints

        self.init_params(path_waypoints)
        poly_coeffs = self.compute_poly_coefficients(path_waypoints)
        self.trajectory_setpoints, self.time_setpoints = self.poly_setpoint_extraction(poly_coeffs, self.obstacles, path_waypoints)

        ## ---------------------------------------------------------------------------------------------------- ##

    def init_params(self, path_waypoints):

        # Inputs:
        # - path_waypoints: The sequence of input path waypoints provided by the path-planner, including the start and final goal position: Vector of m waypoints, consisting of a tuple with three reference positions each as provided by AStar

        # TUNE THE FOLLOWING PARAMETERS (PART 2) ----------------------------------------------------------------- ##
        self.disc_steps = 20 #Integer number steps to divide every path segment into to provide the reference positions for PID control # IDEAL: Between 10 and 20
        self.vel_lim = 7.0 #Velocity limit of the drone (m/s)
        self.acc_lim = 50.0 #Acceleration limit of the drone (m/s²)
        t_f = 3.5  # Final time at the end of the path (s)

        # Determine the number of segments of the path
        self.times = np.linspace(0, t_f, len(path_waypoints)) # The time vector at each path waypoint to traverse (Vector of size m) (must be 0 at start)

    def compute_poly_matrix(self, t):
        # Inputs:
        # - t: The time of evaluation of the A matrix (t=0 at the start of a path segment, else t >= 0) [Scalar]
        # Outputs: 
        # - The constraint matrix "A_m(t)" [5 x 6]
        # The "A_m" matrix is used to represent the system of equations [x, \dot{x}, \ddot{x}, \dddot{x}, \ddddot{x}]^T  = A_m(t) * poly_coeffs (where poly_coeffs = [c_0, c_1, c_2, c_3, c_4, c_5]^T and represents the unknown polynomial coefficients for one segment)
        A_m = np.zeros((5,6))
        
        # TASK: Fill in the constraint factor matrix values where each row corresponds to the positions, velocities, accelerations, snap and jerk here
        # SOLUTION ---------------------------------------------------------------------------------- ## 
        
        A_m = np.array([
            [t**5, t**4, t**3, t**2, t, 1], #pos
            [5*(t**4), 4*(t**3), 3*(t**2), 2*t, 1, 0], #vel
            [20*(t**3), 12*(t**2), 6*t, 2, 0, 0], #acc  
            [60*(t**2), 24*t, 6, 0, 0, 0], #jerk
            [120*t, 24, 0, 0, 0, 0] #snap
        ])

        return A_m

    def compute_poly_coefficients(self, path_waypoints):
        
        # Computes a minimum jerk trajectory given time and position waypoints.
        # Inputs:
        # - path_waypoints: The sequence of input path waypoints provided by the path-planner, including the start and final goal position: Vector of m waypoints, consisting of a tuple with three reference positions each as provided by AStar
        # Outputs:
        # - poly_coeffs: The polynomial coefficients for each segment of the path [6(m-1) x 3]

        # Use the following variables and the class function self.compute_poly_matrix(t) to solve for the polynomial coefficients
        
        seg_times = np.diff(self.times) #The time taken to complete each path segment
        m = len(path_waypoints) #Number of path waypoints (including start and end)
        poly_coeffs = np.zeros((6*(m-1),3))

        # YOUR SOLUTION HERE ---------------------------------------------------------------------------------- ## 

        # 1. Fill the entries of the constraint matrix A and equality vector b for x,y and z dimensions in the system A * poly_coeffs = b. Consider the constraints according to the lecture: We should have a total of 6*(m-1) constraints for each dimension.
        # 2. Solve for poly_coeffs given the defined system

        for dim in range(3):  # Compute for x, y, and z separately
            A = np.zeros((6*(m-1), 6*(m-1)))
            b = np.zeros(6*(m-1))
            pos = np.array([p[dim] for p in path_waypoints])
            A_0 = self.compute_poly_matrix(0) # A_0 gives the constraint factor matrix A_m for any segment at t=0, this is valid for the starting conditions at every path segment

            # SOLUTION
            row = 0
            for i in range(m-1):
                pos_0 = pos[i] #Starting position of the segment
                pos_f = pos[i+1] #Final position of the segment
                # The prescribed zero velocity (v) and acceleration (a) values at the start and goal position of the entire path
                v_0, a_0 = 0, 0
                v_f, a_f = 0, 0
                A_f = self.compute_poly_matrix(seg_times[i]) # A_f gives the constraint factor matrix A_m for a segment i at its relative end time t=seg_times[i]
                if i == 0: # First path segment
                #     # 1. Implement the initial constraints here for the first segment using A_0
                #     # 2. Implement the final position and the continuity constraints for velocity, acceleration, snap and jerk at the end of the first segment here using A_0 and A_f (check hints in the exercise description)
                    A[row, i*6:(i+1)*6] = A_0[0] #Initial position constraint
                    b[row] = pos_0
                    row += 1
                    A[row, i*6:(i+1)*6] = A_f[0] #Final position constraint
                    b[row] = pos_f
                    row += 1
                    A[row, i*6:(i+1)*6] = A_0[1] #Initial velocity constraint
                    b[row] = v_0
                    row += 1
                    A[row, i*6:(i+1)*6] = A_0[2] #Initial acceleration constraint
                    b[row] = a_0
                    row += 1
                    #Continuity of velocity, acceleration, jerk, snap
                    A[row:row+4, i*6:(i+1)*6] = A_f[1:]
                    A[row:row+4, (i+1)*6:(i+2)*6] = -A_0[1:]
                    b[row:row+4] = np.zeros(4)
                    row += 4
                elif i < m-2: # Intermediate path segments
                #     # 1. Similarly, implement the initial and final position constraints here for each intermediate path segment
                #     # 2. Similarly, implement the end of the continuity constraints for velocity, acceleration, snap and jerk at the end of each intermediate segment here using A_0 and A_f
                    A[row, i*6:(i+1)*6] = A_0[0] #Initial position constraint
                    b[row] = pos_0
                    row += 1
                    A[row, i*6:(i+1)*6] = A_f[0] #Final position constraint
                    b[row] = pos_f
                    row += 1
                    #Continuity of velocity, acceleration, jerk and snap
                    A[row:row+4, i*6:(i+1)*6] = A_f[1:]
                    A[row:row+4, (i+1)*6:(i+2)*6] = -A_0[1:]
                    b[row:row+4] = np.zeros(4)
                    row += 4
                elif i == m-2: #Final path segment
                #     # 1. Implement the initial and final position, velocity and accelerations constraints here for the final path segment using A_0 and A_f
                    A[row, i*6:(i+1)*6] = A_0[0] #Initial position constraint
                    b[row] = pos_0
                    row += 1
                    A[row, i*6:(i+1)*6] = A_f[0] #Final position constraint
                    b[row] = pos_f
                    row += 1
                    A[row, i*6:(i+1)*6] = A_f[1] #Final velocity constraint
                    b[row] = v_f
                    row += 1
                    A[row, i*6:(i+1)*6] = A_f[2] #Final acceleration constraint
                    b[row] = a_f
                    row += 1
            # Solve for the polynomial coefficients for the dimension dim

            poly_coeffs[:,dim] = np.linalg.solve(A, b)   

        return poly_coeffs

    def poly_setpoint_extraction(self, poly_coeffs, obs, path_waypoints):

        # DO NOT MODIFY --------------------------------------------------------------------------------------- ##

        # Uses the class features: self.disc_steps, self.times, self.poly_coeffs, self.vel_lim, self.acc_lim
        x_vals, y_vals, z_vals = np.zeros((self.disc_steps*len(self.times),1)), np.zeros((self.disc_steps*len(self.times),1)), np.zeros((self.disc_steps*len(self.times),1))
        v_x_vals, v_y_vals, v_z_vals = np.zeros((self.disc_steps*len(self.times),1)), np.zeros((self.disc_steps*len(self.times),1)), np.zeros((self.disc_steps*len(self.times),1))
        a_x_vals, a_y_vals, a_z_vals = np.zeros((self.disc_steps*len(self.times),1)), np.zeros((self.disc_steps*len(self.times),1)), np.zeros((self.disc_steps*len(self.times),1))

        # Define the time reference in self.disc_steps number of segements
        time_setpoints = np.linspace(self.times[0], self.times[-1], self.disc_steps*len(self.times))  # Fine time intervals

        # Extract the x,y and z direction polynomial coefficient vectors
        coeff_x = poly_coeffs[:,0]
        coeff_y = poly_coeffs[:,1]
        coeff_z = poly_coeffs[:,2]

        for i,t in enumerate(time_setpoints):
            seg_idx = min(max(np.searchsorted(self.times, t)-1,0), len(coeff_x) - 1)
            # Determine the x,y and z position reference points at every refernce time
            x_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[0],coeff_x[seg_idx*6:(seg_idx+1)*6])
            y_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[0],coeff_y[seg_idx*6:(seg_idx+1)*6])
            z_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[0],coeff_z[seg_idx*6:(seg_idx+1)*6])
            # Determine the x,y and z velocities at every reference time
            v_x_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[1],coeff_x[seg_idx*6:(seg_idx+1)*6])
            v_y_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[1],coeff_y[seg_idx*6:(seg_idx+1)*6])
            v_z_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[1],coeff_z[seg_idx*6:(seg_idx+1)*6])
            # Determine the x,y and z accelerations at every reference time
            a_x_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[2],coeff_x[seg_idx*6:(seg_idx+1)*6])
            a_y_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[2],coeff_y[seg_idx*6:(seg_idx+1)*6])
            a_z_vals[i,:] = np.dot(self.compute_poly_matrix(t-self.times[seg_idx])[2],coeff_z[seg_idx*6:(seg_idx+1)*6])

        yaw_vals = np.zeros((self.disc_steps*len(self.times),1))
        trajectory_setpoints = np.hstack((x_vals, y_vals, z_vals, yaw_vals))
            
        # Find the maximum absolute velocity during the segment
        vel_max = np.max(np.sqrt(v_x_vals**2 + v_y_vals**2 + v_z_vals**2))
        vel_mean = np.mean(np.sqrt(v_x_vals**2 + v_y_vals**2 + v_z_vals**2))
        acc_max = np.max(np.sqrt(a_x_vals**2 + a_y_vals**2 + a_z_vals**2))
        acc_mean = np.mean(np.sqrt(a_x_vals**2 + a_y_vals**2 + a_z_vals**2))

        print("Maximum flight speed: " + str(vel_max))
        print("Average flight speed: " + str(vel_mean))
        print("Average flight acceleration: " + str(acc_mean))
        print("Maximum flight acceleration: " + str(acc_max))
        
        # Check that it is less than an upper limit velocity v_lim
        assert vel_max <= self.vel_lim, "The drone velocity exceeds the limit velocity : " + str(vel_max) + " m/s"
        assert acc_max <= self.acc_lim, "The drone acceleration exceeds the limit acceleration : " + str(acc_max) + " m/s²"

        # ---------------------------------------------------------------------------------------------------- ##

        return trajectory_setpoints, time_setpoints
    





# Module-level singleton so main.py can call assignment.get_command() unchanged
_controller = MyAssignment()
_detector = GatesDetector()

def get_command(sensor_data, camera_data, dt):
    return _controller.compute_command(sensor_data, camera_data, dt)

def show_mask (camera_data):
    return _detector.pink_mask(camera_data)

if __name__ == "__main__":
    print(_controller._get_actual_segment(np.array([4.1, 4, 0])))
