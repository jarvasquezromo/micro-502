import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Union, Dict, Optional

from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from lib.a_star_3D import AStar3D

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
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

class MyAssignment:
    SEGMENT_LOCATION_GATES = [2, 4, 6, 8, 10]
    INIT_POS = [1, 4, 1.35, np.deg2rad(45)]
    GOAL_TOL = 0.25
    
    MP_GRID_SIZE = 0.25
    MP_BOUNDS = (0, 8, 0, 8, 0, 3)  # (x_min, x_max, y_min, y_max, z_min, z_max)

    def __init__(self):
        # ---- INITIALISE YOUR VARIABLES HERE ----
        self.R_yaw90 = R.from_euler('xyz', [0, 0, np.pi / 2]).as_matrix()

        self.mode = Keeper() # 0: searching gates, 1: go to segment 0, 2: fast mode, 3: finish
        self.mode_trajectory = Keeper() 
        # 0: waiting for waypoints
        # 1: computing trajectory from start and goal
        # 2: compute trajectory from waypoints
        # 3: running trajectory
        self.mode_searching = Keeper()
        # 0: go to position of searching
        # 1: turn around to look for the gate
        # 2: go near the gate
        # 3: confirm position of gate
        # 4: go from actual position, through the gate to the next position
        self.mode_fast = Keeper()
        
        self.pos_gates = []
        self.idx_gate_search = 0
        self.trajectory = []
        self.last_command = self.INIT_POS
        self.target_control_command = None

        self.obstacles = []
        self.start = (0.0, 0.0, 0.5)
        self.goal = (5, 1, 1)
        self.mp = MotionPlanner3D()

        self.tracker = Tracker()
        self.detector = GatesDetectorTriangulation()

        self.drone = None

        # Gap-fill state (used by _detect_and_estimate_gates)
        self._last_estimates = []
        self._gap_timer = 0.0

        # ── Triangulation / observation buffer ───────────────────────────
        # Accumulated (corners, R, T) tuples while the drone sweeps step 2.
        self._obs_buffer: List[tuple] = []
        self._obs_buffer_refined: List[tuple] = []
        # Minimum baseline between consecutive stored observations (metres).
        self._TRI_MIN_BASELINE: float = 0.10
        self._reset_search_state()
 
        # Gap-fill state
        self._last_estimates: List = []
        self._gap_timer: float = 0.0


    def run_motion_planing (self, start, goal, obstacles):
        self.mp.run_motion_planner(start, obstacles, self.MP_BOUNDS, self.MP_GRID_SIZE, goal)
        self.tracker.reset()


    def run_planner (self, trajectories, erase_obstacles=False, **kwargs):
        if erase_obstacles: self.mp.obstacles = []
        
        # Compute trajectory
        self.mp.run_planner(trajectories, **kwargs)
        # self.trajectory_setpoints, self.time_setpoints, self.critic_setpoints = \
        #     self.mp.trajectory_setpoints, self.mp.time_setpoints, self.mp.critic_setpoint

        self.tracker.reset()
        self.tracker.load(
            self.mp.time_setpoints, self.mp.trajectory_setpoints,
            self.mp.velocity_setpoints, self.mp.acceleration_setpoints
        )


    def compute_command(self, sensor_data, camera_data, dt):

        # NOTE: Displaying the camera image with cv2.imshow() will throw an error because GUI operations should be performed in the main thread.
        # If you want to display the camera image you can call it in main.py.

        # Take off example
        if sensor_data['z_global'] < 0.49:
            control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]
        else:

        # ---- YOUR CODE HERE ----
            if self.mode_trajectory != 0: # Trajectory (priority)
                control_command = self.get_next_waypoint (sensor_data, dt)
            elif self.mode == 0: # TODO: Searching
                control_command = self.search_gates (sensor_data, camera_data, dt)
            elif self.mode == 1: # Go to segment 0
                control_command = self.INIT_POS # TODO: check the last gate is pathing throug correctly
            elif self.mode == 2: # TODO: Fast mode
                control_command = self.fast_mode (sensor_data, camera_data)
            else: # TODO: Finish
                control_command = self.dancing (sensor_data)

            # Updateing mode
            # print (control_command)
            if len(control_command) != 4 and len(control_command) != 10:
                logger.warning("Your control its bad: %s", str(control_command))
            pos = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']])
            actual_segment = self._get_actual_segment(pos[:3])
            self.update_mode(actual_segment, pos)

        # DEBUG
        # control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, np.deg2rad(45)]
        self.last_command = control_command
        return control_command # Ordered as array with: [pos_x_cmd, pos_y_cmd, pos_z_cmd, yaw_cmd] in meters and radians
    
    # ------------------------------------------------------------------ #
    #  Detection helpers                                                   #
    # ------------------------------------------------------------------ #
 
    def _detect_gates_raw(self, camera_data, drone_rot, drone_pos):
        """Run pink-mask detection and return (corners_list, estimates_list).
        Always returns fresh detections — no gap-fill here."""
        _, corners_list = self.detector._detect_pink_gates(camera_data)
        estimates = self.detector.mono_gate_positions(corners_list, drone_rot, drone_pos) \
                    if corners_list else []
        return corners_list, estimates
 
    def _accumulate_observation(self, corners, drone_rot, drone_pos, obs_buffer):
        """Store one camera observation if the drone has moved enough since the last one."""
        if not obs_buffer:
            obs_buffer.append((corners, drone_rot, drone_pos.copy()))
            return
        last_T = obs_buffer[-1][2]
        if np.linalg.norm(drone_pos - last_T) >= self._TRI_MIN_BASELINE:
            obs_buffer.append((corners, drone_rot, drone_pos.copy()))
 
    def _compute_best_gate_estimate(self, drone_pos, obs_buffer):
        """
        Triangulate across ALL pairs in _obs_buffer and return the median
        centre estimate as a (4,) command [x,y,z,yaw].  Falls back to the
        last mono estimate stored in _mono_gate_cmd if the buffer is too small.
        """ 
        if len(obs_buffer) < 2:
            logger.warning("Not enough observations for triangulation, using mono fallback")
            return self._mono_gate_cmd  # may be None
 
        centres = []
        yaws    = []
        for i in range(len(obs_buffer) - 1):
            c1, r1, t1 = obs_buffer[i]
            c2, r2, t2 = obs_buffer[i + 1]
            obs1 = CameraObservation(gate_corners=c1, R_drone_to_world=r1, T_drone_to_world=t1)
            obs2 = CameraObservation(gate_corners=c2, R_drone_to_world=r2, T_drone_to_world=t2)
            results = self.detector._triangulate_pair(obs1, obs2)
            for est in results:
                if est.residual is not None and est.residual < 0.35:
                    centres.append(est.center)
                    yaws.append(est.yaw)
 
        if not centres:
            logger.warning("All triangulation pairs had large residuals, using mono fallback")
            return self._mono_gate_cmd
 
        # Median over all valid triangulation results for robustness
        centre = np.median(centres, axis=0)
        yaw    = float(np.median(yaws))
        logger.info(
            "Triangulated gate from %d pairs → centre=%s  yaw=%.2f rad",
            len(centres), np.round(centre, 3), yaw
        )
        return np.concatenate([centre, [yaw]])
 
    def _gate_estimate_to_command(self, estimate) -> dict:
        """Adapt a GateEstimate to the dict format _get_idx_gate_search expects."""
        return {"command": estimate.command, "valid": estimate.valid}
 
    def _reset_search_state(self):
        """Clear per-gate transient state before starting the next gate search."""
        self._obs_buffer.clear()
        self._obs_buffer_refined.clear()
        self._sweep_waypoints = []
        self._sweep_idx       = 0
        self._mono_gate_cmd   = None
        self.timer_searching  = 0.0
        self.num_turns        = 0
        self._sweep_direction = 1
        self._sweep_pass      = 0

 
    # ------------------------------------------------------------------ #
    #  Main search state machine                                           #
    # ------------------------------------------------------------------ #
 
    def search_gates(self, sensor_data, camera_data, dt):
        """
        Trajectory-based gate search state machine.
 
        Step 0  – Fly to pos_1 = _get_point_segment(gate_seg - 1, 3).
                  Kick off a trajectory and wait for it to finish.
 
        Step 1  – Launch a trajectory from pos_1 → pos_2 =
                  _get_point_segment(gate_seg - 1, 0.5), collecting corners
                  into _obs_buffer while the tracker runs.
                  Detection happens every tick while mode_trajectory == 3.
 
        Step 1½ – If after the pass we have fewer than MIN_CORNER_INSTANCES
                  corner observations, reverse the trajectory (pos_2 → pos_1 or
                  pos_1 → pos_2 depending on direction) and repeat until we
                  collect enough data.
 
        Step 2  – Launch a refined confirmation trajectory:
                    actual_pos → pos_1 → point 0.6 m in front of the gate.
                  Collect corners into _obs_buffer_refined along the way.
 
        Step 3  – Compute the best gate estimate from _obs_buffer_refined
                  (falls back to _obs_buffer if refined is sparse).
                  Save gate position.
 
        Step 4  – Pass through the gate: trajectory from actual_pos through
                  gate centre to 0.5 m beyond.  Reset and move to next gate.
        """
        # ── Minimum number of corner detections before we trust the sweep ──
        MIN_CORNER_INSTANCES = 4
 
        actual_pos = np.array([
            sensor_data['x_global'], sensor_data['y_global'],
            sensor_data['z_global'], sensor_data['yaw'],
        ])
        drone_rot = R.from_euler(
            'xyz',
            [sensor_data['roll'], sensor_data['pitch'], sensor_data['yaw']]
        )
        control_command = actual_pos.copy()
 
        # ── Pre-compute the two sweep endpoints for this gate ────────────
        gate_seg = self.SEGMENT_LOCATION_GATES[self.idx_gate_search]
        pos_far = self._get_point_segment(gate_seg - 1, 3)   # far boundary
        pos_near = self._get_point_segment(gate_seg - 1, 0.5) # near boundary
 
        # ── Shared detection — runs every tick in steps 1, 1½, and 2 ────
        corners_list, estimates = self._detect_gates_raw(
            camera_data, drone_rot, actual_pos[:3]
        )
        gates_positions = [self._gate_estimate_to_command(e) for e in estimates]
        idx_gate = self._get_idx_gate_search(gates_positions, self.idx_gate_search)
 
        # ================================================================
        if self.mode_searching == 0:
            # ── Step 0: fly to pos_1 ─────────────────────
            control_command = pos_far.copy()
 
            # Transition once the tracker signals completion
            if self._achieve_goal(actual_pos, pos_far):
                logger.debug("Step 0 done — arrived at pos_far, starting sweep")
                self._reset_search_state()
                # Launch step-1 trajectory immediately
                trajectory = [actual_pos[:3], np.mean((actual_pos[:3], pos_near[:3]), axis=0), pos_near[:3]]
                self.run_planner(trajectory, t_final=5)
                self.mode_searching.set(1)
 
        # ================================================================
        elif self.mode_searching == 1:
            # ── Step 1: sweep pos_1 → pos_2, accumulate corners ─────────
            # Accumulate detections while the tracker is running
            control_command, done = self.tracker.trajectory_tracking (
                sensor_data, dt, self.GOAL_TOL)
            control_command[3] = sensor_data['yaw']
            
            if not done:
                if idx_gate is not None and corners_list:
                    self._accumulate_observation(
                        corners_list[idx_gate], drone_rot, actual_pos[:3], self._obs_buffer
                    )
                    if estimates[idx_gate].valid:
                        self._mono_gate_cmd = gates_positions[idx_gate]["command"].copy()

                    if self._mono_gate_cmd is not None:
                        dir_to_gate = self._mono_gate_cmd[:2] - actual_pos[:2]
                        control_command[3] = np.arctan2(dir_to_gate[1], dir_to_gate[0])
            else: 
                # Trajectory finished → check data quality
                n_obs = len(self._obs_buffer)
                logger.debug(
                    "Step 1 pass %d done — %d corner observations collected",
                    self._sweep_pass, n_obs,
                )
                self._sweep_pass += 1

                if n_obs >= MIN_CORNER_INSTANCES:
                    # Enough data — proceed to confirmation sweep
                    logger.debug("Step 1 sufficient data, moving to Step 2")
                    gate_cmd = self._compute_best_gate_estimate(actual_pos[:3], self._obs_buffer)
                    pos_middle = self._get_point_segment(gate_seg - 1, 1.5)
                    trajectory = [
                        actual_pos[:3], 
                        pos_middle,
                        self._compute_space_to_pos(gate_cmd, -0.5)
                    ]
                    self.run_planner(trajectory, t_final=5)
                    self.mode_searching.set(2)
                else:
                    # ── Step 1½: not enough data — reverse and retry ─────
                    logger.debug(
                        "Step 1½ — only %d observations, reversing sweep (pass %d)",
                        n_obs, self._sweep_pass,
                    )
                    # Alternate direction: if we just went pos_1→pos_2 go back,
                    # and vice-versa, so the drone keeps scanning the same region.
                    if self._sweep_direction == 1:
                        trajectory = [
                            actual_pos[:3], 
                            np.mean((actual_pos[:3], pos_far[:3]), axis=0), 
                            pos_far[:3]
                        ]
                        self.run_planner(trajectory, t_final=5)
                        self._sweep_direction = -1
                    else:
                        trajectory = [
                            actual_pos[:3], 
                            np.mean((actual_pos[:3], pos_near[:3]), axis=0), 
                            pos_near[:3]
                        ]
                        self.run_planner(trajectory, t_final=5)
                        self._sweep_direction = 1
 
        # ================================================================
        elif self.mode_searching == 2:
            # ── Step 2: confirmation trajectory, accumulate refined data ─
            # actual_pos → pos_1 → 0.6 m in front of rough gate estimate
            control_command, done = self.tracker.trajectory_tracking (
                sensor_data, dt, self.GOAL_TOL)
            control_command[3] = sensor_data['yaw']
            end_trajectory = self._achieve_goal(actual_pos, self.tracker.setpoints[-1])

            looking = False
            if self._mono_gate_cmd is not None:
                dir_to_gate = self._mono_gate_cmd[:2] - actual_pos[:2]
                control_command[3] = np.arctan2(dir_to_gate[1], dir_to_gate[0])
                
                if self._achieve_goal(control_command[3], actual_pos[3], tol=0.1):
                    looking = True
            
            if idx_gate is not None and corners_list:
                self._accumulate_observation(
                    corners_list[idx_gate], drone_rot, actual_pos[:3], self._obs_buffer_refined
                )
                if estimates[idx_gate].valid:
                    self._mono_gate_cmd = gates_positions[idx_gate]["command"].copy()

            if (done or end_trajectory) and looking:
                control_command = actual_pos.copy()
                logger.debug(
                    "Step 2 done — %d refined observations collected",
                    len(self._obs_buffer_refined),
                )
                self.mode_searching.set(3)
 
        # ================================================================
        elif self.mode_searching == 3:
            # ── Step 3: compute best estimate from refined (or raw) data ─
            gate_cmd = self._compute_best_gate_estimate(
                actual_pos[:3], self._obs_buffer_refined
            )
            if gate_cmd is None:
                # Fall back to the coarser buffer from step 1
                logger.warning(
                    "Step 3 refined estimation failed, falling back to raw buffer"
                )
                gate_cmd = self._compute_best_gate_estimate(actual_pos[:3], self._obs_buffer)
 
            if gate_cmd is None:
                logger.warning("Gate estimation failed entirely, retrying from Step 1")
                self._reset_search_state()
                self.mode_searching.set(0)
            else:
                self.pos_gates.append(gate_cmd)
                self._check_gate_pos(gate_cmd)
                logger.info(
                    "Gate %d saved at %s", self.idx_gate_search, np.round(gate_cmd, 3)
                )
                self.mode_searching.set(4)
 
            control_command = actual_pos.copy()   # hover while computing
 
        # ================================================================
        elif self.mode_searching == 4:
            # ── Step 4: pass through the gate ───────────────────────────
            gate_cmd   = self.pos_gates[-1]
            beyond_pos = self._compute_space_to_pos(gate_cmd, 0.5)
 
            self.trajectory = [
                actual_pos[:3],
                gate_cmd[:3],
                beyond_pos[:3],
            ]
            self.mode_trajectory.set(2)
 
            # Advance gate index and reset for next gate
            self.idx_gate_search += 1
            self.idx_gate_search %= len(self.SEGMENT_LOCATION_GATES)
            self._reset_search_state()
            self.mode_searching.set(0)
 
            control_command = actual_pos.copy()   # trajectory takes over next tick
 
        else:
            raise ValueError(f"Unknown mode_searching value: {self.mode_searching}")
 
        return control_command

    def fast_mode (self, sensor_data, camera_data):
        # TODO: do trayectory and look for the gates, ensure passing through
        # Default values
        actual_pos = np.array([
            sensor_data['x_global'], sensor_data['y_global'], 
            sensor_data['z_global'], sensor_data['yaw']]
        )
        if self.mode_fast == 0:
            # set trajectory
            self.trajectory = [
                actual_pos[:3],
                *[pos[:3] for pos in self.pos_gates],
                self.INIT_POS[:3]
            ]

            # Set parameters for trajectory
            self.mp.final_time = 10
            self.tracker.repeat = True

            # Update modes
            self.mode_trajectory.set(2)
            self.mode_fast.set(1)

        elif self.mode_fast == 1:
            # Keep going
            pass

        return actual_pos

    def get_next_waypoint (self, sensor_data, dt) -> Tuple[bool, np.ndarray]:
        # Follow the trajectory until finish
        control_command = [
            sensor_data['x_global'], sensor_data['y_global'], 
            sensor_data['z_global'], sensor_data['yaw']
        ]
        
        # mode_trajectory: 0 waiting for waypoints
        if self.mode_trajectory == 1:
            # Compute trajectory
            self.run_motion_planing(self.start, self.goal, self.obstacles)

            # Change mode
            self.mode_trajectory.set(3)
            return control_command
        elif self.mode_trajectory == 2:
            # Compute trajectory form waypoints
            self.run_planner(self.trajectory, True)

            # Change mode
            self.mode_trajectory.set(3)
            return control_command
        elif self.mode_trajectory == 3:
            control_command, done = self.tracker.trajectory_tracking (
                sensor_data, dt, self.GOAL_TOL)
            control_command[3] = sensor_data['yaw']
            
            if done and not self.tracker.repeat: 
                self.mode_trajectory.set(0)
        
        return control_command

    def dancing (self, sensor_data): # TODO
        pass

    def update_mode (self, actual_segment, pos):
        if self.mode == 0:
            # Verify if we have all the gates
            if len(self.pos_gates) == len(self.SEGMENT_LOCATION_GATES):
                location_gates = []

                for p_gate in self.pos_gates:
                    segment = self._get_actual_segment(p_gate[:3])
                    location_gates.append(segment)

                if (sorted(location_gates) == sorted(self.SEGMENT_LOCATION_GATES) 
                    and self.mode_searching == self.mode_searching.default):
                    # We have all the gates, we can run
                    self.mode.set()

                    for pos in self.pos_gates:
                        self._check_gate_pos(pos)
        elif self.mode == 1:
            # Check if we are in segment 0
            if np.linalg.norm(pos - self.INIT_POS) < 1e-2:
                self.mode.set()
        elif self.mode == 2:
            # Check if we pass through all the gates and went to zero
            # TODO: check each gate

            # if actual_segment == 0:
            #     self.mode.set()
            pass
        else:
            # Finish or something else
            pass

    def _get_actual_segment (self, global_pos):
        # Get cinlindrical coords
        _, theta, _ = self._coord_to_cilindrical(global_pos)
        dtheta = np.rad2deg (theta)
        
        # TODO: make beauty
        segment = None
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
    
    def _get_actual_segment_math (self, global_pos):
        return self._get_actual_segment(global_pos)
        # Get cinlindrical coords
        _, theta, _ = self._coord_to_cilindrical(global_pos)
        theta_deg = np.rad2deg (theta)
        
        # TODO: make beauty

        if theta_deg < -165 or theta_deg > 165:   segment = 9
        elif -165 <= theta_deg < -135:         segment = 10
        elif -135 <= theta_deg < -105:         segment = 11
        elif -105 <= theta_deg <  -75:         segment = 0
        elif  -75 <= theta_deg <  -45:         segment = 1
        elif  -45 <= theta_deg <  -15:         segment = 2
        elif  -15 <= theta_deg <   15:         segment = 3
        elif   15 <= theta_deg <   45:         segment = 4
        elif   45 <= theta_deg <   75:         segment = 5
        elif   75 <= theta_deg <  105:         segment = 6
        elif  105 <= theta_deg <  135:         segment = 7
        elif  135 <= theta_deg <  165:         segment = 8

        return segment
    
    def _cilindrical_to_coord (self, r, theta, z):
        x_centered = r * np.cos(theta)
        y_centered = r * np.sin(theta)

        rotated_pos = self.R_yaw90.T @ np.array([x_centered, y_centered, z])

        global_pos = rotated_pos + np.array([4, 4, 0])

        return global_pos

    def _coord_to_cilindrical (self, global_pos):
        # Fixed coordinates
        centered = (global_pos - np.array([4, 4, 0])) * np.array([1, 1, 1])
        rotated = self.R_yaw90 @ (centered)

        # Get theta
        theta = np.arctan2 (rotated[1], rotated[0])

        # return r, theta, z
        return np.linalg.norm (centered), theta, global_pos[2]
    
    def _achieve_goal (self, pos, goal, tol=None) -> bool:
        tol = self.GOAL_TOL if tol is None else tol
        if np.linalg.norm(pos - goal) < tol: 
            return True
        return False

    def _get_point_segment (self, segment, radius=3):
        # Get position in theta
        obj_theta = segment * np.deg2rad(30) - np.deg2rad(90)

        # Get in global pos
        global_pos = self._cilindrical_to_coord(radius, obj_theta, 1.35)

        return np.concatenate([global_pos, [obj_theta + np.deg2rad(0)]])

    def _compute_space_to_pos (self, pos, dist) -> np.ndarray:
        """
        Compute the position a certain distance to the given position
        
        :param pos: Given position size (4,) with (x, y, z, yaw)
        :param dist: Distance to the position in meters
        :return: The position with distance dist away of the posistion given
        :rtype: ndarray
        """ 
        # Get the normal vector
        normal = np.array([np.cos(pos[3]), np.sin(pos[3]), 0.0])

        # Target position
        target_pos = pos[:3] + normal * dist
        target_pos[2] = pos[2] # TODO: keep looking the gate

        # Compute yaw
        direction_to_gate = pos[:2] - target_pos[:2]
        yaw_to_gate = np.arctan2(direction_to_gate[1], direction_to_gate[0])

        return np.concatenate([target_pos, [yaw_to_gate]])

    def _get_idx_gate_search (self, gates_positions, idx_gate_search) -> int:
        """
        Get the index of the gate that we are searching for.
        
        :param gates_positions: Positions of the gates
        :param idx_gate_search: index of the gate that we are looking for
        :return: index of the gate in self.SEGMENT_LOCATION_GATES
        :rtype: int
        """
        segments = [self._get_actual_segment(pos["command"][:3]) for pos in gates_positions]
        # TODO: its probable that the gate its in the limit, try to solve that cases

        try: 
            idx_gate = segments.index(self.SEGMENT_LOCATION_GATES[idx_gate_search])
        except ValueError:
            idx_gate = None

        return idx_gate

    def _check_gate_pos (self, gate_pos):
        if self.drone is not None:
            print ("\nGATES POSITIONS")

            detected = False
            segment_gate = self._get_actual_segment(gate_pos[:3])
            for real in self.drone.gate_positions:
                segment = self._get_actual_segment(real)
                if np.linalg.norm (real - gate_pos[:3]) < 0.25:
                    print (f"> Gate detected correctly! Segment ({segment})")
                    print ("    > Real:", real)
                    print ("    > Estimated:", gate_pos)
                    detected = True
            if not detected:
                print ("> WARNING: Gate in segment", segment_gate, "not detected.")
                
class GatesDetector:

    CAM_FIELD_OF_VIEW = 1.5
    CAM_WIDTH = 300
    CAM_HEIGHT = 300

    CAM_FOCAL_DIST = CAM_WIDTH / (2 * np.tan(CAM_FIELD_OF_VIEW / 2))

    CAM_POS_REL = np.array([0.03, 0.0, 0.01])
    CAM_ROT_REL = R.from_euler('xz', [-np.deg2rad(90), -np.deg2rad(90)])

    GATE_HEIGHT = 0.4 # m

    def __init__ (self):
        self.k = np.array([
            [self.CAM_FOCAL_DIST, 0, self.CAM_WIDTH / 2],
            [0, self.CAM_FOCAL_DIST, self.CAM_HEIGHT / 2],
            [0, 0, 1]
        ])
        self.inv_k = np.linalg.inv(self.k)

        self.gates = []
        self.sensor_data = []

        self.last_mask = None

        self.timer = 0
        self.last_gates = None

    def pink_mask (self, camera_data):
        # https://stackoverflow.com/questions/70071741/opencv-python-how-recognize-pink-wood-in-the-image
        img = camera_data.copy()
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # PINK limits
        # H: 140–170, S: 50–255, V: 80–255
        COLOR_MIN = np.array([140, 50, 80], np.uint8) 
        COLOR_MAX = np.array([170, 255, 255], np.uint8)

        frame_threshed = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)
        ret, thresh = cv2.threshold(frame_threshed, 127, 255, 0)

        img_points, gates = self.find_boxes (frame_threshed, img)
        self.last_mask = img_points

        return img_points, gates
    
    def get_buffer (self, gates, dt):
        self.timer += dt

        if (len(gates) == 0 and self.timer < 0.5 
            and (self.last_gates is not None and len(self.last_gates) != 0)):
            return self.last_gates
        else:
            self.last_gates = gates
            self.timer = 0
            return gates

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
    
    def compute_gates_position (
            self, gates_corners, R_drone_to_world: R, T_drone_to_world: np.ndarray
        ) -> List[Dict[str, Union[np.ndarray, bool]]]:
        positions = []
        for gate in gates_corners:

            cam_rel_pos_left = self.get_rel_center_pos (gate[0], gate[3], self.GATE_HEIGHT)
            cam_rel_pos_right = self.get_rel_center_pos (gate[1], gate[2], self.GATE_HEIGHT)

            # Transform to world frame
            drone_rel_pos_left = self.CAM_ROT_REL.apply(cam_rel_pos_left) + self.CAM_POS_REL
            gate_pos_left = R_drone_to_world.apply(drone_rel_pos_left) + T_drone_to_world

            drone_rel_pos_right= self.CAM_ROT_REL.apply(cam_rel_pos_right) + self.CAM_POS_REL
            gate_pos_right = R_drone_to_world.apply(drone_rel_pos_right) + T_drone_to_world

            valid_gate = False
            # The width has to be larger than 0.3 m
            if np.linalg.norm(gate_pos_left - gate_pos_right) > 0.29:
                valid_gate = True
            # if the position of the corners are near to the limits of the camera
            if not self.gate_on_frame_limits (gate):
                valid_gate = False

            # Center position
            gate_center = (gate_pos_left + gate_pos_right) / 2

            # Gate orientation
            lateral = gate_pos_right - gate_pos_left
            drone_to_gate = gate_center - T_drone_to_world
            normal = np.array([-lateral[1], lateral[0], 0])

            # Flip normal if it points away from the drone
            if np.dot(normal, drone_to_gate) < 0:
                normal = -normal
            gate_yaw = np.arctan2(normal[1], normal[0])

            positions.append({
                "command": np.concatenate([gate_center, [gate_yaw]]),
                "valid": valid_gate
            })

        return positions

    def get_rel_center_pos (self, pos_pixel_top: np.ndarray, pos_pixel_bottom: np.ndarray, real_dist: float) -> np.ndarray:
        """
        Get the relative position of the center point between top and bottom (or different points) of 
        an object where you know the real distance between each point.
        
        :param pos_pixel_top: Position of the pixel in camera frame
        :param pos_pixel_bottom: Position of the pixel in camera frame
        :param real_dist: Real distance in meter between each point.
        :return: Relative position of the center point between points in camera frame (in meters).
        :return type: np.ndarray of shape (3,)

        NOTE: This position will be not accurate if the object is rotated, try to go closer and get a
        better aproximation.
        """

        # Get distnace to arist
        dist_between_pixels = np.linalg.norm (pos_pixel_top - pos_pixel_bottom)
        dist_to_center = self.get_distance(dist_between_pixels, real_dist)

        # Get x,y rates
        pos_center_pixel = (pos_pixel_top + pos_pixel_bottom) / 2
        pixel = np.concatenate([pos_center_pixel, [1]])
        pos_rate = self.inv_k @ pixel

        # Get position relative to camera
        p_z = dist_to_center / np.linalg.norm (pos_rate)
        p_y = pos_rate[1] * p_z
        p_x = pos_rate[0] * p_z

        return np.array([p_x, p_y, p_z])

    def get_distance (self, dist_px, size_object):
        return size_object * self.CAM_FOCAL_DIST / max(dist_px, 1e-6) # meters
        
    def compute_gate_position (self, gate_corners, sensor) -> Tuple[np.ndarray, bool]:
        self.gates.append(gate_corners)
        self.sensor_data.append(sensor)

        if len(self.gates) > 1:
            # Compute the position
            pass
        else:
            return None, False
    
    def gate_on_frame_limits (self, gate, limit=5) -> bool:
        for corner in gate:
            if ((0 + limit > corner[0] or corner[0] > self.CAM_WIDTH - limit) or
                (0 + limit > corner[1] or corner[1] > self.CAM_HEIGHT - limit)):
                return False
        return True

@dataclass
class CameraObservation:
    """Stores one camera frame observation of gate corners together with the drone pose."""
    gate_corners: np.ndarray          # shape (4, 2) – ordered [TL, TR, BR, BL]
    R_drone_to_world: R               # drone orientation in world frame
    T_drone_to_world: np.ndarray      # drone position  in world frame (3,)


@dataclass
class GateEstimate:
    """Full gate estimate produced by triangulation (or fallback mono estimate)."""
    center: np.ndarray                # (3,) world-frame XYZ
    yaw: float                        # radians
    valid: bool
    method: str                       # "triangulation" | "mono"
    residual: Optional[float] = None  # distance |FG| – quality indicator (triangulation only)

    @property
    def command(self) -> np.ndarray:
        return np.concatenate([self.center, [self.yaw]])


class GatesDetectorTriangulation:
    """
    Gate detector that uses two-view triangulation (VIO appendix method) to estimate
    the 3-D position of racing-gate corners.

    Workflow
    --------
    1. Call ``load_observation(camera_data, R_drone, T_drone)`` each time a new
       camera frame arrives (with its associated drone pose).
    2. After at least two observations have been loaded, call
       ``compute_gate_positions()`` to get triangulated estimates.
    3. Alternatively, call the single-frame helper ``mono_gate_positions()`` to
       fall back to the height-based monocular estimate from the original class.

    Camera constants are identical to GatesDetector so both classes are
    interchangeable.
    """

    # ------------------------------------------------------------------ #
    #  Camera / gate constants (mirror GatesDetector)                     #
    # ------------------------------------------------------------------ #
    CAM_FIELD_OF_VIEW: float = 1.5          # rad
    CAM_WIDTH:  int = 300                   # px
    CAM_HEIGHT: int = 300                   # px

    CAM_FOCAL_DIST: float = CAM_WIDTH / (2 * np.tan(CAM_FIELD_OF_VIEW / 2))

    # Camera offset relative to the drone body frame
    CAM_POS_REL: np.ndarray = np.array([0.03, 0.0, 0.01])
    # Rotation: body → camera  (zcam = xdrone, xcam = -ydrone, ycam = -zdrone)
    CAM_ROT_REL: R = R.from_euler('xz', [-np.deg2rad(90), -np.deg2rad(90)])

    GATE_HEIGHT: float = 0.4            # m – vertical post height
    GATE_WIDTH:  float = 0.6            # m – horizontal width (for validity check)

    # Buffer: keep the last N observations per detected gate
    MAX_OBS_BUFFER: int = 10

    # ------------------------------------------------------------------ #
    #  Constructor                                                         #
    # ------------------------------------------------------------------ #
    def __init__(self) -> None:
        self.k = np.array([
            [self.CAM_FOCAL_DIST, 0,                   self.CAM_WIDTH  / 2],
            [0,                   self.CAM_FOCAL_DIST,  self.CAM_HEIGHT / 2],
            [0,                   0,                   1],
        ])
        self.inv_k = np.linalg.inv(self.k)

        # Ring-buffer of observations (one list per tracked gate slot)
        # For simplicity we keep a single observation queue; you can extend
        # this to multi-gate tracking with a gate-association step.
        self._observations: List[CameraObservation] = []

        # Last valid estimate for temporal smoothing / gap-filling
        self._last_estimates: Optional[List[GateEstimate]] = None
        self._gap_timer: float = 0.0
        self.GAP_FILL_DURATION: float = 0.5   # seconds

        self.last_mask: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def load_observation(
        self,
        camera_data: np.ndarray,
        R_drone_to_world: R,
        T_drone_to_world: np.ndarray,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Detect gates in *camera_data*, store the observation, and return the
        annotated image together with the raw list of corner arrays.

        Parameters
        ----------
        camera_data      : RGB image (H × W × 3).
        R_drone_to_world : Current drone orientation.
        T_drone_to_world : Current drone position in world frame (3,).

        Returns
        -------
        annotated_img  : Image with drawn corners.
        gates_corners  : List of (4,2) corner arrays for each detected gate.
        """
        annotated_img, gates_corners = self._detect_pink_gates(camera_data)

        for gate_corners in gates_corners:
            obs = CameraObservation(
                gate_corners=gate_corners,
                R_drone_to_world=R_drone_to_world,
                T_drone_to_world=T_drone_to_world.copy(),
            )
            self._observations.append(obs)

        # Keep buffer bounded
        if len(self._observations) > self.MAX_OBS_BUFFER:
            self._observations = self._observations[-self.MAX_OBS_BUFFER:]

        return annotated_img, gates_corners

    def compute_gate_positions(
        self,
        R_drone_to_world: Optional[R] = None,
        T_drone_to_world: Optional[np.ndarray] = None,
    ) -> List[GateEstimate]:
        """
        Compute gate positions using triangulation from the two most recent
        observations.  Falls back to mono estimate if only one observation is
        available (requires current pose arguments in that case).

        Returns a list of GateEstimate objects (one per gate detected in the
        latest observation).
        """
        if len(self._observations) < 2:
            if R_drone_to_world is not None and T_drone_to_world is not None:
                return self._mono_from_last_obs(R_drone_to_world, T_drone_to_world)
            return []

        obs1 = self._observations[-2]
        obs2 = self._observations[-1]
        return self._triangulate_pair(obs1, obs2)

    def mono_gate_positions(
        self,
        gates_corners: List[np.ndarray],
        R_drone_to_world: R,
        T_drone_to_world: np.ndarray,
    ) -> List[GateEstimate]:
        """
        Original height-based monocular position estimate (kept as fallback).
        Mirrors ``compute_gates_position`` from GatesDetector but returns
        GateEstimate objects.
        """
        estimates = []
        for gate in gates_corners:
            left_pos  = self._get_rel_center_pos(gate[0], gate[3], self.GATE_HEIGHT)
            right_pos = self._get_rel_center_pos(gate[1], gate[2], self.GATE_HEIGHT)

            gate_pos_left  = self._cam_to_world(left_pos,  R_drone_to_world, T_drone_to_world)
            gate_pos_right = self._cam_to_world(right_pos, R_drone_to_world, T_drone_to_world)

            valid = (
                np.linalg.norm(gate_pos_left - gate_pos_right) > 0.29
                and self._gate_on_frame_limits(gate)
            )

            center, yaw = self._center_and_yaw(gate_pos_left, gate_pos_right, T_drone_to_world)
            estimates.append(GateEstimate(center=center, yaw=yaw, valid=valid, method="mono"))

        return estimates

    def get_buffered_estimates(
        self,
        estimates: List[GateEstimate],
        dt: float,
    ) -> List[GateEstimate]:
        """Temporal gap-filler: return last valid estimate for up to GAP_FILL_DURATION seconds."""
        self._gap_timer += dt
        if (
            len(estimates) == 0
            and self._gap_timer < self.GAP_FILL_DURATION
            and self._last_estimates
        ):
            return self._last_estimates

        self._last_estimates = estimates
        self._gap_timer = 0.0
        return estimates

    # ------------------------------------------------------------------ #
    #  Triangulation core (appendix method)                               #
    # ------------------------------------------------------------------ #

    def _triangulate_pair(
        self, obs1: CameraObservation, obs2: CameraObservation
    ) -> List[GateEstimate]:
        """
        Triangulate ALL four gate corners using the two-line midpoint method
        described in the VIO appendix.

        For each corner pair (left column from obs1, left column from obs2, etc.)
        we triangulate independently, then average to get the gate centre.
        """
        estimates = []

        # We associate obs1 and obs2 as two views of the same gate.
        # Corner indices: 0=TL, 1=TR, 2=BR, 3=BL
        # Left post: corners 0 (top) and 3 (bottom)
        # Right post: corners 1 (top) and 2 (bottom)

        for corner_top_idx, corner_bot_idx, label in [
            (0, 3, "left"),
            (1, 2, "right"),
        ]:
            # --- Camera positions in world frame ---
            P = self._camera_world_pos(obs1.R_drone_to_world, obs1.T_drone_to_world)
            Q = self._camera_world_pos(obs2.R_drone_to_world, obs2.T_drone_to_world)

            # --- Bearing vectors in world frame ---
            # For each observation we use the *midpoint* pixel of the post
            mid1 = (obs1.gate_corners[corner_top_idx] + obs1.gate_corners[corner_bot_idx]) / 2
            mid2 = (obs2.gate_corners[corner_top_idx] + obs2.gate_corners[corner_bot_idx]) / 2

            r = self._pixel_to_world_ray(mid1, obs1.R_drone_to_world)
            s = self._pixel_to_world_ray(mid2, obs2.R_drone_to_world)

            # --- Solve for λ, μ via pseudo-inverse (appendix eq. 9) ---
            A = np.column_stack([r, -s])            # 3×2
            b = Q - P
            lm, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            lam, mu = lm

            F = P + lam * r                          # point on ray 1
            G = Q + mu  * s                          # point on ray 2

            midpoint = (F + G) / 2                   # triangulated 3-D point (appendix eq. 8)
            residual = float(np.linalg.norm(F - G))  # quality indicator

            if label == "left":
                left_3d  = midpoint
                left_res = residual
            else:
                right_3d  = midpoint
                right_res = residual

        # Gate validity: width and frame limits
        width = np.linalg.norm(left_3d - right_3d)
        valid = (
            width > 0.29
            and self._gate_on_frame_limits(obs2.gate_corners)
        )

        center, yaw = self._center_and_yaw(left_3d, right_3d, obs2.T_drone_to_world)
        avg_residual = (left_res + right_res) / 2

        estimates.append(GateEstimate(
            center=center,
            yaw=yaw,
            valid=valid,
            method="triangulation",
            residual=avg_residual,
        ))
        return estimates

    def triangulate_pixel_pair(
        self,
        pixel1: np.ndarray,
        R1: R,
        T1: np.ndarray,
        pixel2: np.ndarray,
        R2: R,
        T2: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Low-level helper: triangulate a single pixel seen in two frames.

        Parameters
        ----------
        pixel1, pixel2 : (2,) pixel coordinates in each image.
        R1, T1         : Drone pose for frame 1.
        R2, T2         : Drone pose for frame 2.

        Returns
        -------
        point_3d  : (3,) triangulated world-frame point.
        residual  : |FG| – distance between the two closest points on rays.
        """
        P = self._camera_world_pos(R1, T1)
        Q = self._camera_world_pos(R2, T2)

        r = self._pixel_to_world_ray(pixel1, R1)
        s = self._pixel_to_world_ray(pixel2, R2)

        A = np.column_stack([r, -s])
        b = Q - P
        lm, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        lam, mu = lm

        F = P + lam * r
        G = Q + mu  * s
        return (F + G) / 2, float(np.linalg.norm(F - G))

    # ------------------------------------------------------------------ #
    #  Geometry helpers                                                    #
    # ------------------------------------------------------------------ #

    def _camera_world_pos(self, R_drone_to_world: R, T_drone_to_world: np.ndarray) -> np.ndarray:
        """Compute the camera optical-centre position in world frame."""
        return R_drone_to_world.apply(self.CAM_ROT_REL.apply(np.zeros(3)) + self.CAM_POS_REL) + T_drone_to_world

    def _pixel_to_world_ray(self, pixel: np.ndarray, R_drone_to_world: R) -> np.ndarray:
        """
        Convert a pixel coordinate to a unit bearing vector in world frame.

        Steps (appendix §1):
          1. Unproject pixel through K⁻¹ → direction in camera frame.
          2. Rotate camera → drone body (CAM_ROT_REL).
          3. Rotate drone body → world (R_drone_to_world).
        """
        px_h = np.array([pixel[0], pixel[1], 1.0])
        v_cam = self.inv_k @ px_h                              # direction in camera frame
        v_body = self.CAM_ROT_REL.apply(v_cam)                # camera → body
        v_world = R_drone_to_world.apply(v_body)               # body → world
        return v_world / np.linalg.norm(v_world)               # unit vector

    def _cam_to_world(
        self,
        p_cam: np.ndarray,
        R_drone_to_world: R,
        T_drone_to_world: np.ndarray,
    ) -> np.ndarray:
        """Transform a point from camera frame to world frame."""
        p_body = self.CAM_ROT_REL.apply(p_cam) + self.CAM_POS_REL
        return R_drone_to_world.apply(p_body) + T_drone_to_world

    def _get_rel_center_pos(
        self,
        pos_pixel_top: np.ndarray,
        pos_pixel_bottom: np.ndarray,
        real_dist: float,
    ) -> np.ndarray:
        """Monocular depth estimate from known object height (original method)."""
        dist_px = np.linalg.norm(pos_pixel_top - pos_pixel_bottom)
        depth   = real_dist * self.CAM_FOCAL_DIST / max(dist_px, 1e-6)

        mid_px = (pos_pixel_top + pos_pixel_bottom) / 2
        ray    = self.inv_k @ np.array([mid_px[0], mid_px[1], 1.0])
        p_z    = depth / np.linalg.norm(ray)
        return np.array([ray[0] * p_z, ray[1] * p_z, p_z])

    @staticmethod
    def _center_and_yaw(
        left_3d: np.ndarray,
        right_3d: np.ndarray,
        drone_pos: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Compute gate centre and yaw from left/right post positions."""
        center  = (left_3d + right_3d) / 2
        lateral = right_3d - left_3d
        normal  = np.array([-lateral[1], lateral[0], 0.0])
        if np.dot(normal, center - drone_pos) < 0:
            normal = -normal
        yaw = float(np.arctan2(normal[1], normal[0]))
        return center, yaw

    # ------------------------------------------------------------------ #
    #  Detection helpers (mirror GatesDetector)                           #
    # ------------------------------------------------------------------ #

    def _detect_pink_gates(
        self, camera_data: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        img = camera_data.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([140, 50, 80], np.uint8),
                                np.array([170, 255, 255], np.uint8))
        annotated, gates = self._find_boxes(mask, img)
        self.last_mask = annotated
        return annotated, gates

    def _find_boxes(
        self, mask: np.ndarray, img: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        gates = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 50:
                continue
            eps    = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True)
            if len(approx) != 4:
                continue

            corners = approx.reshape(4, 2)
            s    = corners.sum(axis=1)
            diff = np.diff(corners, axis=1).flatten()
            ordered = np.array([
                corners[np.argmin(s)],      # TL
                corners[np.argmin(diff)],   # TR
                corners[np.argmax(s)],      # BR
                corners[np.argmax(diff)],   # BL
            ])
            gates.append(ordered)
            for pt in ordered:
                cv2.circle(img, tuple(pt), 4, (255, 0, 0), -1)
        return img, gates

    def _gate_on_frame_limits(self, gate: np.ndarray, limit: int = 5) -> bool:
        for corner in gate:
            if (corner[0] < limit or corner[0] > self.CAM_WIDTH  - limit or
                    corner[1] < limit or corner[1] > self.CAM_HEIGHT - limit):
                return False
        return True

    def _mono_from_last_obs(
        self, R_drone_to_world: R, T_drone_to_world: np.ndarray
    ) -> List[GateEstimate]:
        if not self._observations:
            return []
        last = self._observations[-1]
        return self.mono_gate_positions(
            [last.gate_corners], R_drone_to_world, T_drone_to_world
        )

    # ------------------------------------------------------------------ #
    #  Diagnostics                                                         #
    # ------------------------------------------------------------------ #

    def clear_observations(self) -> None:
        """Flush the observation buffer (e.g. after crossing a gate)."""
        self._observations.clear()

    def observation_count(self) -> int:
        return len(self._observations)
    
class Tracker:
    def __init__ (self):
        self.reset()

    def reset(self):
        self.index_current_setpoint = None
        self.timer = None
        self.timer_done = None
        self.repeat = False

    def load (self, timepoints, setpoints, vel_setpoints, acc_setpoints):
        self.timepoints = timepoints
        self.setpoints = setpoints
        self.vel_sp = vel_setpoints
        self.acc_sp = acc_setpoints
        
    def trajectory_tracking(
            # self, sensor_data, dt, timepoints, setpoints, critic_sepoints=None, tol=1e-1, repeat = None):
            self, sensor_data, dt, tol=1e-1, repeat = None, method=1):
        repeat = self.repeat if repeat is None else repeat

        show = len(self.setpoints) > 4

        start_point = self.setpoints[0]
        end_point = self.setpoints[-1]

        if self.timer is None:
            # Begin timer and start trajectory
            self.timer = 0
            print("Trajectory tracking started")
            self.index_current_setpoint = 1
        else:
            self.timer += dt

        # Determine the current setpoint based on the time
        if self.timer is not None:
            if self.index_current_setpoint < len(self.timepoints) - 1:
                # Update new setpoint
                # Method 1
                # if self.timer >= self.timepoints[self.index_current_setpoint]:
                #     self.index_current_setpoint += 1
                current_setpoint = np.concatenate((
                    self.setpoints[self.index_current_setpoint,:],
                    self.vel_sp[self.index_current_setpoint],
                    self.acc_sp[self.index_current_setpoint]
                ))

                # Method 2
                if method == 2:
                    # current_setpoint = self.setpoints[self.index_current_setpoint,:]
                    # if self.timer >= self.timepoints[self.index_current_setpoint]:
                    #     if critic_sepoints is not None and critic_sepoints[self.index_current_setpoint]:
                    #         verify_distance = (
                    #             abs(sensor_data['x_global'] - current_setpoint[0]) < 0.12
                    #             and abs(sensor_data['y_global'] - current_setpoint[1]) < 0.12
                    #             and abs(sensor_data['z_global'] - current_setpoint[2]) < 0.12
                    #         )

                    #         if verify_distance:
                    #             self.index_current_setpoint += 1
                    #     else:
                    #         self.index_current_setpoint += 1
                    pass

                # Method 3
                elif method == 3:
                    current_setpoint = self.setpoints[self.index_current_setpoint,:]
                    verify_distance = (
                        abs(sensor_data['x_global'] - current_setpoint[0]) < 0.12
                        and abs(sensor_data['y_global'] - current_setpoint[1]) < 0.12
                        and abs(sensor_data['z_global'] - current_setpoint[2]) < 0.12
                    )
                    if verify_distance:
                        self.index_current_setpoint += 1

                # Method 4
                elif method == 4:
                    POS_TOL      = 0.20   # m  — must be within this to advance
                    TIME_TIMEOUT = 1.5    # s  — advance anyway after this much extra time (safety)

                    actual_pos = np.array([
                        sensor_data['x_global'], sensor_data['y_global'], 
                        sensor_data['z_global'], sensor_data['yaw']]
                    )
                    current_sp_pos = self.setpoints[self.index_current_setpoint, :3]
                    pos_error      = np.linalg.norm(actual_pos[:3] - current_sp_pos)

                    time_due    = self.timer >= self.timepoints[self.index_current_setpoint]
                    close_enough = pos_error < POS_TOL
                    timed_out   = self.timer >= self.timepoints[self.index_current_setpoint] + TIME_TIMEOUT

                    if (time_due and close_enough) or timed_out:
                        self.index_current_setpoint += 1
                else:
                    # Method 1
                    if self.timer >= self.timepoints[self.index_current_setpoint]:
                        self.index_current_setpoint += 1
                    current_setpoint = self.setpoints[self.index_current_setpoint,:]

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
        self.velocity_setpoints = None
        self.acceleration_setpoints = None
        self.obstacles = []
        self.final_time = 3.5

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

    def run_planner(self, path_waypoints, **kwargs):    
        # Run the subsequent functions to compute the polynomial coefficients and extract and visualize the trajectory setpoints

        self.init_params(path_waypoints, **kwargs)
        poly_coeffs = self.compute_poly_coefficients(path_waypoints)
        # self.trajectory_setpoints, self.time_setpoints, self.critic_setpoint, _ = \
        self.trajectory_setpoints, self.time_setpoints, self.velocity_setpoints, self.acceleration_setpoints, _ = \
            self.poly_setpoint_extraction(poly_coeffs, self.obstacles, path_waypoints)

        ## ---------------------------------------------------------------------------------------------------- ##

    def run_planner_opt (self, path_waypoints):

        tol = 1e-2
        learn_vel = 0.2
        learn_acc = 0.1
        t = 10
        done = False

        trajectory_setpoint, time_setpoint = None, None
        count = 0

        while not done:
            # Copmute trajectory
            self.init_params(path_waypoints, t_final=t)
            poly_coeffs = self.compute_poly_coefficients(path_waypoints)
            traj_set, time_set, _, _, info = self.poly_setpoint_extraction(poly_coeffs, self.obstacles, path_waypoints)

            result, msg = self.check_limits(info)
            if result:
                # Save good traj
                trajectory_setpoint, time_setpoint = traj_set, time_set
                
                # Compute diff
                vel_disponible = self.vel_lim - info["vel"]["max"]
                acc_disponible = self.acc_lim - info["acc"]["max"]

                # Compute new time
                # new_time = max (t - vel_disponible * learn_vel, t - acc_disponible * learn_acc)
                new_time = t - vel_disponible * learn_vel

                # Check if we converge or we find a limit
                if new_time < 0 or (abs(t - new_time) < tol):
                    done = True 
                else:
                    t = new_time
            else: 
                done = True
                print (msg)

            count += 1

        print ("Trajectory computed in", count, "steps")

        self.trajectory_setpoints, self.time_setpoints = trajectory_setpoint, time_setpoint

    def init_params(self, path_waypoints, t_final=None):

        # Inputs:
        # - path_waypoints: The sequence of input path waypoints provided by the path-planner, including the start and final goal position: Vector of m waypoints, consisting of a tuple with three reference positions each as provided by AStar

        # TUNE THE FOLLOWING PARAMETERS (PART 2) ----------------------------------------------------------------- ##
        self.disc_steps = 10 #Integer number steps to divide every path segment into to provide the reference positions for PID control # IDEAL: Between 10 and 20
        self.vel_lim = 7.0 #Velocity limit of the drone (m/s)
        self.acc_lim = 50.0 #Acceleration limit of the drone (m/s²)
        t_f = self.final_time if t_final is None else t_final

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
        velocity_setpoints = np.hstack((v_x_vals, v_y_vals, v_z_vals))
        acceleration_setpoints = np.hstack((a_x_vals, a_y_vals, a_z_vals))
        
        # setpoint_is_waypoint = []
        # TOCLEAN: Add critical points
        # for point in trajectory_setpoints:
        #     for waypoint in path_waypoints:
        #         if np.linalg.norm(point[:3] - waypoint[:3]) < 0.25:
        #             setpoint_is_waypoint.append(True)
        #         else:
        #             setpoint_is_waypoint.append(False)

        # if len(path_waypoints) > 3:
            # self.plot(obs, path_waypoints, trajectory_setpoints)
            
        # Find the maximum absolute velocity during the segment
        vel_max = np.max(np.sqrt(v_x_vals**2 + v_y_vals**2 + v_z_vals**2))
        vel_mean = np.mean(np.sqrt(v_x_vals**2 + v_y_vals**2 + v_z_vals**2))
        acc_max = np.max(np.sqrt(a_x_vals**2 + a_y_vals**2 + a_z_vals**2))
        acc_mean = np.mean(np.sqrt(a_x_vals**2 + a_y_vals**2 + a_z_vals**2))

        info = {
            "vel": {
                "max": vel_max,
                "mean": vel_mean
            },
            "acc": {
                "max": acc_max,
                "mean": acc_mean
            }
        }
        # ---------------------------------------------------------------------------------------------------- ##

        return trajectory_setpoints, time_setpoints, velocity_setpoints, acceleration_setpoints, info
    
    def check_limits (self, info) -> Tuple[bool, str]:
        # Check that it is less than an upper limit velocity v_lim
        if info["vel"]["max"] > self.vel_lim:
            return False, "The drone velocity exceeds the limit velocity : " + str(info["vel"]["max"]) + " m/s"
        if info["acc"]["max"] > self.acc_lim:
            return False, "The drone acceleration exceeds the limit acceleration : " + str(info["acc"]["max"]) + " m/s²"
        return True, ""
    
    def plot_obstacle(self, ax, x, y, z, dx, dy, dz, color='gray', alpha=0.3):

        # DO NOT MODIFY --------------------------------------------------------------------------------------- ##

        """Plot a rectangular cuboid (obstacle) in 3D space."""
        vertices = np.array([[x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],
                            [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]])
        
        faces = [[vertices[j] for j in [0, 1, 2, 3]], [vertices[j] for j in [4, 5, 6, 7]], 
                [vertices[j] for j in [0, 1, 5, 4]], [vertices[j] for j in [2, 3, 7, 6]], 
                [vertices[j] for j in [0, 3, 7, 4]], [vertices[j] for j in [1, 2, 6, 5]]]
        
        ax.add_collection3d(Poly3DCollection(faces, color=color, alpha=alpha))
    
    def plot(self, obs, path_waypoints, trajectory_setpoints):

        # DO NOT MODIFY --------------------------------------------------------------------------------------- ##

        # Plot 3D trajectory
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        for ob in obs:
            self.plot_obstacle(ax, ob[0], ob[1], ob[2], ob[3], ob[4], ob[5])

        ax.plot(trajectory_setpoints[:,0], trajectory_setpoints[:,1], trajectory_setpoints[:,2], label="Minimum-Jerk Trajectory", linewidth=2)
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_zlim(0, 2.5)

        # Plot waypoints
        waypoints_x = [p[0] for p in path_waypoints]
        waypoints_y = [p[1] for p in path_waypoints]
        waypoints_z = [p[2] for p in path_waypoints]
        ax.scatter(waypoints_x, waypoints_y, waypoints_z, color='red', marker='o', label="Waypoints")

        # Labels and legend
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")
        ax.set_title("3D Motion planning trajectories")
        ax.legend()
        plt.show()

class Keeper ():
    def __init__ (self, value_default=0):
        self.default = value_default
        self.value = value_default
        self.last = value_default

    def set(self, value=None):
        if value is None:
            value = self.value + 1

        self.last = self.value 
        self.value = value

    def get(self):
        return self.value
    
    def return_to_last(self):
        now_value = self.value 
        self.value = self.last 
        self.last = now_value
    
    def __eq__(self, other):
        return self.value == other
    
    def __repr__(self):
        return str(self.value)

# Module-level singleton so main.py can call assignment.get_command() unchanged
_controller = MyAssignment()
_detector = GatesDetectorTriangulation()

def get_command(sensor_data, camera_data, dt):
    return _controller.compute_command(sensor_data, camera_data, dt)

def show_mask (camera_data, drone):
    if _controller.drone is None:
        _controller.drone = drone

    if _controller.detector.last_mask is None:
        img = camera_data.copy()
    else:
        img = _controller.detector.last_mask
    return img

def draw_stats(image: np.ndarray = None) -> np.ndarray:
    """Draw stats overlay on the image."""
    stats = [
        f"Mode: {_controller.mode}",
        f"Mode Search: {_controller.mode_searching}",
        f"Mode Traj: {_controller.mode_trajectory}",
        f"Gate idx: {_controller.idx_gate_search}",
        f"Target: {np.round(_controller.target_control_command, 2) if _controller.target_control_command is not None else 'None'}",
        f"Command: {np.round(_controller.last_command, 2)}",
        f"Len Gates Detected: {len(_controller.pos_gates)}",
    ]

    # Semi-transparent background
    if image is None:
        image = np.zeros((300, 300, 3), dtype=np.uint8)
    overlay = image.copy()
    cv2.rectangle(overlay, (5, 5), (300, 20 + 25 * len(stats)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)

    # Draw each line
    for i, line in enumerate(stats):
        cv2.putText(
            image,
            line,
            (10, 25 + i * 25),        # x, y position (moves down each line)
            cv2.FONT_HERSHEY_SIMPLEX,  # font
            0.6,                        # font scale
            (0, 255, 0),               # color (green)
            1,                          # thickness
            cv2.LINE_AA                # anti-aliased
        )

    return image

if __name__ == "__main__":
    pos = np.array([4.1, 4, 0])
    pos = np.array([1, 4, 0.1])
    print(_controller._get_actual_segment(pos))

    cilindrical = _controller._coord_to_cilindrical(pos)
    print(cilindrical[0], np.rad2deg(cilindrical[1]), cilindrical[2])
    
    new_pos = _controller._cilindrical_to_coord(*cilindrical)
    print(new_pos)