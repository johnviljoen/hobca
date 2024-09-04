import heapq
import numpy as np
import scipy.spatial
from matplotlib.path import Path
import matplotlib.pyplot as plt
from heapdict import heapdict

from utils.transforms import get_corners
from utils.transforms import get_corners
from utils.reeds_shepp import pi_2_pi, calc_all_paths
from utils.tpcap import plot_case

#### All Parameters that define Car and Planner ####

# Car
car_params = {
    "wheel_base": 2.8,
    "width": 1.942,
    "front_hang": 0.96,
    "rear_hang": 0.929,
    "max_steer": 0.5,
}

# Testing equivalent setup to other hybrid a*

# LB = 2.3
# LF = 2.3
# max_steer = np.deg2rad(40)
# total_length = LB + LF
# wheel_base = 2.7
# width = 1.85
# front_hang = LF - wheel_base/2
# rear_hang = LB - wheel_base/2

# car_params = {
#     "wheel_base": wheel_base,
#     "width": width,
#     "front_hang": front_hang,
#     "rear_hang": rear_hang,
#     "max_steer": max_steer,
# }

# bubble for fast detection of potential collisions later on
car_params["total_length"] = car_params["rear_hang"] + car_params["wheel_base"] + car_params["front_hang"]
car_params["bubble_radius"] = np.hypot(car_params["total_length"] / 2, car_params["width"] / 2)

# origin is defined around the rear axle, default orientiation is facing east
car_params["corners"] = np.array([
    [car_params["wheel_base"] + car_params["front_hang"], car_params["width"] / 2], # front left
    [- car_params["rear_hang"], car_params["width"] / 2], # back left
    [- car_params["rear_hang"], - car_params["width"] / 2], # back right
    [car_params["wheel_base"] + car_params["front_hang"], - car_params["width"] / 2] # front right
])

car_params["center_to_front"] = car_params["wheel_base"]/2 + car_params["front_hang"]
car_params["center_to_back"] = car_params["wheel_base"]/2 + car_params["rear_hang"]

#### Planner ####

planner_params = {
    "xy_resolution": 0.5,
    "yaw_resolution": np.deg2rad(5.0),
    "steer_options": 5,
    "movement_options": 2,   # resolution of action space at every timestep of the search
    "max_movement": 1.0,          # max movement forwards or backwards at every timestep of the search
    "max_iter": 100,
    "reverse_cost": 1.0,         # used in reeds shepp cost in reeds_shepp.py
    "direction_change_cost": 1.0, # used in reeds shepp cost in reeds_shepp.py
    "steer_angle_cost": 0.5,      # used in reeds shepp cost in reeds_shepp.py
    "steer_angle_change_cost": 0.5, # ^ same
    "rs_step_size": 1.0,          # the rs step size used in calculating the reeds shepp trajectory at the leaves
    "hybrid_cost": 1.0,            # used in holonomic_cost_map weighting in non_holonomic_search.py
    "kinematic_simulation_length": 1,   # for the kinematic simulation node in non_holonomic_search.py
    "kinematic_simulation_step": 1.0
}

#### Grid Formation ####

def calculate_obstacle_grid_and_kdtree(planner_params, case_params):
    """
    Here we calculate the grid and where the obstacles lie within it, according to the resolutions of
    the planner. We also create a scipy.spatial.KDTree data structure for the discretized obstacle positions
    in the grid, this allows fast calculation of ball distances to all obstacles at runtime as a first 
    check for collision. If a collision is then detected according to this we calculate the true collisions.
    This ends up saving a lot of time when we are not close to obstacles.

    Args:
        planner_params (Dict): defining parameters of the hybrid A* planner
        case_params (Dict): defining parameters of the environment we are in

    Returns:
        grid (): _description_
        grid_bounds (): _description_
        obstacle_kdtree (): _description_
    """
    grid_width = int((case_params["xmax"] - case_params["xmin"]) // planner_params["xy_resolution"] + 1)
    grid_height = int((case_params["ymax"] - case_params["ymin"]) // planner_params["xy_resolution"] + 1)

    grid = np.zeros((grid_width, grid_height))
    obstacle_x_idx = []
    obstacle_y_idx = []
    for obs in case_params["obs"]:
        path = Path(obs)
        
        # Create a meshgrid for the bounding box
        x_range = np.arange(case_params['xmin'], case_params["xmax"], planner_params["xy_resolution"])
        y_range = np.arange(case_params["ymin"], case_params["ymax"], planner_params["xy_resolution"])
        xv, yv = np.meshgrid(x_range, y_range)
        points = np.vstack((xv.flatten(), yv.flatten())).T
        
        # Check which points are inside the obstacle
        inside = path.contains_points(points)
        
        # Get grid coordinates of the inside points
        for point in points[inside]:
            grid_x_idx = int((point[0] - case_params["xmin"]) // planner_params["xy_resolution"])
            grid_y_idx = int((point[1] - case_params["ymin"]) // planner_params["xy_resolution"])
            grid[grid_x_idx, grid_y_idx] = 1
            obstacle_x_idx.append(grid_x_idx)
            obstacle_y_idx.append(grid_y_idx)
    obstacle_x_idx = np.array(obstacle_x_idx)
    obstacle_y_idx = np.array(obstacle_y_idx)

    # calculate the map bounds in terms of the grid
    grid_bounds = {}
    grid_bounds["xmax"] = round(case_params["xmax"] / planner_params["xy_resolution"])
    grid_bounds["xmin"] = round(case_params["xmin"] / planner_params["xy_resolution"])
    grid_bounds["ymax"] = round(case_params["ymax"] / planner_params["xy_resolution"])
    grid_bounds["ymin"] = round(case_params["ymin"] / planner_params["xy_resolution"])
    grid_bounds["yawmax"] = round(2*np.pi / planner_params["yaw_resolution"])
    grid_bounds["yawmin"] = round(0.0 / planner_params["yaw_resolution"])

    obstacle_x = (obstacle_x_idx + 0.5) * planner_params["xy_resolution"] + grid_bounds["xmin"] * planner_params["xy_resolution"]
    obstacle_y = (obstacle_y_idx + 0.5) * planner_params["xy_resolution"] + grid_bounds["ymin"] * planner_params["xy_resolution"]
    obstacle_kdtree = scipy.spatial.KDTree([[x, y] for x, y in zip(obstacle_x, obstacle_y)])

    return grid, grid_bounds, obstacle_kdtree

#### Collision Detection ####

def is_traj_valid(car_params, traj, obs_kdtree):

    for state in traj:
        x, y, yaw = state

        # check bubble first
        cx = x + car_params["wheel_base"]/2 * np.cos(yaw)
        cy = y + car_params["wheel_base"]/2 * np.sin(yaw)
        points_in_obstacle = obs_kdtree.query_ball_point([cx, cy], car_params["bubble_radius"])

        # skip past points not close to obstacles by just checking bubble
        if not points_in_obstacle:
            continue

        # check corners based on grid
        if rectangle_check(car_params, x, y, yaw,
                               [obs_kdtree.data[i][0] for i in points_in_obstacle], [obs_kdtree.data[i][1] for i in points_in_obstacle]):
            return False  # collision

    return True

def rectangle_check(car_params, x, y, yaw, ox, oy, eps=1e-8):
    # transform obstacles to base link frame
    rot = scipy.spatial.transform.Rotation.from_euler('z', yaw).as_matrix()[0:2, 0:2]
    for iox, ioy in zip(ox, oy):
        tx = iox - x
        ty = ioy - y
        converted_xy = np.stack([tx, ty]).T @ rot
        rx, ry = converted_xy[0], converted_xy[1]
        corners = get_corners(car_params, 0, 0, 0)[:-1]
        rx_max = corners[1,0]
        rx_min = corners[0,0]
        ry_max = corners[2,1]
        ry_min = corners[0,1]
        crit = (rx > rx_max or rx < rx_min or ry > ry_max or ry < ry_min)
        if not crit:
            return True # collision
        
    return False  # no collision

#### Holonomic Cost Map ####

def calculate_holonomic_cost_map(planner_params, goal_non_holonomic_node, grid, grid_bounds):

    """A* planner

    This function reduces the hybrid A* problem of searching across {x,y,yaw} space, to {x,y} space. This
    is *almost* equivalent to the traditional A* algorithm and its result is used as a heuristic to guide the 
    hybrid A* algorithm. The difference is that we search across the entire grid rather than finding a single path.
    This lets us compare our cost in the non holonomic search with the holonomic result on the fly using the
    precomputed holonomic cost map.

    Returns:
        np.ndarray: grid cost values according to A*
    """
    
    grid_index = (round(goal_non_holonomic_node["traj"][-1][0]/planner_params["xy_resolution"]) - grid_bounds["xmin"], 
                  round(goal_non_holonomic_node["traj"][-1][1]/planner_params["xy_resolution"]) - grid_bounds["ymin"])
    
    goal_holonomic_node = {
        "grid_index": grid_index,
        "cost": 0,
        "parent_index": grid_index
    }

    holonomic_motion_commands = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]

    def is_holonomic_node_valid(neighbour_node, grid):

        # check environment bounds
        if neighbour_node["grid_index"][0] < 0 or \
           neighbour_node["grid_index"][0] >= grid.shape[0] or \
           neighbour_node["grid_index"][1] < 0 or \
           neighbour_node["grid_index"][1] >= grid.shape[1]:
            return False
        
        # check no obstacle collisions - grid is 1 where obstacles are
        if grid[neighbour_node["grid_index"][0]][neighbour_node["grid_index"][1]]:
            return False
        
        return True
    
    # only tuples are hashable apparently - no lists 4 me >:(
    open_set = {goal_holonomic_node["grid_index"]: goal_holonomic_node}
    closed_set = {}

    priority_queue = []
    heapq.heappush(priority_queue, (goal_holonomic_node["cost"], goal_holonomic_node["grid_index"]))

    while True:
        if not open_set: break

        _, current_node_index = heapq.heappop(priority_queue)
        current_node = open_set[current_node_index]
        open_set.pop(current_node_index)
        closed_set[current_node_index] = current_node

        for action in holonomic_motion_commands:

            neighbour_holonomic_node = {
                "grid_index": (current_node["grid_index"][0] + action[0], \
                               current_node["grid_index"][1] + action[1]),
                "cost": current_node["cost"] + np.hypot(action[0], action[1]), # euclidean cost added
                "parent_index": current_node_index
            }

            if not is_holonomic_node_valid(neighbour_holonomic_node, grid): continue

            if neighbour_holonomic_node["grid_index"] not in closed_set:
                if neighbour_holonomic_node["grid_index"] in open_set:
                    if neighbour_holonomic_node["cost"] < open_set[neighbour_holonomic_node["grid_index"]]["cost"]:
                        open_set[neighbour_holonomic_node["grid_index"]]['cost'] = neighbour_holonomic_node["cost"]
                        open_set[neighbour_holonomic_node["grid_index"]]["parent_index"] = neighbour_holonomic_node["parent_index"]
                else:
                    open_set[neighbour_holonomic_node["grid_index"]] = neighbour_holonomic_node
                    heapq.heappush(priority_queue, (neighbour_holonomic_node['cost'], neighbour_holonomic_node["grid_index"]))
    
    holonomic_cost = np.ones_like(grid) * np.inf
    for nodes in closed_set.values():
        holonomic_cost[nodes["grid_index"][0]][nodes["grid_index"][1]] = nodes["cost"]

    return holonomic_cost

#### Functions for hybrid A* using RS ####

def reeds_shepp_node(planner_params, car_params, current_node, goal_node, obstacle_kdtree):

    start_x, start_y, start_yaw = current_node["traj"][-1][0], current_node["traj"][-1][1], current_node["traj"][-1][2]
    goal_x, goal_y, goal_yaw = goal_node["traj"][-1][0], goal_node["traj"][-1][1], goal_node["traj"][-1][2]

    # instantaneous radius of curvature for maximum steer
    radius = np.tan(car_params["max_steer"])/car_params["wheel_base"]

    # find all possible reeds-shepp paths between current and goal node
    rs_paths = calc_all_paths(start_x, start_y, start_yaw, goal_x, goal_y, goal_yaw, radius, planner_params["rs_step_size"])

    # Check if reedsSheppPaths is empty -> means we fall back to searching
    if not rs_paths:
        return None

    # Find path with lowest cost considering non-holonomic constraints
    cost_queue = heapdict()
    for path in rs_paths:
        cost_queue[path] = reeds_shepp_cost(planner_params, car_params, current_node, path)

    # Find first path in priority queue that is collision free
    while len(cost_queue)!=0:
        path = cost_queue.popitem()[0]
        traj=[]
        traj = [[path.x[k],path.y[k],path.yaw[k]] for k in range(len(path.x))]
        if is_traj_valid(car_params, traj, obstacle_kdtree):
            cost = reeds_shepp_cost(planner_params, car_params, current_node, path)
            node = {
                "grid_index": goal_node["grid_index"],
                "traj": traj,
                "cost": cost,
                "parent_index": current_node["grid_index"],
            }
            return node # Node(goalNode.gridIndex ,traj, None, None, cost, index(currentNode))


def reeds_shepp_cost(planner_params, car_params, current_node, path):
    cost = current_node["cost"]

    # distance cost
    for i in path.lengths:
        if i >= 0:
            cost += 1
        else:
            cost += abs(i) * planner_params["reverse_cost"]

    # direction change cost
    for i in range(len(path.lengths)-1):
        if path.lengths[i] * path.lengths[i+1] < 0:
            cost += planner_params["direction_change_cost"]
    
    # steering angle cost
    for i in path.ctypes:
        # chec types which are not straight lines
        if i!="S":
            cost += car_params['max_steer'] * planner_params["steer_angle_cost"]
    
    # Steering Angle change cost
    turnAngle=[0.0 for _ in range(len(path.ctypes))]
    for i in range(len(path.ctypes)):
        if path.ctypes[i] == "R":
            turnAngle[i] = - car_params["max_steer"]
        if path.ctypes[i] == "WB":
            turnAngle[i] = car_params["max_steer"]

    for i in range(len(path.lengths)-1):
        cost += abs(turnAngle[i+1] - turnAngle[i]) * planner_params["steer_angle_change_cost"]

    return cost


#### Kinematic Simulation ####

def _simulated_path_cost(planner_params, current_node, action):
    
    # prior node cost
    cost = current_node["cost"]

    # distance cost
    if action[1] > 0:
        cost += planner_params["kinematic_simulation_length"] * action[1]
    else:
        cost += planner_params["kinematic_simulation_length"] * action[1] * planner_params["reverse_cost"]

    # direction change cost
    if np.sign(current_node["direction"]) != np.sign(action[1]):
        cost += planner_params["direction_change_cost"]

    # steering angle cost
    cost += action[0] * planner_params["steer_angle_cost"]

    # steering angle change cost
    cost += np.abs(action[0] - current_node["steering_angle"]) * planner_params["steer_angle_change_cost"]

    return cost

def _is_valid(case_params, car_params, traj, obstacle_kdtree):

    # check if node is out of map bounds
    np_traj = np.array(traj)
    for state in np_traj:
        points = get_corners(car_params, state[0], state[1], state[2])
        for point in points:
            if point[0]<=case_params["xmin"] or point[0]>=case_params["xmax"] or \
            point[1]<=case_params["ymin"] or point[1]>=case_params["ymax"]:
                return False

    # Check if Node is colliding with an obstacle
    if not is_traj_valid(car_params, traj, obstacle_kdtree):
        return False
    return True

def kinematic_simulation_node(planner_params, case_params, car_params, current_node, action, obstacle_kdtree, grid_bounds):

    # Simulate node using given current Node and Motion Commands
    traj = []
    angle = pi_2_pi(current_node["traj"][-1][2] + action[1] * planner_params["kinematic_simulation_step"] / car_params["wheel_base"] * np.tan(action[0]))
    traj.append([current_node["traj"][-1][0] + action[1] * planner_params["kinematic_simulation_step"] * np.cos(angle),
                current_node["traj"][-1][1] + action[1] * planner_params["kinematic_simulation_step"] * np.sin(angle),
                pi_2_pi(angle + action[1] * planner_params["kinematic_simulation_step"] / car_params["wheel_base"] * np.tan(action[0]))])
    for i in range(int((planner_params["kinematic_simulation_length"]/planner_params["kinematic_simulation_step"]))-1):
        traj.append([traj[i][0] + action[1] * planner_params["kinematic_simulation_step"] * np.cos(traj[i][2]),
                    traj[i][1] + action[1] * planner_params["kinematic_simulation_step"] * np.sin(traj[i][2]),
                    pi_2_pi(traj[i][2] + action[1] * planner_params["kinematic_simulation_step"] / car_params["wheel_base"] * np.tan(action[0]))])
    
    # Find grid index
    grid_index = (round(traj[-1][0]/planner_params["xy_resolution"] - grid_bounds["xmin"]), \
                  round(traj[-1][1]/planner_params["xy_resolution"] - grid_bounds["ymin"]), \
                  round(traj[-1][2]/planner_params["yaw_resolution"] - grid_bounds["yawmin"]))

    if not _is_valid(case_params, car_params, traj, obstacle_kdtree):
        return None

    # Calculate Cost of the node
    cost = _simulated_path_cost(planner_params, current_node, action)

    return {
        "grid_index": grid_index,
        "traj": traj,
        "cost": cost,
        "direction": action[1],
        "steering_angle": action[0],
        "parent_index": current_node["grid_index"],
    }

#### Hybrid A* Planner ####

def plan(planner_params, case_params, car_params):

    # gets the grid which represents obstacles, the grid bounds which tells us the x,y,yaw limits of the grid in
    # cartesian space which lets us map the grid to real x,y positions. The obstacle_kdtree is a scipy.spatial.KDTree
    # formed of the obstacle true x,y positions in the space, which lets us quickly query ball collisions which
    # lets us accelerate the algorithms collision detection in sparsely constrained areas.
    grid, grid_bounds, obstacle_kdtree = calculate_obstacle_grid_and_kdtree(planner_params, case_params)

    start_grid_index = (round(case_params["x0"] /  planner_params["xy_resolution"]) - grid_bounds["xmin"], \
                        round(case_params["y0"] /  planner_params["xy_resolution"]) - grid_bounds["ymin"], \
                        round(case_params["yaw0"]/ planner_params["yaw_resolution"]) - grid_bounds["yawmin"])

    goal_grid_index = (round(case_params["xf"] /  planner_params["xy_resolution"]) - grid_bounds["xmin"], \
                       round(case_params["yf"] /  planner_params["xy_resolution"]) - grid_bounds["ymin"], \
                       round(case_params["yawf"]/ planner_params["yaw_resolution"]) - grid_bounds["yawmin"])

    start_node = {
        "grid_index": start_grid_index,
        "traj": [[case_params["x0"], case_params["y0"], case_params["yaw0"]]],
        "cost": 0.0,
        "direction": 0.0,
        "steering_angle": 0.0,
        "parent_index": start_grid_index,
    }

    goal_node = {
        "grid_index": goal_grid_index,
        "traj": [[case_params["xf"], case_params["yf"], case_params["yawf"]]],
        "cost": 0.0,
        "direction": 0.0,
        "steering_angle": 0.0,
        "parent_index": goal_grid_index,
    }

    # Motion commands for a Non-Holonomic Robot like a Car or Bicycle (Trajectories using Steer Angle and Direction)
    action_space = []
    for i in np.linspace(-car_params["max_steer"], car_params["max_steer"], planner_params["steer_options"]):
        for j in np.linspace(-planner_params["max_movement"], planner_params["max_movement"], planner_params["movement_options"]):
            action_space.append([i,j])
    action_space = np.vstack(action_space)

    # calculate the costmap for the A* solution to the environment, which we guide the non-holonomic search with
    holonomic_cost_map = calculate_holonomic_cost_map(planner_params, goal_node, grid, grid_bounds)

    # Add start node to open Set
    open_set = {start_node["grid_index"]: start_node}
    closed_set = {}

    # Create a priority queue for acquiring nodes based on their cost's
    cost_queue = heapdict()

    # Add start mode into priority queue
    cost_queue[start_node["grid_index"]] = max(
        start_node["cost"], 
        planner_params["hybrid_cost"] * holonomic_cost_map[start_node["grid_index"][0]][start_node["grid_index"][1]]
    )
    counter = 0

    # Run loop while path is found or open set is empty
    while True:

        counter += 1
        print(counter)
        
        # if empty open set then no solution available
        if not open_set: 
            return None

        # bookkeeping
        current_node_index = cost_queue.popitem()[0]
        current_node = open_set[current_node_index]
        open_set.pop(current_node_index)
        closed_set[current_node_index] = current_node

        # is the reeds shepp solution collision free?
        rs_node = reeds_shepp_node(planner_params, car_params, current_node, goal_node, obstacle_kdtree)

        # if reeds shepp trajectory exists then we store solution and break the loop
        if rs_node: closed_set[rs_node["grid_index"]] = rs_node; break

        # edge case of directly finding the solution without reeds shepp, break loop
        if current_node_index == goal_node["grid_index"]: print("path found"); break

        # get all simulated nodes from the current node
        for action in action_space:
            simulated_node = kinematic_simulation_node(planner_params, case_params, car_params, current_node, action, obstacle_kdtree, grid_bounds)

            # check if path is valid
            if not simulated_node: 
                continue

            # Draw Simulated Node
            x,y,z =zip(*simulated_node["traj"])
            plt.plot(x, y, linewidth=0.3, color='g')

            # Check if simulated node is already in closed set
            if simulated_node["grid_index"] not in closed_set: 

                # Check if simulated node is already in open set, if not add it open set as well as in priority queue
                if simulated_node["grid_index"] not in open_set:
                    open_set[simulated_node["grid_index"]] = simulated_node
                    cost_queue[simulated_node["grid_index"]] = max(simulated_node["cost"], planner_params["hybrid_cost"] * holonomic_cost_map[simulated_node["grid_index"][0]][simulated_node["grid_index"][1]])
                else:
                    if simulated_node["cost"] < open_set[simulated_node["grid_index"]]["cost"]:
                        open_set[simulated_node["grid_index"]] = simulated_node
                        cost_queue[simulated_node["grid_index"]] = max(simulated_node["cost"], planner_params["hybrid_cost"] * holonomic_cost_map[simulated_node["grid_index"][0]][simulated_node["grid_index"][1]])


    # extract trajectory
    def backtrack(start_node, goal_node, closed_set):

        # Goal Node data
        current_node_index = goal_node["parent_index"]
        current_node = closed_set[current_node_index]
        x=[]
        y=[]
        yaw=[]

        # Iterate till we reach start node from goal node
        while current_node_index != start_node["grid_index"]:
            a, b, c = zip(*current_node["traj"])
            x += a[::-1] 
            y += b[::-1] 
            yaw += c[::-1]
            current_node_index = current_node["parent_index"]
            current_node = closed_set[current_node_index]

        traj = np.array([x[::-1], y[::-1], yaw[::-1]]).T
        return traj

    traj = backtrack(start_node, goal_node, closed_set)

    return traj

#### Main runner ####

def main(case_params, save_name=None):

    # plots the true case so that we can see where the non discretized obstacles and start/end spots are
    plot_case(case_params, car_params, show=False, save=False, bare=False)

    # actually plan the trajectory
    traj = plan(planner_params, case_params, car_params)

    #### A lot of plotting to demonstrate whats happening ####

    # check out the data yourself!
    grid, grid_bounds, _ = calculate_obstacle_grid_and_kdtree(planner_params, case_params)

    # this plots the grid with its obstacles
    plt.imshow(grid.T, cmap='cividis_r', origin='lower', extent=(case_params["xmin"], case_params["xmax"], case_params["ymin"], case_params["ymax"]))

    # some data to plot the holonomic cost map
    start_grid_index = (round(case_params["x0"] /  planner_params["xy_resolution"]) - grid_bounds["xmin"], \
                        round(case_params["y0"] /  planner_params["xy_resolution"]) - grid_bounds["ymin"], \
                        round(case_params["yaw0"] / planner_params["yaw_resolution"]) - grid_bounds["yawmin"])
    goal_grid_index = (round(case_params["xf"] /  planner_params["xy_resolution"]) - grid_bounds["xmin"], \
                       round(case_params["yf"] /  planner_params["xy_resolution"]) - grid_bounds["ymin"], \
                       round(case_params["yawf"] / planner_params["yaw_resolution"]) - grid_bounds["yawmin"])
    goal_node = {
        "grid_index": goal_grid_index,
        "traj": [[case_params["xf"], case_params["yf"], case_params["yawf"]]],
        "cost": 0.0,
        "parent_index": goal_grid_index,
    }
    # double check that the holonomic heuristic starts at end and goes from there.
    holonomic_cost_map = calculate_holonomic_cost_map(planner_params, goal_node, grid, grid_bounds)
    assert holonomic_cost_map[goal_grid_index[0], goal_grid_index[1]] == 0.0
    plt.imshow(holonomic_cost_map.T, cmap='gray', origin='lower', extent=(case_params["xmin"], case_params["xmax"], case_params["ymin"], case_params["ymax"]))
    cbar = plt.colorbar()
    cbar.set_label('Holonomic Cost', rotation=270, labelpad=15)

    # plot the actual corners of the car throughout the trajectory
    plt.plot(traj[:,0], traj[:,1])
    for state in traj:
        corners = get_corners(car_params, state[0], state[1], state[2])
        plt.plot(corners[:,0], corners[:,1], 'magenta')

    # plot the start and end positions of center of mass of car in grid space
    plt.scatter([(goal_grid_index[0] + grid_bounds["xmin"]) * planner_params["xy_resolution"]],
                [(goal_grid_index[1] + grid_bounds["ymin"]) * planner_params["xy_resolution"]], linewidths=1.0, color="red")
    plt.scatter([(start_grid_index[0] + grid_bounds["xmin"]) * planner_params["xy_resolution"]],
                [(start_grid_index[1] + grid_bounds["ymin"]) * planner_params["xy_resolution"]], linewidths=1.0, color="green")
    
    # saving and closing...
    if save_name is None:
        plt.savefig('output.png', dpi=500)
    else:
        plt.savefig(save_name, dpi=500)
    plt.close()

if __name__ == "__main__":

    from utils.tpcap import read

    # do the manual reverse park example
    scenario_name = "reverse_park"
    case_params = read(f"data/manual_cases/{scenario_name}.csv")
    main(case_params, save_name=f"data/output/{scenario_name}")

    # this will run the TPCAP benchmark itself
    case_num = 1
    for case_num in range(1,21):
        scenario_name = f"Case{case_num}"
        case_params = read(f"data/tpcap_cases/{scenario_name}.csv")
        main(case_params, save_name=f"data/output/{scenario_name}")

    print('fin')