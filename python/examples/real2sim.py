"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Franka Cube Pick
----------------
Use Jacobian matrix and inverse kinematics control of Franka robot to pick up a box.
Damped Least Squares method from: https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
"""

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
import random
import time




def compute_camera_peoperty(image_width, image_height, intrinsic):

    horizontal_fov = 2 * math.atan(image_width/ (2 * intrinsic[0][0]) ) * 180.0 / np.pi
    vertical_fov = 2 * math.atan(image_height/ (2 * intrinsic[1][1]) ) * 180.0 / np.pi

    camera_properties = gymapi.CameraProperties()
    camera_properties.width = image_width
    camera_properties.height = image_height

    camera_properties.horizontal_fov = horizontal_fov
    # camera_properties.vertical_fov = vertical_fov

    return camera_properties


# set random seed
np.random.seed(42)

# acquire gym interface
gym = gymapi.acquire_gym()

# parse arguments

# parse arguments
args = gymutil.parse_arguments(description="Body Physics Properties Example")

# configure sim
sim_params = gymapi.SimParams()
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.relaxation = 0.9
    sim_params.flex.dynamic_friction = 0.0
    sim_params.flex.static_friction = 0.0
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")
# set torch device
device = 'cpu'

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = False
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

# Set controller parameters
# IK params
damping = 0.05

# OSC params
kp = 150.
kd = 2.0 * np.sqrt(kp)
kp_null = 10.
kd_null = 2.0 * np.sqrt(kp_null)

# create sim
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

asset_root = "../../assets"

# create table asset
table_dims = gymapi.Vec3(0.7, 1.0, 0.02)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

# create box asset
box_size = 0.045
asset_options = gymapi.AssetOptions()
box_asset = gym.create_box(sim, box_size, box_size, box_size, asset_options)

asset_root = "../../assets"
asset_file_object = "urdf/ycb/red_cube/red_cube.urdf"
# asset_file_object = "urdf/ycb/010_potted_meat_can/010_potted_meat_can.urdf"
object_asset = gym.load_asset(sim, asset_root, asset_file_object, gymapi.AssetOptions())


# configure env grid
num_envs = 1
num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)

franka_pose = gymapi.Transform()
franka_pose.p = gymapi.Vec3(0, 0, 0)

table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.35, 0.0, 0.5 * table_dims.z)

box_pose = gymapi.Transform()

passive_obj_pose = gymapi.Transform()

envs = []
box_idxs = []
passive_obj_idxs = []

hand_idxs = []
init_pos_list = []
init_rot_list = []

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)


# [ 0.99706722  0.07580188  0.01053763  0.30986352]
#  [-0.07639295  0.99405243  0.07761369 -0.23455131]
#  [-0.0045917  -0.07819107  0.99692782 -0.02332963]
# object_scale = 1.0
object_scale = 0.013164701807471129

box_handle = None
passive_obj_handle = None

# object_scale = 1. / 161.03548483729227


for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add table
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)
    
    #change to load from yaml file
    box_pose.p.x = 0.30986352
    box_pose.p.y = -0.23455131
    box_pose.p.z = 0.1
    box_pose.r = gymapi.Quat()

    # change to load from yaml file
    box_pose.r.x = -0.02543199
    box_pose.r.y = -0.00559756 
    box_pose.r.z = -0.01357897
    box_pose.r.w =   0.99956865


    #change to load from yaml file
    passive_obj_pose.p.x = 0.30
    passive_obj_pose.p.y = 0.0
    passive_obj_pose.p.z = 0.1
    passive_obj_pose.r = gymapi.Quat()

    # # change to load from yaml file
    passive_obj_pose.r.x = 0.0
    passive_obj_pose.r.y = 0.0 
    passive_obj_pose.r.z = 0.0
    passive_obj_pose.r.w = 1.0


    # box_handle = gym.create_actor(env, box_asset, box_pose, "box", i, 0)
    box_handle = gym.create_actor(env, object_asset, box_pose, "box", i, 0)
    gym.set_actor_scale(env, box_handle, object_scale)
    # get global index of box in rigid body state tensor
    box_idx = gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
    box_idxs.append(box_idx)

    passive_obj_handle = gym.create_actor(env, object_asset, passive_obj_pose, "box2", i, 0)
    gym.set_actor_scale(env, passive_obj_handle, object_scale)
    # get global index of box in rigid body state tensor
    passive_obj_idx = gym.get_actor_rigid_body_index(env, passive_obj_handle, 0, gymapi.DOMAIN_SIM)
    passive_obj_idxs.append(passive_obj_idx)


# point camera at middle env
# cam_pos = gymapi.Vec3(4, 3, 2)
# cam_target = gymapi.Vec3(-4, -3, 0)



# add camera
img_size = 512
resized_img_size = (img_size,img_size)
original_image_size = (1080, 1920) #(h,)
# resized_intrinsic = o3d.camera.PinholeCameraIntrinsic( 256., 25, 80., 734.1779174804688*scale_y, 993.6226806640625*scale_x, 551.8895874023438*scale_y)
fxfy = float(img_size)

resized_intrinsic_np = np.array([
    [fxfy, 0., img_size/2],
    [0. ,fxfy, img_size/2],
    [0., 0., 1.0]
])
camera_properties = compute_camera_peoperty(img_size, img_size, resized_intrinsic_np)
cam_pos = gymapi.Vec3(-0.13913296, 0.053, 0.43643044)
cam_target = gymapi.Vec3(0.62799622, 0.00756501, -0.2034511)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# ==== prepare tensors =====
# from now on, we will use the tensor API that can run on CPU or GPU
gym.prepare_sim(sim)



time_idx = 0
traj = []
traj_pose = gymapi.Transform()

#change to load from yaml file
traj_pose.p.x = 0.30986352
traj_pose.p.y = 0.0
traj_pose.p.z = 0.2
traj_pose.r = gymapi.Quat()

# change to load from yaml file
traj_pose.r.x = -0.02543199
traj_pose.r.y = -0.00559756 
traj_pose.r.z = -0.01357897
traj_pose.r.w =   0.99956865
# simulation loop
while not gym.query_viewer_has_closed(viewer):


    time_idx += 1
    if(time_idx >= 30 and time_idx <=40 ):

        gym.set_rigid_transform(
            envs[0],
            gym.get_rigid_handle(
                envs[0], 
                "box", 
                gym.get_actor_rigid_body_names(envs[0], box_handle)[0] 
            ),                    
            traj_pose
        )
        # gym.set_rigid_linear_velocity(
        #     envs[0],
        #     gym.get_rigid_handle(
        #     envs[0], 
        #     "box", 
        #     gym.get_actor_rigid_body_names(envs[0], box_handle)[0] 
        #     ),
        #     gymapi.Vec3(0., 0., 0.01)
        # )
    #                             )
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # refresh tensors
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)

    # update viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
