from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
import random
import time

import PIL.Image as PIL_Image

def save_np_image(img_np, file_name = "test.jpg"):
    max_val = np.max(img_np)
    if(max_val < 2.0 ):
        img_np = img_np*255.0
    image_np = (img_np).astype(np.uint8)
    im = PIL_Image.fromarray(image_np)        
    im = im.save(file_name)


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
sim_params.dt = 1.0 / 10.0
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

# create sim
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

asset_root = "../../assets"

# create box asset

asset_root = "../../assets"
asset_file_obj0 = "urdf/ycb/red_cube/red_cube.urdf"
obj0_asset = gym.load_asset(sim, asset_root, asset_file_obj0, gymapi.AssetOptions())

asset_root = "../../assets"
asset_file_obj1 = "urdf/ycb/red_cube/red_cube.urdf"
obj1_asset = gym.load_asset(sim, asset_root, asset_file_obj1, gymapi.AssetOptions())

# configure env grid
num_envs = 1
num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)

box_pose = gymapi.Transform()

passive_obj_pose = gymapi.Transform()

env = None
envs = []
box_idxs = []
passive_obj_idxs = []

hand_idxs = []
init_pos_list = []
init_rot_list = []

# # add ground plane
# plane_params = gymapi.PlaneParams()
# plane_params.normal = gymapi.Vec3(0, 0, 1)
# gym.add_ground(sim, plane_params)


obj0_scale = 0.013164701807471129
obj1_scale = 0.013164701807471129

obj0_handle = None
obj1_handle = None

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    #change to load from yaml file
    obj0_pose.p.x = 0.30
    obj0_pose.p.y = 0.0
    obj0_pose.p.z = 0.1
    obj0_pose.r = gymapi.Quat()
    obj0_pose.r.x = 0.0
    obj0_pose.r.y = 0.0 
    obj0_pose.r.z = 0.0
    obj0_pose.r.w = 1.0
    obj0_scale = 1.0

    #change to load from yaml file
    obj1_pose.p.x = 0.30
    obj1_pose.p.y = 0.0
    obj1_pose.p.z = 0.1
    obj1_pose.r = gymapi.Quat()
    obj1_pose.r.x = 0.0
    obj1_pose.r.y = 0.0 
    obj1_pose.r.z = 0.0
    obj1_pose.r.w = 1.0
    obj1_scale = 1.0

    obj0_handle = gym.create_actor(env, obj0_asset, obj0_pose, "obj0", i, 0)
    gym.set_actor_scale(env, obj0_handle, obj0_scale)
    # get global index of box in rigid body state tensor
    obj0_idx = gym.get_actor_rigid_body_index(env, obj0_handle, 0, gymapi.DOMAIN_SIM)
    obj0_idxs.append(obj0_idx)
    gym.set_rigid_body_segmentation_id(env, obj0_handle, 0, 1)


    obj1_handle = gym.create_actor(env, obj1_asset, obj1_pose, "obj1", i, 0)
    gym.set_actor_scale(env, obj1_handle, obj1_scale)
    # get global index of box in rigid body state tensor
    obj1_idx = gym.get_actor_rigid_body_index(env, obj1_handle, 0, gymapi.DOMAIN_SIM)
    obj1_idxs.append(obj_idx)
    gym.set_rigid_body_segmentation_id(env, obj1_handle, 0, 2)


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

camera_handles = [[]]
for i in range(num_envs):
    camera_handles.append([])
    
    camera_properties = compute_camera_peoperty(img_size, img_size, resized_intrinsic_np)

    # Set a fixed position and look-target for the first camera
    # position and target location are in the coordinate frame of the environment
    h1 = gym.create_camera_sensor(envs[i], camera_properties)
    camera_position = gymapi.Vec3(-0.13913296, 0.053, 0.43643044)
    camera_target = gymapi.Vec3(0.62799622, 0.00756501, -0.2034511)
    gym.set_camera_location(h1, envs[i], camera_position, camera_target)
    camera_handles[i].append(h1)

while not gym.query_viewer_has_closed(viewer):
    if(time_idx > 40):
        break

    time_idx += 1
    if(time_idx >= 30 and time_idx <=40 ):

        gym.set_rigid_transform(
            envs[0],
            gym.get_rigid_handle(
                envs[0], 
                "box", 
                gym.get_actor_rigid_body_names(envs[0], obj1_handle)[0] 
            ),                    
            traj_pose
        )

        # rgb_filename = "rgb_env%d_cam0_frame%d.png" % (0, time_idx)
        # gym.write_camera_image_to_file(sim, envs[0], camera_handles[0][0], gymapi.IMAGE_COLOR, rgb_filename)

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
    # render the camera sensors
    gym.render_all_camera_sensors(sim)

    gym.sync_frame_time(sim)

    if(time_idx >= 30 and time_idx <=40 ):

        rgb_filename = "rgb_env%d_cam0_frame%d.png" % (0, time_idx)
        gym.write_camera_image_to_file(sim, envs[0], camera_handles[0][0], gymapi.IMAGE_COLOR, rgb_filename)

        segmentation = gym.get_camera_image(
            sim, envs[0], camera_handles[0][0], gymapi.IMAGE_SEGMENTATION,
        )
        print("seg: ", segmentation.shape)
        print("idx: ", np.max(segmentation))

        # depth = gym.get_camera_image(
        #     sim, envs[0], camera_handles[0][0], gymapi.IMAGE_DEPTH,
        # )
        # print("depth: ", depth.shape)
        # print("max: ", np.max(depth))
        # rgba = gym.get_camera_image(
        #     sim, envs[0], camera_handles[0][0], gymapi.IMAGE_COLOR,
        # ).reshape(512, 512, 4)
        # rgb = rgba[:,:,0:3]

# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
