"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Body physics properties example
-------------------------------
An example that demonstrates how to load rigid body, update its properties
and apply various actions. Specifically, there are three scenarios that
presents the following:
- Load rigid body asset with varying properties
- Modify body shape properties
- Modify body visual properties
- Apply body force
- Apply body linear velocity
"""
import numpy as np
from isaacgym import gymutil
from isaacgym import gymapi
from math import sqrt
import math
from PIL import Image as im

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Body Physics Properties Example")

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
# sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
sim_params.up_axis = gymapi.UP_AXIS_Z
if args.physics_engine == gymapi.SIM_FLEX:
    pass
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu


sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')



# set up the env grid
num_envs = 1
spacing = 1.8
env_lower = gymapi.Vec3(-spacing, -spacing, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# create list to mantain environment and asset handles
envs = []
table_handles = []
actor_handles = []

# create table asset with gravity disabled
asset_root = "../../assets"
asset_file = "urdf/ycb/table_top/table_top.urdf"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
# asset_table = gym.load_asset(sim, asset_root, asset_file, asset_options)

# create static box asset
asset_options.fix_base_link = True
asset_box = gym.create_box(sim, 1.0, 1.0, 0.03, asset_options)


# load ball asset
asset_root = "../../assets"
# asset_file_object = "urdf/ycb/red_cube/red_cube.urdf"
asset_file_object = "urdf/ycb/red_mug/red_mug.urdf"

asset_object = gym.load_asset(sim, asset_root, asset_file_object, gymapi.AssetOptions())


color = gymapi.Vec3(0, 1, 0)
pose = gymapi.Transform()
pose.r = gymapi.Quat(0, 0, 0, 1)


# mesh scale:  16.103548483729227
# transform:  [[   0.97441594    0.22451232    0.01038267  452.03628145]
#  [  -0.22332016    0.96196666    0.15731573 -290.96597253]
#  [   0.02533154   -0.15560962    0.98749378 -461.58481256]
#  [   0.            0.            0.            1.        ]]

object_scale = 1. # / 161.03548483729227
print('Creating %d environments' % num_envs)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, 1)
    envs.append(env)

    # add table actor
    # pose = gymapi.Transform()
    # pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
    # pose.r = gymapi.Quat(0, 0, 0, 1)
    # table_handle = gym.create_actor(env, asset_table, pose, "table_top", i, 0)
    # actor_handles.append(table_handle)

    # add box actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
    pose.r = gymapi.Quat(0, 0, 0, 1)
    table_handle = gym.create_actor(env, asset_box, pose, "table_top", i, 0)
    actor_handles.append(table_handle)
    shape_props = gym.get_actor_rigid_shape_properties(env, table_handle)
    shape_props[0].restitution = 1
    shape_props[0].compliance = 0.5
    gym.set_actor_rigid_shape_properties(env, table_handle, shape_props)

    # mug
    pose.p = gymapi.Vec3(0., 0.0, 0.15)
    object_handle = gym.create_actor(env, asset_object, pose, "object", i, 0)
    actor_handles.append(object_handle)
    shape_props = gym.get_actor_rigid_shape_properties(env, object_handle)
    shape_props[0].restitution = 1
    shape_props[0].compliance = 0.5
    gym.set_actor_rigid_shape_properties(env, object_handle, shape_props)
    gym.set_actor_scale(env, object_handle, object_scale)

def compute_camera_intrinsics_matrix(image_width, image_heigth, horizontal_fov, device):
    vertical_fov = (image_heigth / image_width * horizontal_fov) * np.pi / 180
    horizontal_fov *= np.pi / 180

    f_x = (image_width / 2.0) / np.tan(horizontal_fov / 2.0)
    f_y = (image_heigth / 2.0) / np.tan(vertical_fov / 2.0)

    K = torch.tensor([[f_x, 0.0, image_width / 2.0], [0.0, f_y, image_heigth / 2.0], [0.0, 0.0, 1.0]], device=device, dtype=torch.float)

    return K

def compute_camera_peoperty(image_width, image_height, intrinsic):

    horizontal_fov = 2 * math.atan(image_width/ (2 * intrinsic[0][0]) ) * 180.0 / np.pi
    vertical_fov = 2 * math.atan(image_height/ (2 * intrinsic[1][1]) ) * 180.0 / np.pi

    camera_properties = gymapi.CameraProperties()
    camera_properties.width = image_width
    camera_properties.height = image_height

    camera_properties.horizontal_fov = horizontal_fov
    # camera_properties.vertical_fov = vertical_fov

    return camera_properties

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

# look at the first env
cam_pos = gymapi.Vec3(-0.13913296, 0.053, 0.43643044)
# cam_target = gymapi.Vec3(0.5, 0.053, 0.0)
cam_target = gymapi.Vec3(0.62799622, 0.00756501, -0.2034511)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)



# camera setting
frame_count = 0
# show axis
axes_geom = gymutil.AxesGeometry(1.0)
while not gym.query_viewer_has_closed(viewer):

    goal_quat = np.array([0., 0., 0., 1.0])
    goal_viz_T = gymapi.Transform(r=gymapi.Quat(*goal_quat))
    gymutil.draw_lines(axes_geom, gym, viewer, env, goal_viz_T)

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # render the camera sensors
    gym.render_all_camera_sensors(sim)

    if(frame_count > 3):
        continue

    for i in range(num_envs):
        # The gym utility to write images to disk is recommended only for RGB images.
        rgb_filename = "graphics_images/rgb_env%d_cam0_frame%d.png" % (i, frame_count)
        gym.write_camera_image_to_file(sim, envs[i], camera_handles[i][0], gymapi.IMAGE_COLOR, rgb_filename)

        # Retrieve image data directly. Use this for Depth, Segmentation, and Optical Flow images
        # Here we retrieve a depth image, normalize it to be visible in an
        # output image and then write it to disk using Pillow
        depth_image = gym.get_camera_image(sim, envs[i], camera_handles[i][0], gymapi.IMAGE_DEPTH)

        # -inf implies no depth value, set it to zero. output will be black.
        depth_image[depth_image == -np.inf] = 0

        # clamp depth image to 10 meters to make output image human friendly
        depth_image[depth_image < -10] = -10

        # flip the direction so near-objects are light and far objects are dark
        normalized_depth = -255.0*(depth_image/np.min(depth_image + 1e-4))

        # Convert to a pillow image and write it to disk
        normalized_depth_image = im.fromarray(normalized_depth.astype(np.uint8), mode="L")
        normalized_depth_image.save("graphics_images/depth_env%d_cam_frame%d.jpg" % (i, frame_count))

    frame_count = frame_count + 1

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
