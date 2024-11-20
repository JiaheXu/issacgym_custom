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

from isaacgym import gymutil
from isaacgym import gymapi

# initialize gym
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

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.static_friction = 0.0
plane_params.dynamic_friction = 0.0

gym.add_ground(sim, plane_params)

# set up the env grid
num_envs = 1
spacing = 1.8
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# create list to mantain environment and asset handles
envs = []
box_handles = []
actor_handles = []

# create box assets w/ varying densities (measured in kg/m^3)
box_size = 0.2
box_densities = [8., 32., 1024.]
for dx in range(3):
    # AssetOptions enables loading assets with different properties
    # Properties include density, angularDamping, maxAngularVelocity, linearDamping, maxLinearVelocity etc.
    asset_options = gymapi.AssetOptions()
    asset_options.density = box_densities[dx]
    asset_box = gym.create_box(sim, box_size, box_size, box_size, asset_options)
    box_handles.append(asset_box)

# create capsule asset
asset_options = gymapi.AssetOptions()
asset_options.density = 100.
asset_capsule = gym.create_capsule(sim, 0.2, 0.2, asset_options)

# create ball asset with gravity disabled
asset_root = "../../assets"
asset_file = "urdf/ycb/rigid_body/rigid_body.urdf"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
# asset_options.disable_gravity = True
print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
asset_table = gym.load_asset(sim, asset_root, asset_file, asset_options)
# asset_box = gym.load_asset(sim, asset_root, asset_file, asset_options)

# create static box asset
# asset_options.fix_base_link = True
# asset_box = gym.create_box(sim, 0.5, 0.1, 0.5, asset_options)

print('Creating %d environments' % num_envs)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, 1)
    envs.append(env)


    # add box actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.5, 0.0)
    pose.r = gymapi.Quat(0, 0, 0, 1)
    box_handle = gym.create_actor(env, asset_table, pose, "actor1", i, 0)
    actor_handles.append(box_handle)

    # set restitution for box actor
    shape_props = gym.get_actor_rigid_shape_properties(env, box_handle)
    shape_props[0].restitution = 1
    shape_props[0].compliance = 0.5
    gym.set_actor_rigid_shape_properties(env, box_handle, shape_props)




    # # add capsule actor
    # pose.p = gymapi.Vec3(0.0, 2.0, 0.0)
    # capsule_handle = gym.create_actor(env, asset_capsule, pose, "actor2", i, 0)

    # # set restitution for capsule actor
    # shape_props = gym.get_actor_rigid_shape_properties(env, capsule_handle)
    # shape_props[0].restitution = 1
    # shape_props[0].compliance = 0.5
    # gym.set_actor_rigid_shape_properties(env, capsule_handle, shape_props)

# look at the first env
cam_pos = gymapi.Vec3(6, 4.5, 3)
cam_target = gymapi.Vec3(-0.8, 0.5, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
