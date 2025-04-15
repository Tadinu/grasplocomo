# INTRO
**Note**: This guide is for only antipodal grasp pose generation.

`GraspLoCoMo` is a grasp pose detection library to help generate antipodal grasps from a point cloud, be it representing
a single object or a cluttered scene of various objects in the environment.

https://ieeexplore.ieee.org/document/8594226<br>
https://arxiv.org/abs/2107.12492

The output grasps can be then used directly as inputs to motion planning in a picking task, or to generate grasp dataset
for an AI-based grasp detection pipeline.

> Disclaimer: GraspLoCoMo is picked up mostly as an introductory example with criteria: ease of compiling/use, performance, grasping scoring.
Of all analytical approaches, there are very few published implementations. Others:<br>
https://github.com/atenpas<br>
https://github.com/mkiatos/geometric-object-grasper<br>
https://www.mdpi.com/2075-1702/13/1/12

In the last decade of grasp detection literature, the research community is geared more toward AI-based methods, which typically have performance advantages especially in cluttered scene, though requiring dataset preparation (either from automatically generated data using methods like GraspLoCoMo or manually crafted) & training. However, analytical methods that are focused on detecting grasps based on geometric analysis of object mesh/point clouds are also favored due to its concrete explainability, especially in the industry, where safety and robustness are always critical factors.

References:

https://manipulation.csail.mit.edu/clutter.html - https://manipulation.csail.mit.edu/pick.html<br>
[An Overview of 3D Object Grasp Synthesis Algorithms](https://hal.science/hal-00731127)<br>
https://rhys-newbury.github.io/projects/6dof<br>
https://paperswithcode.com/task/robotic-grasping<br>
https://github.com/rhett-chen/Robotic-grasping-papers<br>
https://geo-match.github.io<br>
https://contactdb.cc.gatech.edu/contactgrasp.html<br>
https://graspit-simulator.github.io<br>
https://graspnet.net<br>
https://github.com/graspnet/anygrasp_sdk<br>

...and much more

Feel free to pick anyone of those (or any other you find) that suit your interests.

*Note*: In practical scenarios likes picking tasks in the factory/warehouse, no matter what method producing grasps, at the
end of the pipeline, they all need to go through a robust collision detection check to output the final safe colliding-free ones.

-> It's included in this guide to suggest a MuJoCo-based collision check, which is both performant (even not using GPU) and accurate.

[//]: # (============)
# GRIPPER MODELS
(A) Create gripper mjcf/xml to be used in MuJoCo

MJCF/XML:
https://github.com/google-deepmind/mujoco_menagerie

URDF:
https://github.com/Daniella1/urdf_files_dataset/tree/main/urdf_files/ros-industrial/xacro_generated/robotiq

Download a specific dir from github without cloning all:
https://download-directory.github.io

URDF->MJCF: https://github.com/FFTAI/Wiki-MJCF
1. Edit .urdf: For CAD paths -> make it starts directly from "meshes/"
2. Move .urdf to be the same level with *meshes* dir
3. Run `urdf2mjcf <urdf_path> <mjcf_path>`

[//]: # (============)
# MESH -> POINT CLOUD

**(B) Generate object point cloud in txt (pos + normal, as inputs to GraspLoCoMo)**<br>
**(B.1) Open3D python**<br>
https://github.com/isl-org/Open3D/blob/main/examples/python/geometry/triangle_mesh_sampling.py<br>
Save to .pcd then just edit extension to .txt (Note: only output pos + normals)

**Visualize**<br>
https://www.open3d.org/docs/latest/tutorial/geometry/pointcloud.html
https://www.open3d.org/docs/latest/tutorial/Basic/visualization.html

**(B.2) PCL (installation + usage are abit more complicated)**
https://github.com/PointCloudLibrary/pcl<br>
```
cd release/bin
./pcl_ply2pcd [-format 0|1] input.ply output.pcd
```

ply2txt<br>
`./pcl_ply2raw input.ply output.txt`

pcd2txt: using pcl-cpp<br>
https://github.com/cristianrubioa/pcd2txt<br>
https://github.com/Jornsd/Modelfreeobjectgrasping/blob/main/pcd2txt/src/pcd2txt.cpp<br>
Or python<br>
https://github.com/MapIV/pypcd4<br>
https://github.com/dimatura/pypcd<br>

pcd viewer
`./pcl_viewer input.pcd`

[//]: # (===========================)
# POINT CLOUD -> GRASP POSE DETECTION
**(C) NOTES on the original GraspLoCoMo repo**<br> https://github.com/maximeadjigble/grasplocomo
> - The grasps as well as pre-provided clouds are outputted in the global frame. Those clouds are also not centered around {0,0,0}.
-> Output grasps are NOT relative to object point cloud!<br>
> - Those clouds themselves are sampled directly from a camera, without accompanying original meshes.<br>
-> NOT DIRECTLY USABLE for ones that want to output grasps as relative to a target mesh-object.<br>
> - Example demo is only for SchunkPG70 Gripper<br>
> - Output grasp pose is in clunky matrix instead of pos + quat format

Fork of GraspLoCoMo: https://github.com/Tadinu/grasplocomo/tree/dev
- Configs for Robotiq2F85
- Output grasp pose as pos + quat format

[//]: # (===========================)
# GRASP POSE GENERATION PROCEDURE
*Note: <obj_type>: typename of object, eg: mug, hammer, etc.*

(1) Start with Obj mesh -> Use p3d to sample point cloud -> `grasplocomo/Clouds/<obj_type>_cloud.txt`

(2) Use GraspLoCoMo -> generate original grasps (as in global coord) -> `<obj_type>_grasp_results.txt`, as `GRASPS_ORIGINAL_FILEPATH`

`./run_grasp_gen.sh <obj_type>`

**Use a sim framework eg. MuJoCo to:**

(3.1) Spawn obj composed of pointcloud as voxels from (1)

(3.2) Spawn gripper with original grasps generated from (2)

(4) Set `GRASPS_FILE_PATH = GRASPS_ORIGINAL_FILEPATH`

Log out local grasps of gripper (3.2) relative to object (3.1) -> `<obj_type>_grasp_results_recalculated.txt`, as `GRASPS_RECALCULATED_FILEPATH`

(5) Set `GRASPS_FILE_PATH = GRASPS_RECALCULATED_FILEPATH`

(6) Load grasps from `GRASPS_FILE_PATH` (as in (5)) and run a rollout on collision checking in a your current scene (either in cluttered, or assembly context), wherever the target to-be-grasped object is.
https://mujoco.readthedocs.io/en/stable/python.html#pyrollout<br>
https://github.com/google-deepmind/mujoco/blob/main/python/rollout.ipynb<br>
https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/rollout_test.py

**Collision check**: Use `mjSENS_GEOMDIST` sensor which reports distance between any pair of geoms or geom and a body in MuJoCo,
which is accuracy-wise recommended instead of `mj_geomDistance` as in observation.<br>
-> The collision happens when the distance is < 0

(7) Publish through a ROS node
https://github.com/NVIDIA-ISAAC-ROS/isaac_manipulator/blob/main/isaac_manipulator_pick_and_place/scripts/grasp_reader.py

(8) Gripper open/close actions

https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/models/grippers/robotiq_85_gripper.py<br>
https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/models/grippers/robotiq_140_gripper.py
