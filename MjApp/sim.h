/*
 * BSD 3 - Clause License
 *
 * Copyright(c) 2025, Duc Than
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met :
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <atomic>
#include <chrono>
#include <filesystem>
#include <iosfwd>
#include <utility>
#include <vector>

// Eigen
#include <Eigen/Core>

// MuJoCo
#include <array_safety.h>
#include <fmt/format.h>
#include <mujoco/mujoco.h>

// GraspLoCoMo
#include "Core/Search/include/dxPointCloud.h"
#include "Grasp/include/dxGraspLoCoMo.h"

// MjApp
#include "MjApp/Utils/math_util.h"
#include "sim_base.h"

#define MJAPP_GRASP_POSE_GENERATION (1)

using namespace std;

namespace mj_app {
namespace mju = ::mujoco::sample_util;

static std::vector<std::pair<std::vector<double>, std::vector<double>>> obj_pointcloud;
static constexpr float VOXEL_SIZE = 0.003;
static constexpr bool USE_SCHUNK_PG70 = true;
static const std::string DEFAULT_OBJECT_TYPE = "mug";

// Simulate states not contained in MuJoCo structures
class Simulate : public SimulateBase {
public:
  Simulate(std::unique_ptr<mujoco::PlatformUIAdapter> platform_ui_adapter, mjvCamera *cam, mjvOption *opt,
           mjvPerturb *pert, bool is_passive)
      : SimulateBase(std::move(platform_ui_adapter), cam, opt, pert, is_passive) {
    obj_pointcloud = LoadPointCloud(OBJECT_POINTCLOUD_FILEPATH_TXT);
  }

  const std::string GRASP_LOCOMO_DIR = std::filesystem::current_path().string();
  const std::string MODELS_DIR = GRASP_LOCOMO_DIR + "/Models";

  // OBJECT
  const std::string OBJECT_MESH_FILEPATH;  // MODELS_DIR + "/mj_mug.obj";
  const std::string OBJECT_TYPE = OBJECT_MESH_FILEPATH.empty()
                                      ? DEFAULT_OBJECT_TYPE
                                      : std::string(std::filesystem::path(OBJECT_MESH_FILEPATH).stem());
  const std::string OBJECT_NAME = "TargetObject_" + OBJECT_TYPE;
  const std::string OBJECT_FREEJOINT_NAME = OBJECT_NAME + "_free_joint";
  const std::string OBJECT_POINTCLOUD_FILEPATH_TXT =
      GRASP_LOCOMO_DIR + "/Clouds/" + OBJECT_TYPE + "_cloud.txt";

  // GRIPPER
  // [SCHUNK_PG70]
  const std::string SCHUNK_PG70_XML_PATH = MODELS_DIR + "/schunk/schunk_pg70.xml";
  const std::string SCHUNK_PG70_NAME = std::filesystem::path(SCHUNK_PG70_XML_PATH).stem();
  static constexpr const char *SCHUNK_PG70_BASE_NAME = "pg70_palm_link";

  // [ROBOTIQ_2F85]
  const std::string ROBOTIQ_2F85_XML_PATH = MODELS_DIR + "/robotiq_2f85/robotiq_2f85.xml";
  const std::string ROBOTIQ_2F85_NAME = std::filesystem::path(ROBOTIQ_2F85_XML_PATH).stem();
  static constexpr const char *ROBOTIQ_2F85_BASE_NAME = "base_mount";

  // [ROBOTIQ_2F140]
  const std::string ROBOTIQ_2F140_XML_PATH = MODELS_DIR + "/robotiq_2f85/robotiq_2f140.xml";
  const std::string ROBOTIQ_2F140_NAME = std::filesystem::path(ROBOTIQ_2F140_XML_PATH).stem();
  static constexpr const char *ROBOTIQ_2F140_BASE_NAME = "base_mount";

  // MAIN GRIPPER MODEL
  const std::string GRIPPER_XML_PATH = USE_SCHUNK_PG70 ? SCHUNK_PG70_XML_PATH : ROBOTIQ_2F85_XML_PATH;
  const std::string GRIPPER_XML_DIRNAME = std::filesystem::path(GRIPPER_XML_PATH).parent_path();
  const std::string GRIPPER_NAME = std::filesystem::path(GRIPPER_XML_PATH).stem();
  const std::string GRIPPER_BASE_NAME = (GRIPPER_NAME == SCHUNK_PG70_NAME)    ? SCHUNK_PG70_BASE_NAME
                                        : (GRIPPER_NAME == ROBOTIQ_2F85_NAME) ? ROBOTIQ_2F85_BASE_NAME
                                                                              : ROBOTIQ_2F140_BASE_NAME;
  const std::string GRIPPER_BASE_FREEJOINT_NAME = GRIPPER_BASE_NAME + "_free_joint";

  // GRASP VISUALIZING
  static constexpr uint8_t GRASP_VISUALIZING_TIME_STEP = 200;  // ms

  // MAIN XML
  const std::string SCENE_XML_PATH = GRIPPER_XML_DIRNAME + "/scene.xml";

  struct PclPoint {
    bool is_quat = false;
    std::vector<mjtNum> position;
    std::vector<mjtNum> normal_or_quat;

    std::vector<mjtNum> quat() const {
      return is_quat ? normal_or_quat : std::vector<mjtNum>{1.0, 0.0, 0.0, 0.0};
    }

    std::vector<mjtNum> normal() const {
      return is_quat ? std::vector<mjtNum>{0.0, 0.0, 0.0} : normal_or_quat;
    }
  };

  std::vector<PclPoint> obj_pointcloud;

  std::vector<PclPoint> LoadPointCloud(const std::string &filepath, bool normal_as_quat = true) {
    std::vector<PclPoint> points;
    try {
      std::ifstream file(filepath);
      if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return points;
      }

      std::string line;
      int i = 0;
      while (std::getline(file, line)) {
        // Skip the header
        if (i++ == 0) continue;

        std::istringstream iss(line);
        std::vector<double> data;
        double num;
        while (iss >> num) {
          data.push_back(num);
        }

        // Ensure at least position + normal
        if (data.size() < 6) continue;

        PclPoint point;
        point.is_quat = normal_as_quat;
        point.position.assign({data[0], data[1], data[2]});
        const std::vector<double> normal = {data[3], data[4], data[5]};

        if (normal_as_quat) {
          point.normal_or_quat.resize(4);
          mj_app::MjuNormalToQuat(point.normal_or_quat.data(), normal.data());
        } else {
          point.normal_or_quat = normal;
        }

        points.emplace_back(std::move(point));
      }
    } catch (const std::ifstream::failure &e) {
      std::cerr << "Failed reading file: " << filepath << " " << e.what() << std::endl;
      std::cerr << "Points num read: " << points.size() << std::endl;
    }
    return points;
  }

  mjModel *ConstructCustomModel() const {
    mjModel *model = nullptr;

    // 1- Create [scene_spec]
    std::array<char, 1024> error{};
    mjSpec *scene_spec = mj_parseXML(SCENE_XML_PATH.c_str(), nullptr, error.data(), error.size());
    if (error[0] != '\0') {
      std::cout << "[mj_grasplocomo - Simulate::ConstructCustomModel()]" << error.data() << std::endl;
    }
    if (!scene_spec) {
      return nullptr;
    }
    // scene_spec->memory = 10000000000;
    // scene_spec->option.noslip_iterations = 5;
    // scene_spec->option.noslip_tolerance = 1e-06;
    // scene_spec->option.enableflags |= mjENBL_MULTICCD;
    // scene_spec->option.ccd_tolerance = 1e-6;
    // scene_spec->option.ccd_iterations = 50;

    mjsBody *world_body = mj_app::FindWorldBodySpec(scene_spec);

    // 1- [GRIPPER]
    mjsBody *gripper_base_spec = mjs_findBody(scene_spec, GRIPPER_BASE_NAME.c_str());
    gripper_base_spec->pos[2] = 0.5;
    // A free joint is required to move gripper freely around the scene
    // A mocap can be added if it needs to be control dynamically, instead of just teleportation
    mjsJoint *gripper_base_joint = mjs_addJoint(gripper_base_spec, nullptr);
    mjs_setString(gripper_base_joint->name, GRIPPER_BASE_FREEJOINT_NAME.c_str());
    gripper_base_joint->type = mjJNT_FREE;
    gripper_base_joint->align = false;
    // Disable gripper collision to avoid physical disturbance to scene upon overlapping check
    mj_app::SetBodyTreeCollisionEnabled(gripper_base_spec, false);
    // Enable gravity compensation for [gripper]
    mj_app::SetBodyTreeGravityCompensationEnabled(gripper_base_spec, true);

    // 2- [TARGET OBJECT]
    // NOTE: FIRST, OBJ MUST BE SPAWNED AT THE ORIGIN FOR [LOCAL_GRASPS] TO BE POST_PROCESSED
    mjsBody *target_obj = mjs_addBody(world_body, nullptr);
    mjs_setString(target_obj->name, OBJECT_NAME.c_str());
    memcpy(target_obj->pos, mj_app::POSITION_ZERO, sizeof(target_obj->pos));
    memcpy(target_obj->quat, mj_app::QUAT_IDENTITY, sizeof(target_obj->quat));
    target_obj->mocap = false;

    // Either composed of a single rigid body from [OBJECT_MESH_FILEPATH] or voxels from [OBJECT_POINTCLOUD]
    if (!OBJECT_MESH_FILEPATH.empty()) {
      mjsMesh *obj_mesh = mjs_addMesh(scene_spec, nullptr);
      const auto mesh_path = std::filesystem::path(OBJECT_MESH_FILEPATH);
      const auto mesh_name = mesh_path.stem();
      mjs_setString(obj_mesh->file, mesh_path.filename().c_str());
      mjsGeom *obj_mesh_geom = mjs_addGeom(target_obj, nullptr);
      mjs_setString(obj_mesh_geom->meshname, mesh_name.c_str());
      mjs_setString(obj_mesh_geom->name, mesh_name.c_str());
      obj_mesh_geom->type = mjGEOM_MESH;
      // Disable collision to avoid physical disturbance upon overlapping check with the gripper
      obj_mesh_geom->contype = 0;
      obj_mesh_geom->conaffinity = 0;

      // Disable target_obj collision, so it can freely move around just to be easier to visualize
      mj_app::SetBodyTreeCollisionEnabled(target_obj, false);

      // Free joint
      mjsJoint *target_obj_base_joint = mjs_addJoint(target_obj, nullptr);
      mjs_setString(target_obj_base_joint->name, OBJECT_FREEJOINT_NAME.c_str());
      target_obj_base_joint->type = mjJNT_FREE;
      target_obj_base_joint->align = false;

      // Gravity compensation
      if (target_obj_base_joint) {
        mj_app::SetBodyTreeGravityCompensationEnabled(target_obj, true);
      }
    } else {
      for (const auto &point : obj_pointcloud) {
        mjsGeom *point_geom = mjs_addGeom(world_body, nullptr);
        point_geom->type = mjGEOM_BOX;
        static const double voxel_size[] = {VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE};
        memcpy(point_geom->size, voxel_size, sizeof(point_geom->size));
        const float rgba[] = {0.6, 0.5, 0.3, 1.0};
        memcpy(point_geom->rgba, rgba, sizeof(point_geom->rgba));
        // Disable collision to avoid physical disturbance upon overlapping check with the gripper
        point_geom->contype = 0;
        point_geom->conaffinity = 0;
        memcpy(point_geom->pos, point.position.data(), sizeof(point_geom->pos));
        memcpy(point_geom->quat, point.quat().data(), sizeof(point_geom->quat));
      }
    }

    // 3- Compile [scene_spec] to [mjModel]
    model = mj_compile(scene_spec, nullptr);
    if (model) {
#if 0
      std::array<char, 1024> err;
      mj_saveXML(scene_spec, (std::string(mjs_getString(scene_spec->modelname)) + ".xml").c_str(), err.data(),
                 err.size());
#endif
    }
    return model;
  }

  void InitInThread(mjModel *model, mjData *data) override {
    // Home pos
    mj_resetDataKeyframe(model, data, mj_app::QueryKeyId(model, "home"));

    // Configure visualization
    // Site groups
    for (auto i = 0; i < model->nsite; ++i) {
      opt.sitegroup[mjMAX(0, mjMIN(mjNGROUP - 1, model->site_group[i]))] = true;
    }
  }

  void PosLoadInit(const mjModel *model, const mjData *data) {}

  static Eigen::Vector2d Circle(double t, double r, double h, double k, double f) {
    // Return the (x, y) coordinates of a circle with radius r centered at (h, k)
    // as a function of time t and frequency f.
    auto x = r * cos(2 * M_PI * f * t) + h;
    auto y = r * sin(2 * M_PI * f * t) + k;
    return {x, y};
  }

  void VisualizeGrasps(const mjModel *model, const mjData *data) const {
#if 0
    // Move [target_obj] to its origin pose
    static bool init_target_obj = false;
    if (!init_target_obj) {
      MoveTargetObject(model, data, {0, 0, 0, 1, 0, 0, 0});
      init_target_obj = true;
    }
    PrintTargetObj(model, data);
    PrintPointCloud(model, data);
#endif

    // Show gripper at the next grasp
    static int grasp_idx = 0;
    static const std::string OUTPUT_GRASP_FILEPATH =
        (GRASP_LOCOMO_DIR + "/" + OBJECT_TYPE + "_grasp_results_recalculated.txt");
    static const auto global_grasps = GenerateGrasps(model, data, nullptr);

    if (grasp_idx < global_grasps.size()) {
      MoveGripper(model, data, global_grasps[grasp_idx++]);
      std::this_thread::sleep_for(std::chrono::milliseconds(GRASP_VISUALIZING_TIME_STEP));
    }
  }

  void MoveTargetObject(const mjModel *model, const mjData *data,
                        const std::array<mjtNum, 7> &target_pose) const {
    memcpy(&data->qpos[model->jnt_dofadr[mj_name2id(model, mjOBJ_JOINT, OBJECT_FREEJOINT_NAME.c_str())]],
           target_pose.data(), sizeof(target_pose));
  }

  void PrintTargetObj(const mjModel *model, const mjData *data) const {
    const int obj_id = mj_name2id(model, mjOBJ_BODY, OBJECT_NAME.c_str());
    auto *pos = mj_app::QueryBodyPos(data, obj_id);
    auto *quat = mj_app::QueryBodyQuat(data, obj_id);
    mj_app::Print(OBJECT_NAME, "- pos: ", pos[0], pos[1], pos[2]);
    mj_app::Print(OBJECT_NAME, "- quat: ", quat[0], quat[1], quat[2], quat[3]);
  }

  void PrintPointCloud(const mjModel *model, const mjData *data) const {
    for (int i = 0; i < model->body_geomnum[0]; ++i) {
      mj_app::Print(i, data->geom_xpos[model->body_geomadr[0] + i],
                    data->geom_xpos[model->body_geomadr[0] + i + 1],
                    data->geom_xpos[model->body_geomadr[0] + i + 2]);
    }
  }

protected:
  void ModifyVisualScene(mjvScene *scn, const mjModel *model, const mjData *data) override {}

  using GraspPose = std::array<mjtNum, 7>;
  struct GraspPoseEntry {
    GraspPose pre_grasp;
    GraspPose grasp;
    GraspPose post_grasp;
    double probability = 0;
    double opening = 0;
    double openingMax = 0;
  };
  using GraspPoseEntryList = std::vector<GraspPoseEntry>;

  GraspPoseEntryList GenerateGrasps(const mjModel *model, const mjData *data,
                                    const char *output_filepath = nullptr) const {
    mj_app::Print("LoCoMo Grasping -------- ");
    using dxGraspModel = dxGripperModel::GraspModelPG70;

    // 1- Load the point cloud
    dxPointCloud cloud;
    mj_app::Print("Loading: " + OBJECT_POINTCLOUD_FILEPATH_TXT);
    cloud.loadFromFile(OBJECT_POINTCLOUD_FILEPATH_TXT);

    // 2.1- downsampling: resolution of the point cloud -> IMPORTANT!
    // resolutionFactor: Used to compute the LoCoMo sphere radius
    dxGraspLoCoMo<dxGraspModel> grasp;
    grasp.setResolution(0.008);

    // 2.2- Computing grasps
    mj_app::Print("Computing grasps...");
    const auto &grasp_results = grasp.locomoGrasp(cloud).grasps;
    if (grasp_results.empty()) {
      mj_app::Print("No grasps found!");
      return {};
    }
    mj_app::Print("Results:", grasp_results.size());

    // 3- Display the 10 highest ranked grasps
    int Ngrasps = 10;
    cout << "pre-grasp pose | grasp pose | post-grasp pose | gripper opening | score" << endl;
    for (int i = 0; i < min(Ngrasps, static_cast<int>(grasp_results.size())); i++) {
      const auto &g = grasp_results[i];

      cout << "Grasp #" << i << endl;
#if DX_GRASP_AS_POS_QUAT
      dxGripperModel::GraspSuite::write_grasp(cout, g.preGrasp);
      dxGripperModel::GraspSuite::write_grasp(cout, g.pose);
      dxGripperModel::GraspSuite::write_grasp(cout, g.postGrasp);
#else
      cout << g.getColMajorVector(g.preGrasp) << "|";
      cout << g.getColMajorVector(g.pose) << "|";
      cout << g.getColMajorVector(g.postGrasp) << "|";
#endif
      cout << g.opening << " | ";
      cout << g.fs.prob << endl << endl;
    }

    // 4- Output global grasp poses
    GraspPoseEntryList out_gb_poses;
    const auto fPose = [](const Eigen::Matrix4d &pose) {
      GraspPose grasp_pose;
      Eigen::Vector3d pos = pose.block<3, 1>(0, 3);
      Eigen::Quaterniond quat(Eigen::Matrix3d(pose.block<3, 3>(0, 0)));
      quat.normalize();
      memcpy(grasp_pose.data(), pos.transpose().data(), 3 * sizeof(mjtNum));
      grasp_pose[3] = quat.w();
      grasp_pose[4] = quat.x();
      grasp_pose[5] = quat.y();
      grasp_pose[6] = quat.z();
      return grasp_pose;
    };
    for (const auto &grasp_result : grasp.results.grasps) {
      out_gb_poses.emplace_back(fPose(grasp_result.preGrasp), fPose(grasp_result.pose),
                                fPose(grasp_result.postGrasp), grasp_result.fs.prob, grasp_result.opening,
                                grasp_result.openingMax);
    }

    // 5- Save grasps to file
    if (output_filepath) {
      // [target_obj]'s pose
      int obj_id = mj_name2id(model, mjOBJ_BODY, OBJECT_NAME.c_str());
      const mjtNum *obj_pos = mj_app::QueryBodyPos(data, obj_id);
      const mjtNum *obj_quat = mj_app::QueryBodyQuat(data, obj_id);

      ofstream f;
      f.open(output_filepath);
      f << "Pregrasp pose | Grasp pose | Postgrasp pose | probability | opening | opening_max" << endl;

      const auto fWritePose = [&f, obj_pos, obj_quat](const GraspPose &global_grasp) {
        mjtNum neg_obj_pos[3], neg_obj_quat[4];
        // [global_grasp]
        const mjtNum *gb_pos = global_grasp.data();
        const mjtNum *gb_quat = global_grasp.data() + 3;

        // [local_grasp] = neg(target_obj)*global_grasp
        GraspPose local_grasp;
        mju_negPose(neg_obj_pos, neg_obj_quat, obj_pos, obj_quat);
        mju_mulPose(local_grasp.data(), local_grasp.data() + 3, neg_obj_pos, neg_obj_quat, gb_pos, gb_quat);

        for (auto i = 0; i < local_grasp.size(); i++) {
          f << local_grasp[i];
          if (i == local_grasp.size() - 1) {
            f << "|";
          } else {
            f << " ";
          }
        }
      };

      for (const auto &entry : out_gb_poses) {
        fWritePose(entry.pre_grasp);
        fWritePose(entry.grasp);
        fWritePose(entry.post_grasp);
        f << entry.probability << "|" << entry.opening << "|" << entry.openingMax << endl;
      }
      f.close();
      mj_app::Print("Local Grasps generated to: ", output_filepath);
    }
    return out_gb_poses;
  }

  void MoveGripper(const mjModel *model, const mjData *data, const GraspPoseEntry &pose_entry) const {
    const auto& pose = pose_entry.grasp;
    memcpy(
        &data->qpos[model->jnt_dofadr[mj_name2id(model, mjOBJ_JOINT, GRIPPER_BASE_FREEJOINT_NAME.c_str())]],
        pose.data(), sizeof(pose));

    if constexpr (USE_SCHUNK_PG70) {
      assert(model->nu == 2);
      // This needs further fingers' actuator dynamics tuning to show correct fingers opening
      const auto finger_ctrl = 0.5 * pose_entry.opening;
      data->ctrl[0] = finger_ctrl;
      data->ctrl[1] = finger_ctrl;
    }
  }
};
}  // namespace mj_app
