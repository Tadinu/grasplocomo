/**
 * @file
 * @brief Source file for dxGraspLoCoMo
 * @author Maxime Adjigble
 * @author Naresh Marturi
 * @ref License
 */
 
/*
* BSD 3 - Clause License
*
* Copyright(c) 2021, Maxime Adjigble 
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
#ifndef DX_GRASP_LOCOMO_INCLUDE
#define DX_GRASP_LOCOMO_INCLUDE

#include <algorithm>
#include <iostream>
#include <string>
#include <math.h>
#include <random>
#include <time.h>
#include <exception>
#include "dxGripperModel.h"
#include "dxVoxel.h"
#include "dxSearchTree.h"
//#include "dxVision.h"
#include "dxAsyncTasks.h"


template <typename TGraspModel>
class dxGraspLoCoMo
{
public:
    typedef std::pair<dxGripperModel::Feature, dxGripperModel::Feature> FeaturePair;
    typedef std::vector<FeaturePair> FeaturePairVec;

    typedef std::pair<dxGripperModel::Feature, std::vector<dxGripperModel::Feature>> FeaturePairOneToMany;
    typedef std::vector<FeaturePairOneToMany> FeaturePairOneToManyVec;
    
    typedef std::vector<TGraspModel> GraspVec;

    struct GraspResult
    {
        GraspVec grasps;

        void print()
        {
            cout << "[RESULTS] : "
                 << "Ngrasps " << grasps.size()  << endl;
        }
    };
    GraspResult results;

    static void printMsg(string message, string prep = "", string end = "\n")
    {
        std::cout << prep << message << endl;
    }

    dxGraspLoCoMo()
    {
        setResolution(0.008, 4);
    }

    ~dxGraspLoCoMo()
    {
    }

    void setResolution(double downsampling = 0.008, int resolutionFactor = 4)
    {
        this->downsampling = downsampling;
        this->resolution = resolutionFactor * downsampling;
    }

    GraspResult locomoGrasp(dxPointCloud cloud)
    {
        //Load gripper cloud and initialize
        loadGripperCloud();
        gripper.init(resolution);

        //Load scene cloud and initialize
        scene.cloud = std::move(cloud);
        scene.init(resolution);
        std::vector<dxVoxel::Leaf> s2s = scene.elements;

        GraspVec graspCandidates;
        GraspResult graspResult;

        typedef std::vector<int> IdVec;
        struct Params { int id; };

        auto computeGrasps = [&](const Params& p) -> GraspVec {
            int idx = p.id;
            GraspVec candidates;

            TGraspModel graspModel;
            graspModel.setTree(scene.kdtree);

            dxVoxel::Leaf s1 = s2s[idx];
            std::vector<dxVoxel::Leaf> s2Vec = s2s;

            Eigen::Vector3d p1 = s1.point_;
            Eigen::Vector3d n1 = s1.fisher_mean_;
            dxGripperModel::IndexedPtNorm f1 = dxGripperModel::Feature::getIndexedPtNorm(idx, dxPoint3(p1), dxNormal3(n1));

            for (int j = 0; j < s2Vec.size(); j++) {
                int jdx = j;
                dxVoxel::Leaf s2 = s2Vec[jdx];
                Eigen::Vector3d p2 = s2.point_;
                Eigen::Vector3d n2 = s2.fisher_mean_;
                dxGripperModel::IndexedPtNorm f2 = dxGripperModel::Feature::getIndexedPtNorm(jdx, dxPoint3(p2), dxNormal3(n2));

                dxGripperModel::Feature f;
                f.fromPtNorm(f1, f2);
                if (!f.isValid) {
                    continue;
                }

                f.prob = s1.locomo_*s2.locomo_;

                //PARAMS - Force closure
                if (f.isForceClosurePG70(graspModel.frictionAngle, graspModel.openingMax)) {

                    //Computation of grasp features
                    Eigen::Vector3d p2p1 = p2 - p1;
                    double gripperOpening = p2p1.norm();
                    p2p1.normalize();
                    n1 = -p2p1;
                    n2 = p2p1;
                    dxGripperModel::Feature fg;
                    fg.fromPtNorm(dxGripperModel::Feature::getIndexedPtNorm(0, dxPoint3(p1), dxNormal3(n1)),
                        dxGripperModel::Feature::getIndexedPtNorm(1, dxPoint3(p2), dxNormal3(n2)));

                    // Sampling Nrots rotated features around p1p2
                    int Nrots = graspModel.NrotSampled;
                    Eigen::Vector3d Uz = Eigen::Vector3d(0, 0, 1);
                    for (float angle = 0; angle < 2 * M_PI; angle += (2 * M_PI / Nrots)) {
                        // Rotation around the Z Axis
                        Eigen::AngleAxis<double> T(angle, Uz);
                        Eigen::Matrix4d R = Eigen::Matrix4d::Identity();
                        R.block<3, 3>(0, 0) = T.toRotationMatrix();

                        Eigen::Matrix4d refRotx = graspModel.TfeatureInRgripper * R;
                        Eigen::Matrix4d refRotxInRbase = f.ref * refRotx.inverse();

                        graspModel.setPose(refRotxInRbase, gripperOpening);
                        if (!graspModel.inCollision())
                        {
                            fg.setRef(refRotx);

                            graspModel.fs = f;
                            graspModel.fg = fg;
                            graspModel.computePrePostGrasp();
                            candidates.push_back(graspModel);
                        }
                    }
                }
            }

            return candidates;
        };

        //Create async task
        dxAsyncTasks<Params, GraspVec> tasks;
        tasks.setAsyncFunction(computeGrasps);

        //Set parameters for the tasks
        for (int i = 0; i < s2s.size(); i++) {
            tasks.runTask(Params{.id = i});
        }

        // Get results from tasks
        std::vector<GraspVec> candidates = tasks.getResults();

        // Unpack results
        const auto total_size = std::accumulate(
            candidates.begin(), candidates.end(), 0,
            [](size_t sum, const GraspVec& candidate) {
                return sum + candidate.size();
            }
        );
        graspCandidates.reserve(total_size);

        // Move [candidates] to [graspCandidates]
        for (auto& cand : candidates) {
          std::move(cand.begin(), cand.end(), std::back_inserter(graspCandidates));
        }

        // Sort [graspCandidates]
        std::sort(graspCandidates.begin(), graspCandidates.end(),
            [](const TGraspModel& g1, const TGraspModel& g2) { return g1.fs.prob > g2.fs.prob; });

        graspResult.grasps = std::move(graspCandidates);
        //graspResult.print();

        results = std::move(graspResult);
        return results;
    }

    void saveGrasps(string path) {
        ofstream f;

        GraspVec grasps = results.grasps;
        f.open(path);
        for (int i = 0; i < grasps.size(); i++) {
            //Saving header
            if (i == 0) {
                grasps[i].save(f, true);
            }
            grasps[i].save(f);
        }
        f.close();
    }

protected:

    void loadGripperCloud()
    {
        gripper.cloud.addPoint(1, 0, 0, 1, 0, 0);
        gripper.cloud.addPoint(-1, 0, 0, -1, 0, 0);
    };


    //shared_ptr<vision> pVision;
    //shared_ptr<dxVision> viz;

    dxVoxel::VoxelData scene;
    dxVoxel::VoxelData gripper;
    double resolution;
    double downsampling;
};

#endif // !DX_GRASP_LOCOMO_INCLUDE
