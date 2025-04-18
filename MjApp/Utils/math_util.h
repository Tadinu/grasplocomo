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

// MjApp
#include "MjApp/Utils/core_util.h"

namespace mj_app {
// HELPER FUNCTIONS ---------------------
static constexpr mjtNum POSITION_ZERO[3] = {0, 0, 0};
static constexpr mjtNum QUAT_IDENTITY[4] = {1, 0, 0, 0};
static constexpr mjtNum POSE_IDENTITY[7] = {0, 0, 0, 1, 0, 0, 0};
// NOTE: Ones prefixed with "Mj" are either copied (while waiting to be released) or wrapper of/modified from
// mju_ API
static void MjuNormalToQuat(mjtNum quat[4], const mjtNum norm[3]) {
  // Reference direction (world z-axis)
  const mjtNum z_ref[3] = {0.0, 0.0, 1.0};

  // Compute rotation axis (cross product)
  mjtNum axis[3] = {z_ref[1] * norm[2] - z_ref[2] * norm[1], z_ref[2] * norm[0] - z_ref[0] * norm[2],
                    z_ref[0] * norm[1] - z_ref[1] * norm[0]};

  // Compute rotation angle, clipped for numerical stability
  double dot = std::max(-1.0, std::min(1.0, z_ref[0] * norm[0] + z_ref[1] * norm[1] + z_ref[2] * norm[2]));
  double angle = std::acos(dot);

  // Special case: (0, 0, -1) normal
  if (std::abs(norm[0]) < 1e-6 && std::abs(norm[1]) < 1e-6 && std::abs(norm[2] + 1) < 1e-6) {
    // 180-degree around x-axis
    mju_zero4(quat);
    quat[1] = 1.0;
  }

  // Normalize rotation axis
  if (std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]) > 1e-6) {
    // Avoid divide by zero
    mju_normalize3(axis);
  } else {
    // Default axis if normal is aligned with z_ref
    mju_zero3(axis);
    axis[0] = 1.0;
  }

  // Convert [axis, angle] -> quat
  mju_axisAngle2Quat(quat, axis, angle);
}
}  // end namespace mj_app
