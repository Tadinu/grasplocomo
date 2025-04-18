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

#include <iostream>

namespace mj_app {
template <typename... TArgs>  // Parameter pack
static void Print(const TArgs &...var) {
  // Folding expression
  ((std::cout << var << " "), ...) << std::endl;
}

// Spec
inline mjsBody *FindWorldBodySpec(mjSpec *model_spec) { return mjs_findBody(model_spec, "world"); }

// find all child specs of either a spec or a body
// Ref: https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/specs.cc - FindAllImpl
template <typename TSpec, typename TChildSpec,
          typename = std::enable_if<std::is_same_v<TSpec, mjSpec> | std::is_same_v<TSpec, mjsBody>>>
inline std::vector<TChildSpec *> FindAllChildSpecs(TSpec *base_spec, mjtObj type, int recurse) {
  if (type == mjOBJ_UNKNOWN) {
    // this should never happen
    throw std::runtime_error(
        "[FindAllChildSpecs] supports the types: body, frame, geom, site, "
        "joint, light, camera.");
  }

  std::vector<TChildSpec *> list;
  mjsElement *el = nullptr;
  if constexpr (std::is_same_v<TSpec, mjSpec>) {
    el = mjs_firstElement(base_spec, type);
  } else {
    el = mjs_firstChild(base_spec, type, recurse);
  }

  const std::string error = mjs_getError(mjs_getSpec(base_spec->element));
  if (!el && !error.empty()) {
    throw std::runtime_error(error);
  }
  while (el) {
    if constexpr (std::is_same_v<TChildSpec, mjsElement>) {
      list.push_back(el);
    } else {
      TChildSpec *child_spec = nullptr;
      if constexpr (std::is_same_v<TChildSpec, mjsBody>) {
        child_spec = mjs_asBody(el);
      } else if constexpr (std::is_same_v<TChildSpec, mjsCamera>) {
        child_spec = mjs_asCamera(el);
      } else if constexpr (std::is_same_v<TChildSpec, mjsFrame>) {
        child_spec = mjs_asFrame(el);
      } else if constexpr (std::is_same_v<TChildSpec, mjsGeom>) {
        child_spec = mjs_asGeom(el);
      } else if constexpr (std::is_same_v<TChildSpec, mjsJoint>) {
        child_spec = mjs_asJoint(el);
      } else if constexpr (std::is_same_v<TChildSpec, mjsLight>) {
        child_spec = mjs_asLight(el);
      } else if constexpr (std::is_same_v<TChildSpec, mjsSite>) {
        child_spec = mjs_asSite(el);
      }

      if (child_spec) {
        list.push_back(child_spec);
      }
    }

    if constexpr (std::is_same_v<TSpec, mjSpec>) {
      el = mjs_nextElement(base_spec, el);
    } else {
      el = mjs_nextChild(base_spec, el, recurse);
    }
  }
  return list;
}

// Key
inline int QueryKeyId(const mjModel *model, const std::string &key_name) {
  return model ? mj_name2id(model, mjOBJ_KEY, key_name.c_str()) : -1;
}

// Body
inline int QueryBodyId(const mjModel *model, const char *body_name) {
  return model ? mj_name2id(model, mjOBJ_BODY, body_name) : -1;
}

inline mjtNum *QueryBodyQuat(const mjData *data, int body_id, bool inertia_com = true) {
  if (data && (body_id > -1)) {
    if (inertia_com) {
      static mjtNum quat[4];
      mju_mat2Quat(quat, &data->ximat[9 * body_id]);
      return &quat[0];
    } else {
      return &data->xquat[4 * body_id];
    }
  }
  return nullptr;
}

inline mjtNum *QueryBodyQuat(const mjModel *model, const mjData *data, const char *body_name,
                             bool inertia_com = true) {
  return QueryBodyQuat(data, QueryBodyId(model, body_name), inertia_com);
}

inline mjtNum *QueryBodyRotMat(const mjData *data, int body_id, bool inertia_com = true) {
  if (data && (body_id > -1)) {
    if (inertia_com) {
      return &data->ximat[9 * body_id];
    } else {
      static mjtNum mat[9];
      mju_quat2Mat(mat, &data->xquat[4 * body_id]);
      return &mat[0];
    }
  }
  return nullptr;
}

inline mjtNum *QueryBodyRotMat(const mjModel *model, const mjData *data, const char *body_name,
                               bool inertia_com = true) {
  return QueryBodyRotMat(data, QueryBodyId(model, body_name), inertia_com);
}

inline mjtNum *QueryBodyPos(const mjData *data, int body_id, bool inertia_com = true) {
  if (data && (body_id > -1)) {
    return inertia_com ? &data->xipos[3 * body_id] : &data->xpos[3 * body_id];
  }
  return nullptr;
}

inline mjtNum *QueryBodyPos(const mjModel *model, const mjData *data, const char *body_name,
                            bool inertia_com = true) {
  return QueryBodyPos(data, QueryBodyId(model, body_name), inertia_com);
}

// Recursive function to set collision properties for a body and its descendants
inline void SetBodyTreeCollisionEnabled(mjsBody *base_body_spec, bool enabled) {
  for (const auto &geom_spec : FindAllChildSpecs<mjsBody, mjsGeom>(base_body_spec, mjOBJ_GEOM, true)) {
    geom_spec->contype = enabled;
    geom_spec->conaffinity = enabled;
  }
  for (const auto &child_body_spec : FindAllChildSpecs<mjsBody, mjsBody>(base_body_spec, mjOBJ_BODY, true)) {
    SetBodyTreeCollisionEnabled(child_body_spec, enabled);
  }
}

// Recursive function to set gravity compensation properties for a body and its descendants
inline void SetBodyTreeGravityCompensationEnabled(mjsBody *base_body_spec, bool enabled) {
  base_body_spec->gravcomp = enabled;
  for (const auto &child_body_spec : FindAllChildSpecs<mjsBody, mjsBody>(base_body_spec, mjOBJ_BODY, true)) {
    SetBodyTreeGravityCompensationEnabled(child_body_spec, enabled);
  }
}
}  // namespace mj_app
