# BSD 3-Clause License
# 
# Copyright(c) 2025, Duc Than
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.16)

# INTERPROCEDURAL_OPTIMIZATION is enforced when enabled.
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
# Default to GLVND if available.
set(CMAKE_POLICY_DEFAULT_CMP0072 NEW)

# This line has to appear before 'PROJECT' in order to be able to disable incremental linking
set(MSVC_INCREMENTAL_DEFAULT ON)


enable_language(C)
enable_language(CXX)
if(APPLE)
  enable_language(OBJC)
  enable_language(OBJCXX)
endif()

option(SIMULATE_BUILD_EXECUTABLE "Build the simulate executable binary." ON)
option(SIMULATE_GLFW_DYNAMIC_SYMBOLS "Whether to resolve GLFW symbols dynamically." OFF)

# Simulate library
add_library(libmjgrasplocomo STATIC $<TARGET_OBJECTS:mujoco::platform_ui_adapter>)
set_target_properties(libmjgrasplocomo PROPERTIES OUTPUT_NAME mj_grasplocomo)
add_library(mujoco::libmjgrasplocomo ALIAS libmjgrasplocomo)
target_sources(
  libmjgrasplocomo
  PUBLIC sim_base.h sim.h
         Utils/core_util.h
         Utils/math_util.h
         ${PROJECT_SOURCE_DIR}/Core/Sys/include/dxAsyncTasks.h
         ${PROJECT_SOURCE_DIR}/Core/Sys/include/dxFile.h
         ${PROJECT_SOURCE_DIR}/Core/Math/include/dxFunctions.h
         ${PROJECT_SOURCE_DIR}/Core/Math/include/dxPoint3.h
         ${PROJECT_SOURCE_DIR}/Core/Search/include/dxPointCloud.h
         ${PROJECT_SOURCE_DIR}/Core/Search/include/dxSearchTree.h
         ${PROJECT_SOURCE_DIR}/Grasp/include/dxGraspLoCoMo.h
         ${PROJECT_SOURCE_DIR}/Grasp/include/dxGripperModel.h
         ${PROJECT_SOURCE_DIR}/Grasp/include/dxVoxel.h
  PRIVATE sim_base.cc
          ${PROJECT_SOURCE_DIR}/Grasp/src/dxGraspLoCoMo.cpp
          $<TARGET_OBJECTS:mujoco::platform_ui_adapter>
)

target_include_directories(libmjgrasplocomo
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${PROJECT_SOURCE_DIR}/ThirdParty/nanoflann/include/
)
target_compile_options(libmjgrasplocomo PRIVATE ${MJGRASPLOCOMO_COMPILE_OPTIONS})
target_link_libraries(libmjgrasplocomo PUBLIC
  lodepng mujoco::mujoco mujoco::platform_ui_adapter
  #absl::any_invocable
  #absl::check
  #absl::flat_hash_map
  #absl::log
  absl::random_random
  absl::span
  absl::strings
  absl::status
  absl::statusor
  Eigen3::Eigen
)
target_link_options(libmjgrasplocomo PRIVATE ${MJGRASPLOCOMO_LINK_OPTIONS})

if(APPLE)
  target_sources(libmjgrasplocomo PRIVATE macos_gui.mm)
  target_link_libraries(libmjgrasplocomo PUBLIC "-framework Cocoa")
endif()

# Build simulate executable
if(SIMULATE_BUILD_EXECUTABLE)
  if(APPLE)
    set(SIMULATE_RESOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../dist/mujoco.icns)
  elseif(WIN32)
    set(SIMULATE_RESOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../dist/simulate.rc)
  else()
    set(SIMULATE_RESOURCE_FILES "")
  endif()

  add_executable(mj_grasplocomo main.cc ${SIMULATE_RESOURCE_FILES})
  target_compile_options(mj_grasplocomo PUBLIC ${MJGRASPLOCOMO_COMPILE_OPTIONS})
  if(WIN32)
    add_custom_command(
      TARGET mj_grasplocomo
      PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/../dist/mujoco.ico
              ${CMAKE_CURRENT_SOURCE_DIR}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E rm ${CMAKE_CURRENT_SOURCE_DIR}/mujoco.ico
    )
  endif()

  target_include_directories(mj_grasplocomo
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${PROJECT_SOURCE_DIR}/ThirdParty/nanoflann/include/
  )

  target_link_libraries(
    mj_grasplocomo
    libmjgrasplocomo
    mujoco::mujoco
    glfw
    Threads::Threads
    lodepng
    absl::span
    absl::hash
    absl::btree
    absl::hashtablez_sampler
    absl::strings
    absl::status
    absl::statusor
  )

  target_link_options(mj_grasplocomo PRIVATE ${MJGRASPLOCOMO_LINK_OPTIONS})
endif()
