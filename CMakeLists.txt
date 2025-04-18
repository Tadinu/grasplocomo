# BSD 3-Clause License
# 
# Copyright (c) 2021, Maxime Adjigble
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

project(
        mj_grasplocomo
        VERSION 0.0.1
        DESCRIPTION "Robotic grasp generation C/C++ library for parallel-jaw grippers visualized in MuJoCo"
        HOMEPAGE_URL "https://github.com/maximeadjigble/grasplocomo"
)

enable_language(C)
enable_language(CXX)
if (APPLE)
    enable_language(OBJC)
    enable_language(OBJCXX)
endif ()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMake")
# SimulateDependencies (platform_ui_adapter, etc.)
include(SimulateDependencies)
include(MjGraspLoCoMoOptions)
include(MujocoMacOS)

include(FindOrFetch)

find_package(Python)
find_package(Threads REQUIRED)

set(MUJOCO_BUILD_EXAMPLES OFF)
set(MUJOCO_BUILD_SIMULATE ON)
set(MUJOCO_BUILD_TESTS OFF)
set(MUJOCO_TEST_PYTHON_UTIL OFF)

set(MUJOCO_GIT_TAG
        main
        CACHE STRING "Git revision for MuJoCo."
)

findorfetch(
        USE_SYSTEM_PACKAGE
        OFF
        PACKAGE_NAME
        mujoco
        LIBRARY_NAME
        mujoco
        GIT_REPO
        https://github.com/google-deepmind/mujoco.git
        GIT_TAG
        ${MUJOCO_GIT_TAG}
        TARGETS
        mujoco::mujoco
        mujoco::platform_ui_adapter
        EXCLUDE_FROM_ALL
)

# TODO(nimrod): Update to the latest version of abseil, or use the one defined
# by MuJoCo, once grpc fix their build issues.
set(MUJOCO_DEP_VERSION_abseil
        master
        CACHE STRING "Version of `abseil` to be fetched."
        FORCE
)

set(BUILD_SHARED_LIBS_OLD ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS
        OFF
        CACHE INTERNAL "Build SHARED libraries"
)

findorfetch(
        USE_SYSTEM_PACKAGE
        OFF
        PACKAGE_NAME
        GTest
        LIBRARY_NAME
        googletest
        GIT_REPO
        https://github.com/google/googletest.git
        GIT_TAG
        main
        TARGETS
        gtest
        gmock
        gtest_main
        EXCLUDE_FROM_ALL
)

set(ABSL_PROPAGATE_CXX_STD ON)
set(ABSL_BUILD_TESTING OFF)
# ABSL_ENABLE_INSTALL is needed for
# https://github.com/protocolbuffers/protobuf/issues/12185#issuecomment-1594685860
set(ABSL_ENABLE_INSTALL ON)
findorfetch(
        USE_SYSTEM_PACKAGE
        OFF
        PACKAGE_NAME
        absl
        LIBRARY_NAME
        abseil-cpp
        GIT_REPO
        https://github.com/abseil/abseil-cpp.git
        GIT_TAG
        ${MUJOCO_DEP_VERSION_abseil}
        TARGETS
        absl::core_headers
        EXCLUDE_FROM_ALL
)

set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_BUILD_DOCS OFF)
set(GLFW_INSTALL OFF)

findorfetch(
        USE_SYSTEM_PACKAGE
        MJAPP_USE_SYSTEM_GLFW
        PACKAGE_NAME
        glfw
        LIBRARY_NAME
        glfw
        GIT_REPO
        https://github.com/glfw/glfw.git
        GIT_TAG
        ${MUJOCO_DEP_VERSION_glfw3}
        TARGETS
        glfw
        EXCLUDE_FROM_ALL
)

set(BUILD_SHARED_LIBS
        ${BUILD_SHARED_LIBS_OLD}
        CACHE BOOL "Build SHARED libraries" FORCE
)
unset(BUILD_SHARED_LIBS_OLD)

if (NOT TARGET lodepng)
    FetchContent_Declare(
            lodepng
            GIT_REPOSITORY https://github.com/lvandeve/lodepng.git
            GIT_TAG ${MJAPP_DEP_VERSION_lodepng}
    )

    FetchContent_GetProperties(lodepng)
    if (NOT lodepng_POPULATED)
        FetchContent_Populate(lodepng)
        # This is not a CMake project.
        set(LODEPNG_SRCS ${lodepng_SOURCE_DIR}/lodepng.cpp)
        set(LODEPNG_HEADERS ${lodepng_SOURCE_DIR}/lodepng.h)
        add_library(lodepng STATIC ${LODEPNG_HEADERS} ${LODEPNG_SRCS})
        target_compile_options(lodepng PRIVATE ${MUJOCO_MACOS_COMPILE_OPTIONS})
        target_link_options(lodepng PRIVATE ${MUJOCO_MACOS_LINK_OPTIONS})
        target_include_directories(lodepng PUBLIC ${lodepng_SOURCE_DIR})
    endif ()
endif ()

# Eigen3
find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
if (${CMAKE_CXX_STANDARD} EQUAL "20")
    add_compile_definitions(EIGEN_HAS_STD_RESULT_OF=0)
endif ()

set(MJGRASPLOCOMO_COMPILE_OPTIONS "${AVX_COMPILE_OPTIONS}" "${EXTRA_COMPILE_OPTIONS}")
set(MJGRASPLOCOMO_LINK_OPTIONS "${EXTRA_LINK_OPTIONS}")

add_subdirectory(MjApp)

