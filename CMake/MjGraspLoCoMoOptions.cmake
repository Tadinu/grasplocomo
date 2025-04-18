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

# Global compilation settings
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # For LLVM tooling

if(NOT CMAKE_CONFIGURATION_TYPES)
  if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE
        "Release"
        CACHE STRING "Choose the type of build, recommanded options are: Debug or Release" FORCE
    )
  endif()
  set(BUILD_TYPES
      "Debug"
      "Release"
      "MinSizeRel"
      "RelWithDebInfo"
  )
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${BUILD_TYPES})
endif()

include(GNUInstallDirs)

# Change the default output directory in the build structure. This is not stricly needed, but helps
# running in Windows, such that all built executables have DLLs in the same folder as the .exe
# files.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")

set(OpenGL_GL_PREFERENCE GLVND)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

if(MSVC)
  add_compile_options(/Gy /Gw /Oi)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  #add_compile_options(-fdata-sections -ffunction-sections)
endif()

# We default to shared library.
set(BUILD_SHARED_LIBS
    ON
    CACHE BOOL "Build Mujoco as shared library."
)

option(MUJOCO_ENABLE_AVX "Build binaries that require AVX instructions, if possible." ON)
option(MUJOCO_ENABLE_AVX_INTRINSICS "Make use of hand-written AVX intrinsics, if possible." ON)
option(MUJOCO_ENABLE_RPATH "Enable RPath support when installing Mujoco." ON)
mark_as_advanced(MUJOCO_ENABLE_RPATH)

if(MUJOCO_ENABLE_AVX)
  include(CheckAvxSupport)
  get_avx_compile_options(AVX_COMPILE_OPTIONS)
else()
  set(AVX_COMPILE_OPTIONS)
endif()

option(MUJOCO_BUILD_MACOS_FRAMEWORKS "Build libraries as macOS Frameworks" OFF)

# Get some extra link options.
include(MujocoLinkOptions)
get_mujoco_extra_link_options(EXTRA_LINK_OPTIONS)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT MSVC))
  #set(EXTRA_COMPILE_OPTIONS
  #    -Werror
  #    -Wall
  #    -Wimplicit-fallthrough
  #    -Wunused
  #    -Wvla
  #    -Wno-int-in-bool-context
  #    -Wno-sign-compare
  #  )
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # Set -Wimplicit-fallthrough=5 to only allow fallthrough annotation via __attribute__.
    set(EXTRA_COMPILE_OPTIONS ${EXTRA_COMPILE_OPTIONS} -Wimplicit-fallthrough=5
                              -Wno-maybe-uninitialized
    )
  endif()
endif()

if(NOT CMAKE_INTERPROCEDURAL_OPTIMIZATION AND (CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug"))
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

include(MujocoHarden)
set(EXTRA_COMPILE_OPTIONS ${EXTRA_COMPILE_OPTIONS} ${MUJOCO_HARDEN_COMPILE_OPTIONS})
set(EXTRA_LINK_OPTIONS ${EXTRA_LINK_OPTIONS} ${MUJOCO_HARDEN_LINK_OPTIONS})

if(WIN32)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_DEPRECATE)
endif()
