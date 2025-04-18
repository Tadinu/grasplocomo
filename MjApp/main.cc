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

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>

// MuJoCo
#include <array_safety.h>
#include <glfw_adapter.h>
#include <mujoco/mujoco.h>

// mj_grasplocomo
#include "MjApp/sim.h"

#define MUJOCO_PLUGIN_DIR "mujoco_plugin"

extern "C" {
#if defined(_WIN32) || defined(__CYGWIN__)
#include <windows.h>
#else
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <sys/errno.h>
#include <unistd.h>
#endif
}

namespace mj_app {
namespace mju = ::mujoco::sample_util;
// constants
const double syncMisalign = 0.1;        // maximum mis-alignment before re-sync (simulation seconds)
const double simRefreshFraction = 0.7;  // fraction of refresh available for simulation
const int kErrorLength = 1024;          // load error string length

// sim & model, data
std::unique_ptr<mj_app::Simulate> gbSim;
mjModel* m = nullptr;
mjData* d = nullptr;

using Seconds = std::chrono::duration<double>;

//---------------------------------------- plugin handling -----------------------------------------

// return the path to the directory containing the current executable
// used to determine the location of auto-loaded plugin libraries
std::string getExecutableDir() {
#if defined(_WIN32) || defined(__CYGWIN__)
  constexpr char kPathSep = '\\';
  std::string realpath = [&]() -> std::string {
    std::unique_ptr<char[]> realpath(nullptr);
    DWORD buf_size = 128;
    bool success = false;
    while (!success) {
      realpath.reset(new (std::nothrow) char[buf_size]);
      if (!realpath) {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }

      DWORD written = GetModuleFileNameA(nullptr, realpath.get(), buf_size);
      if (written < buf_size) {
        success = true;
      } else if (written == buf_size) {
        // realpath is too small, grow and retry
        buf_size *= 2;
      } else {
        std::cerr << "failed to retrieve executable path: " << GetLastError() << "\n";
        return "";
      }
    }
    return realpath.get();
  }();
#else
  constexpr char kPathSep = '/';
#if defined(__APPLE__)
  std::unique_ptr<char[]> buf(nullptr);
  {
    std::uint32_t buf_size = 0;
    _NSGetExecutablePath(nullptr, &buf_size);
    buf.reset(new char[buf_size]);
    if (!buf) {
      std::cerr << "cannot allocate memory to store executable path\n";
      return "";
    }
    if (_NSGetExecutablePath(buf.get(), &buf_size)) {
      std::cerr << "unexpected error from _NSGetExecutablePath\n";
    }
  }
  const char* path = buf.get();
#else
  const char* path = "/proc/self/exe";
#endif
  std::string realpath = [&]() -> std::string {
    std::unique_ptr<char[]> realpath(nullptr);
    std::uint32_t buf_size = 128;
    bool success = false;
    while (!success) {
      realpath.reset(new (std::nothrow) char[buf_size]);
      if (!realpath) {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }

      std::size_t written = readlink(path, realpath.get(), buf_size);
      if (written < buf_size) {
        realpath.get()[written] = '\0';
        success = true;
      } else if (written == -1) {
        if (errno == EINVAL) {
          // path is already not a symlink, just use it
          return path;
        }

        std::cerr << "error while resolving executable path: " << strerror(errno) << '\n';
        return "";
      } else {
        // realpath is too small, grow and retry
        buf_size *= 2;
      }
    }
    return realpath.get();
  }();
#endif

  if (realpath.empty()) {
    return "";
  }

  for (std::size_t i = realpath.size() - 1; i > 0; --i) {
    if (realpath.c_str()[i] == kPathSep) {
      return realpath.substr(0, i);
    }
  }

  // don't scan through the entire file system's root
  return "";
}

// scan for libraries in the plugin directory to load additional plugins
void scanPluginLibraries() {
  // check and print plugins that are linked directly into the executable
  int nplugin = mjp_pluginCount();
  if (nplugin) {
    std::printf("Built-in plugins:\n");
    for (int i = 0; i < nplugin; ++i) {
      std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
    }
  }

  // define platform-specific strings
#if defined(_WIN32) || defined(__CYGWIN__)
  const std::string sep = "\\";
#else
  const std::string sep = "/";
#endif

  // try to open the ${EXECDIR}/MUJOCO_PLUGIN_DIR directory
  // ${EXECDIR} is the directory containing the simulate binary itself
  // MUJOCO_PLUGIN_DIR is the MUJOCO_PLUGIN_DIR preprocessor macro
  const std::string executable_dir = getExecutableDir();
  if (executable_dir.empty()) {
    return;
  }

  const std::string plugin_dir = getExecutableDir() + sep + MUJOCO_PLUGIN_DIR;
  mj_loadAllPluginLibraries(
      plugin_dir.c_str(), +[](const char* filename, int first, int count) {
        std::printf("Plugins registered by library '%s':\n", filename);
        for (int i = first; i < first + count; ++i) {
          std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
        }
      });
}

//------------------------------------------- simulation -------------------------------------------

const char* Diverged(int disableflags, const mjData* d) {
  if (disableflags & mjDSBL_AUTORESET) {
    for (mjtWarning w : {mjWARN_BADQACC, mjWARN_BADQVEL, mjWARN_BADQPOS}) {
      if (d->warning[w].number > 0) {
        return mju_warningText(w, d->warning[w].lastinfo);
      }
    }
  }
  return nullptr;
}

mjModel* LoadModel(const char* file, mj_app::Simulate& sim) {
  // this copy is needed so that the mju::strlen call below compiles
  char filename[mj_app::Simulate::kMaxFilenameLength];
  mju::strcpy_arr(filename, file);

  // make sure filename is not empty
  if (!filename[0]) {
    return nullptr;
  }

  // load and compile
  char loadError[kErrorLength] = "";
  mjModel* mnew = 0;
  auto load_start = mj_app::Simulate::Clock::now();
  if (mju::strlen_arr(filename) > 4 &&
      !std::strncmp(filename + mju::strlen_arr(filename) - 4, ".mjb",
                    mju::sizeof_arr(filename) - mju::strlen_arr(filename) + 4)) {
    mnew = mj_loadModel(filename, nullptr);
    if (!mnew) {
      mju::strcpy_arr(loadError, "could not load binary model");
    }
  } else {
    mnew = mj_loadXML(filename, nullptr, loadError, kErrorLength);

    // remove trailing newline character from loadError
    if (loadError[0]) {
      int error_length = mju::strlen_arr(loadError);
      if (loadError[error_length - 1] == '\n') {
        loadError[error_length - 1] = '\0';
      }
    }
  }
  auto load_interval = mj_app::Simulate::Clock::now() - load_start;
  double load_seconds = Seconds(load_interval).count();

  if (!mnew) {
    std::printf("%s\n", loadError);
    mju::strcpy_arr(sim.load_error, loadError);
    return nullptr;
  }

  // compiler warning: print and pause
  else if (load_seconds > 0.25) {
    // mj_forward() below will print the warning message
    std::printf("Model compiled, but simulation warning (paused):\n  %s\n", loadError);
    sim.run = 0;
  }

  // if no error and load took more than 1/4 seconds, report load time
  if (!loadError[0] && load_seconds > 0.25) {
    mju::sprintf_arr(loadError, "Model loaded in %.2g seconds", load_seconds);
  }

  mju::strcpy_arr(sim.load_error, loadError);
  return mnew;
}

void StepModel(mj_app::Simulate& sim, mjModel* m, mjData* d, bool kinematics_only = false) {
  if (kinematics_only) {
    mj_kinematics(m, d);
    mj_comPos(m, d);
  } else {
    mj_step(m, d);
  }
}

void PostLoadModel(mj_app::Simulate& sim, mjModel* m, mjData* d) {
  // 1- Forward model once
  mj_forward(m, d);

  // 2- Init scene (eg: moving mocaps to init pose)
  sim.PosLoadInit(m, d);
}

// simulate in background thread (while rendering in main thread)
void PhysicsLoop(mj_app::Simulate& sim) {
  // cpu-sim syncronization point
  std::chrono::time_point<mj_app::Simulate::Clock> syncCPU;
  mjtNum syncSim = 0;

  // run until asked to exit
  while (!sim.exitrequest.load()) {
    if (sim.droploadrequest.load()) {
      sim.LoadMessage(sim.dropfilename);
      mjModel* mnew = LoadModel(sim.dropfilename, sim);
      sim.droploadrequest.store(false);

      mjData* dnew = nullptr;
      if (mnew) dnew = mj_makeData(mnew);
      if (dnew) {
        sim.Load(mnew, dnew, sim.dropfilename);

        // lock the sim mutex
        const MutexLock lock(sim.mtx);

        mj_deleteData(d);
        mj_deleteModel(m);

        m = mnew;
        d = dnew;
        PostLoadModel(sim, m, d);
      } else {
        sim.LoadMessageClear();
      }
    }  // end drop-load request

    if (sim.uiloadrequest.load()) {
      sim.uiloadrequest.fetch_sub(1);
      sim.LoadMessage(sim.filename);
      mjModel* mnew = LoadModel(sim.filename, sim);
      mjData* dnew = nullptr;
      if (mnew) dnew = mj_makeData(mnew);
      if (dnew) {
        sim.Load(mnew, dnew, sim.filename);

        // lock the sim mutex
        const MutexLock lock(sim.mtx);

        mj_deleteData(d);
        mj_deleteModel(m);

        m = mnew;
        d = dnew;
        PostLoadModel(sim, m, d);
      } else {
        sim.LoadMessageClear();
      }
    }  // end UI-load request

    // sleep for 1 ms or yield, to let main thread run
    //  yield results in busy wait - which has better timing but kills battery life
    if (sim.run && sim.busywait) {
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    {
      // lock the sim mutex
      const MutexLock lock(sim.mtx);

      // run only if model is present
      if (m) {
        // running
        if (sim.run) {
          // Run
          bool stepped = false;

          // record cpu time at start of iteration
          const auto startCPU = mj_app::Simulate::Clock::now();

          // elapsed CPU and simulation time since last sync
          const auto elapsedCPU = startCPU - syncCPU;
          double elapsedSim = d->time - syncSim;

          // requested slow-down factor
          double slowdown = 100 / sim.percentRealTime[sim.real_time_index];

          // misalignment condition: distance from target sim time is bigger than syncmisalign
          bool misaligned = std::abs(Seconds(elapsedCPU).count() / slowdown - elapsedSim) > syncMisalign;

          // out-of-sync (for any reason): reset sync times, step
          if (elapsedSim < 0 || elapsedCPU.count() < 0 || syncCPU.time_since_epoch().count() == 0 ||
              misaligned || sim.speed_changed) {
            // re-sync
            syncCPU = startCPU;
            syncSim = d->time;
            sim.speed_changed = false;

            // run single step, let next iteration deal with timing
            StepModel(sim, m, d);
            const char* message = Diverged(m->opt.disableflags, d);
            if (message) {
              sim.run = 0;
              mju::strcpy_arr(sim.load_error, message);
            } else {
              stepped = true;
            }
          }

          // in-sync: step until ahead of cpu
          else {
            bool measured = false;
            mjtNum prevSim = d->time;

            double refreshTime = simRefreshFraction / sim.refresh_rate;

            // step while sim lags behind cpu and within refreshTime
            while (Seconds((d->time - syncSim) * slowdown) < mj_app::Simulate::Clock::now() - syncCPU &&
                   mj_app::Simulate::Clock::now() - startCPU < Seconds(refreshTime)) {
              // measure slowdown before first step
              if (!measured && elapsedSim) {
                sim.measured_slowdown = std::chrono::duration<double>(elapsedCPU).count() / elapsedSim;
                measured = true;
              }

              // call mj_step
              StepModel(sim, m, d);
              const char* message = Diverged(m->opt.disableflags, d);
              if (message) {
                sim.run = 0;
                mju::strcpy_arr(sim.load_error, message);
              } else {
                stepped = true;
              }

              // break if reset
              if (d->time < prevSim) {
                break;
              }
            }
          }

          // save current state to history buffer
          if (stepped) {
            sim.AddToHistory();
          }
        }

        // paused
        else {
          // run mj_forward, to update rendering and joint sliders
          mj_forward(m, d);
          sim.speed_changed = true;
        }
      }
    }  // release std::lock_guard<std::mutex>
  }
}

// --------------------------------- callbacks ---------------------------------
// controller
extern "C" {
void controller(const mjModel* m, mjData* d);
}

// controller callback
void controller(const mjModel* m, mjData* d) {
  // NOTE: This happens when multiple XML loading or direct model compilation are done, creating multiple
  // [m,d], while there is only a single [mjcb_control]
  if (d != mj_app::d) {
    return;
  }
  gbSim->VisualizeGrasps(m, d);
}

//-------------------------------------- physics_thread --------------------------------------------

void PhysicsThread(mj_app::Simulate* sim, const char* filename) {
  // Request loadmodel if file given (otherwise drag-and-drop)
  if (filename != nullptr) {
    sim->LoadMessage(filename);
    m = LoadModel(filename, *sim);
  } else {
    // Programmingly construct [m]
    m = sim->ConstructCustomModel();
  }

  // Create [d] from loaded/constructed [m]
  if (m) {
    // lock the sim mutex
    const MutexLock lock(sim->mtx);
    d = mj_makeData(m);
  }

  // Load [m,d] into [sim], initiating it
  if (d) {
    sim->Load(m, d, filename);

    // lock the sim mutex
    const MutexLock lock(sim->mtx);
    PostLoadModel(*sim, m, d);
  } else {
    sim->LoadMessageClear();
  }

  PhysicsLoop(*sim);

  // delete everything we allocated
  mj_deleteData(d);
  mj_deleteModel(m);
}
}  // namespace mj_app

//------------------------------------------ main --------------------------------------------------

// machinery for replacing command line error by a macOS dialog box when running under Rosetta
#if defined(__APPLE__) && defined(__AVX__)
extern void DisplayErrorDialogBox(const char* title, const char* msg);
static const char* rosetta_error_msg = nullptr;
__attribute__((used, visibility("default"))) extern "C" void _mj_rosettaError(const char* msg) {
  rosetta_error_msg = msg;
}
#endif

// run event loop
int main(int argc, char** argv) {
  // display an error if running on macOS under Rosetta 2
#if defined(__APPLE__) && defined(__AVX__)
  if (rosetta_error_msg) {
    DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
    std::exit(1);
  }
#endif

  // print version, check compatibility
  std::printf("MuJoCo version %s\n", mj_versionString());
  std::printf("Current dir: %s\n", std::filesystem::current_path().string().c_str());
  if (mjVERSION_HEADER != mj_version()) {
    mju_error("Headers and library have different versions");
  }

  // scan for libraries in the plugin directory to load additional plugins
  mj_app::scanPluginLibraries();

  mjvCamera cam;
  mjv_defaultCamera(&cam);

  mjvOption opt;
  mjv_defaultOption(&opt);

  mjvPerturb pert;
  mjv_defaultPerturb(&pert);

  // simulate object encapsulates the UI
  mj_app::gbSim =
      std::make_unique<mj_app::Simulate>(std::make_unique<mujoco::GlfwAdapter>(), &cam, &opt, &pert,
                                         /* is_passive = */ false);
  // set control callback
  mjcb_control = mj_app::controller;

  const char* filename = nullptr;
  if (argc > 1) {
    filename = argv[1];
  }

  // start physics thread
  std::thread physicsthreadhandle(&mj_app::PhysicsThread, mj_app::gbSim.get(), filename);

  // start simulation UI loop (blocking call)
  mj_app::gbSim->InitializeRenderLoop();
  mj_app::gbSim->RenderLoop();
  physicsthreadhandle.join();

  mj_app::gbSim.release();
  return 0;
}
