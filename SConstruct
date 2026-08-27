#!/usr/bin/env python
import os

env = SConscript("godot-cpp/SConstruct")

# --- Build variables ---------------------------------------------------
# These default to "nothing extra" -- i.e. assume mlpack/ensmallen/cereal/
# armadillo are already discoverable via the compiler's normal search
# paths (installed via a system package manager, MSYS2 pacman, homebrew,
# apt, vcpkg, etc.). Override on the command line if your install lives
# somewhere nonstandard, e.g.:
#   scons platform=windows mlpack_include=C:/msys64/ucrt64/include
opts = Variables([], ARGUMENTS)

opts.Add(PathVariable(
    "mlpack_include",
    "Path to mlpack headers, if not on the default include path",
    "",
))
opts.Add(PathVariable(
    "armadillo_include",
    "Path to Armadillo headers, if not on the default include path",
    "",
))
opts.Add(PathVariable(
    "armadillo_lib",
    "Path to Armadillo/BLAS/LAPACK libraries, if not on the default lib path",
    "",
))

opts.Update(env)
Help(opts.GenerateHelpText(env))

# --- Include paths -------------------------------------------------------
env.Append(CPPPATH=["src/"])
for path_var in ("mlpack_include", "armadillo_include"):
    if env[path_var]:
        env.Append(CPPPATH=[env[path_var]])

env.Append(CPPDEFINES=["MLPACK_ENABLE_ANN_SERIALIZATION"])

if env.get("CXXFLAGS"):
    env.Append(CXXFLAGS=["-std=gnu++20"])

# --- Library paths ---------------------------------------------------------
if env["armadillo_lib"]:
    env.Append(LIBPATH=[env["armadillo_lib"]])
env.Append(LIBS=["armadillo"])

# --- Platform-specific link flags -----------------------------------------
if env["platform"] == "windows":
    env.Append(LINKFLAGS=[
        "-static-libgcc",
        "-static-libstdc++",
        "-Wl,-Bstatic",
        "-lwinpthread",
        "-Wl,-Bdynamic",
        "-Wl,--allow-multiple-definition",
    ])
elif env["platform"] == "linux":
    env.Append(LINKFLAGS=["-Wl,--allow-multiple-definition"])
elif env["platform"] == "macos":
    env.Append(FRAMEWORKS=["Accelerate"])

# --- Sources -------------------------------------------------------------
sources = Glob("src/*.cpp") + \
          Glob("src/model/*.cpp") + \
          Glob("src/environment/*.cpp") + \
          Glob("src/ppo/*.cpp") + \
          Glob("src/input/*.cpp") + \
          Glob("src/output/*.cpp") + \
          Glob("src/linear/*.cpp") + \
          Glob("src/main/*.cpp")

# --- Output ----------------------------------------------------------------
if env["platform"] == "macos":
    library = env.SharedLibrary(
        "addons/GDPPO/bin/libgdppo.{}.{}.framework/libgdppo.{}.{}".format(
            env["platform"], env["target"], env["platform"], env["target"]
        ),
        source=sources,
    )
else:
    library = env.SharedLibrary(
        "addons/GDPPO/bin/libgdppo{}{}".format(env["suffix"], env["SHLIBSUFFIX"]),
        source=sources,
    )

Default(library)
