#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/core/gdvirtual.gen.inc>

/// @file environment.hpp
/// @brief PPOEnvironment: GDScript-overridable RL environment base class.

namespace godot {

/// Base class for RL environments. Subclass from GDScript and override
/// the four virtuals below to define an environment's dynamics(keep
/// them pure state math (no scene-tree/node access) if training will run
/// on a background thread (see PPO::train))
class PPOEnvironment : public Node {
    GDCLASS(PPOEnvironment, Node)

  protected:
    static void _bind_methods();

  public:
    PPOEnvironment();
    ~PPOEnvironment();

    /// Override: reset to an initial state, return it.
    GDVIRTUAL0R(PackedFloat32Array, _reset);
    
    /// Override: apply a discrete action, return
    /// {"state": PackedFloat32Array, "reward": float, "done": bool}.
    GDVIRTUAL1R(Dictionary, _step, int64_t);

    /// Override: return the state vector's length.
    GDVIRTUAL0R(int64_t, _get_state_dims);

    /// Override: return the number of discrete actions.
    GDVIRTUAL0R(int64_t, _get_action_dims);

    /// C++-side callers of the above GDVIRTUALs (returns defaults if
    /// the script hasn't overridden the corresponding virtual).
    PackedFloat32Array reset();
    Dictionary step(int64_t action);
    int64_t get_state_dims();
    int64_t get_action_dims();
};

} // namespace godot
