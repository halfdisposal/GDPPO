#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/core/gdvirtual.gen.inc>

namespace godot {

class PPOEnvironment : public Node {
    GDCLASS(PPOEnvironment, Node)

  protected:
    static void _bind_methods();

  public:
    PPOEnvironment();
    ~PPOEnvironment();

    GDVIRTUAL0R(PackedFloat32Array, _reset);
    GDVIRTUAL1R(Dictionary, _step, int64_t); // returns {"state":.., "reward":.., "done":..}
    GDVIRTUAL0R(int64_t, _get_state_dims);
    GDVIRTUAL0R(int64_t, _get_action_dims); // number of discrete actions

    PackedFloat32Array reset();
    Dictionary step(int64_t action);
    int64_t get_state_dims();
    int64_t get_action_dims();
};

} // namespace godot
