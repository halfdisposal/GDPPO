#include "environment.hpp"
#include <godot_cpp/core/class_db.hpp>

using namespace godot;

PPOEnvironment::PPOEnvironment() {}
PPOEnvironment::~PPOEnvironment() {}

void PPOEnvironment::_bind_methods() {
    GDVIRTUAL_BIND(_reset);
    GDVIRTUAL_BIND(_step, "action");
    GDVIRTUAL_BIND(_get_state_dims);
    GDVIRTUAL_BIND(_get_action_dims);
}

PackedFloat32Array PPOEnvironment::reset() {
    PackedFloat32Array result;
    GDVIRTUAL_CALL(_reset, result);
    return result;
}

Dictionary PPOEnvironment::step(int64_t action) {
    Dictionary result;
    GDVIRTUAL_CALL(_step, action, result);
    return result;
}

int64_t PPOEnvironment::get_state_dims() {
    int64_t result = 0;
    GDVIRTUAL_CALL(_get_state_dims, result);
    return result;
}

int64_t PPOEnvironment::get_action_dims() {
    int64_t result = 0;
    GDVIRTUAL_CALL(_get_action_dims, result);
    return result;
}
