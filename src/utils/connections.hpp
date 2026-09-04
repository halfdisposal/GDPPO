#pragma once
#include <godot_cpp/classes/graph_edit.hpp>
#include <godot_cpp/classes/graph_node.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/core/binder_common.hpp>

namespace godot {
    enum ACTIVATION {
        NONE,
        RELU,
        SIGMOID,
        TANH,
        LEAKYRELU,
        SOFTMAX,
        LOGSOFTMAX,
    };
    enum LAYER {
        INPUT_LAYER,
        OUTPUT_LAYER,
        ACTIVATION_LAYER,
        LINEAR_LAYER,
        CONV2D_LAYER,
        MAXPOOL2D_LAYER,
    };
    enum LOSS {
        MSE,
        CROSSENTROPY,
        NEGATIVELOGLIKELIHOOD,
    };

    struct LayerNode {
        LAYER layer_id;
        size_t in_dims = 0;
        size_t out_dims = 0;
        ACTIVATION activation = ACTIVATION::NONE;

        size_t kernel_w = 3, kernel_h = 3;
        size_t stride_w = 1, stride_h = 1;
        size_t pad_w = 0, pad_h = 0;
        bool floor = true;
        size_t input_width = 0, input_height = 0, input_channels = 1;
    };
}

VARIANT_ENUM_CAST(godot::ACTIVATION)
VARIANT_ENUM_CAST(godot::LAYER)
VARIANT_ENUM_CAST(godot::LOSS)
