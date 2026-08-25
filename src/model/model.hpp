#pragma once
#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <memory>
#include "model_backend.hpp"

namespace godot {

class FFN : public Node {
    GDCLASS(FFN, Node)

  protected:
    static void _bind_methods();

  public:
    FFN();
    ~FFN();

    bool build_from_dictionary_array(const Array &layer_dicts, int loss_type);
    void train(const TypedArray<PackedFloat32Array> &inputs, const TypedArray<PackedFloat32Array> &targets,
           bool use_optimizer, int epochs, double learning_rate, double beta1, double beta2, int batch_size,
           bool print_loss, int print_every, double tolerance, bool shuffle);
    PackedFloat32Array predict(const PackedFloat32Array &input);
    bool save_model(const String &path);
    bool load_model(const String &path, int loss_type);

  private:
    std::unique_ptr<IModelBackend> backend;
    size_t input_dims = 0;
    size_t output_dims = 0;
};

} // namespace godot