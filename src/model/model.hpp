/// @file model.hpp
/// @brief FFN: the Godot-facing Node wrapping IModelBackend, exposing
///        build/train/predict/save/load to GDScript.

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

/// Godot Node exposing a runtime-configurable mlpack feedforward network.
/// Build the layer stack once via build_from_dictionary_array, then
/// train/predict/save/load as needed. Not thread-safe against concurrent
/// calls from multiple threads on the same instance.
class FFN : public Node {
    GDCLASS(FFN, Node)

  protected:
    static void _bind_methods();

  public:
    FFN();
    ~FFN();

    /// Constructs the underlying network from an Array of layer
    /// dictionaries (see LayerNode field usage in connections.hpp) and
    /// selects the loss function.
    /// @param loss_type one of LOSS::MSE, LOSS::CROSSENTROPY,
    ///        LOSS::NEGATIVELOGLIKELIHOOD.
    /// @return false if layer_dicts is empty; true otherwise.
    bool build_from_dictionary_array(const Array &layer_dicts, int loss_type);
    

    /// Probably a better high level api for the above function
    bool add(int layer_name, size_t out_dims, int activation_name, 
                    int kernel_w, int kernel_h, int stride_w, int stride_h,
                    int pad_w, int pad_h, bool floor,
                    int input_width, int input_height, int input_channels);
    bool build(int loss_type);

    /// Trains on GDScript-supplied per-sample arrays (one PackedFloat32Array
    /// per sample in inputs/targets, both must be the same length).
    void train(const TypedArray<PackedFloat32Array> &inputs, const TypedArray<PackedFloat32Array> &targets,
           bool use_optimizer, int epochs, double learning_rate, double beta1, double beta2, int batch_size,
           bool print_loss, int print_every, double tolerance, bool shuffle);

    /// Single-sample forward pass. Returns an empty array if called
    /// before build_from_dictionary_array.
    PackedFloat32Array predict(const PackedFloat32Array &input);

    /// Save the model to a given path
    /// Model save format can be of type json, xml, or bin
    /// The save format is infered directly from the provided path
    bool save_model(const String &path);

    /// Reconstructs an empty network of the given loss type, then loads
    /// weights from disk -- loss_type must match what the file was saved
    /// with, or deserialization will fail or silently mismatch shapes.
    bool load_model(const String &path, int loss_type);

    String summary();
    /// Flattens an Image into the column-major (width-fastest, then
    /// height, then channel) order mlpack's Convolution/InputDimensions()
    /// expects -- NOT the same order as Image::get_data().
    /// @param channels 1 (grayscale/L8), 3 (RGB), or 4 (RGBA).
    /// @return empty array if image is null or channels is unsupported.
    static PackedFloat32Array image_to_packedarray(const Ref<Image> &image, int channels);

  private:
    std::unique_ptr<IModelBackend> backend;
    size_t input_dims = 0;
    size_t output_dims = 0;
    Array layer_dict_build = {};
};

} // namespace godot
