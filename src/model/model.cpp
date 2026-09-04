#include "model.hpp"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <exception>

using namespace godot;

FFN::FFN() {}
FFN::~FFN() {}

void FFN::_bind_methods() {
    BIND_ENUM_CONSTANT(NONE);
    BIND_ENUM_CONSTANT(RELU);
    BIND_ENUM_CONSTANT(SIGMOID);
    BIND_ENUM_CONSTANT(TANH);
    BIND_ENUM_CONSTANT(LEAKYRELU);
    BIND_ENUM_CONSTANT(SOFTMAX);
    BIND_ENUM_CONSTANT(LOGSOFTMAX);

    BIND_ENUM_CONSTANT(INPUT_LAYER);
    BIND_ENUM_CONSTANT(OUTPUT_LAYER);
    BIND_ENUM_CONSTANT(ACTIVATION_LAYER);
    BIND_ENUM_CONSTANT(LINEAR_LAYER);
    BIND_ENUM_CONSTANT(CONV2D_LAYER);
    BIND_ENUM_CONSTANT(MAXPOOL2D_LAYER);

    BIND_ENUM_CONSTANT(MSE);
    BIND_ENUM_CONSTANT(CROSSENTROPY);
    BIND_ENUM_CONSTANT(NEGATIVELOGLIKELIHOOD);


    ClassDB::bind_method(D_METHOD("build_from_dictionary_array", "layer_dicts", "loss_type"), &FFN::build_from_dictionary_array);
    ClassDB::bind_method(D_METHOD("train", "inputs", "targets", "use_optimizer", "epochs", "learning_rate", "beta1", "beta2", "batch_size", "print_loss", "print_every", "tolerance", "shuffle"), &FFN::train, DEFVAL(true), DEFVAL(100), DEFVAL(0.001), DEFVAL(0.9), DEFVAL(0.999), DEFVAL(32), DEFVAL(false), DEFVAL(10), DEFVAL(1e-8), DEFVAL(true));
    ClassDB::bind_method(D_METHOD("predict", "input"), &FFN::predict);
    ClassDB::bind_method(D_METHOD("save_model", "path"), &FFN::save_model);
    ClassDB::bind_method(D_METHOD("load_model", "path", "loss_type"), &FFN::load_model);
}

bool FFN::build_from_dictionary_array(const Array &layer_dicts, int loss_type) {
    std::vector<LayerNode> layers;
    layers.reserve(static_cast<size_t>(layer_dicts.size()));

    for (int i = 0; i < layer_dicts.size(); ++i) {
        Dictionary d = layer_dicts[i];
        LayerNode spec;
        spec.layer_id = static_cast<LAYER>(static_cast<int>(d.get("layer_id", -1)));
        spec.in_dims = static_cast<size_t>(static_cast<int64_t>(d.get("in_dims", 0)));
        spec.out_dims = static_cast<size_t>(static_cast<int64_t>(d.get("out_dims", 0)));
        
        spec.kernel_w = static_cast<size_t>(static_cast<int64_t>(d.get("kernal_w", 3)));
        spec.kernel_h = static_cast<size_t>(static_cast<int64_t>(d.get("kernal_h", 3)));
        spec.stride_w = static_cast<size_t>(static_cast<int64_t>(d.get("stride_w", 1)));
        spec.stride_h = static_cast<size_t>(static_cast<int64_t>(d.get("stride_h", 1)));
        spec.pad_w = static_cast<size_t>(static_cast<int64_t>(d.get("pad_w", 0)));
        spec.pad_h = static_cast<size_t>(static_cast<int64_t>(d.get("pad_h", 0)));
        spec.floor = static_cast<bool>(d.get("floor", true));
        spec.input_width = static_cast<size_t>(static_cast<int64_t>(d.get("input_width", 0)));
        spec.input_height = static_cast<size_t>(static_cast<int64_t>(d.get("input_height", 0)));
        spec.input_channels = static_cast<size_t>(static_cast<int64_t>(d.get("input_channels", 0)));
        layers.push_back(spec);
    }

    if (layers.empty()) {
        return false;
    }

    input_dims = layers.front().in_dims;
    output_dims = layers.back().out_dims;

    if (static_cast<LOSS>(loss_type) == LOSS::CROSSENTROPY) {
        backend = std::make_unique<ModelBackend<mlpack::SigmoidCrossEntropyError>>(layers);
    } else if (static_cast<LOSS>(loss_type) == LOSS::NEGATIVELOGLIKELIHOOD) {
        backend = std::make_unique<ModelBackend<mlpack::NegativeLogLikelihood>>(layers);
    } else {
        backend = std::make_unique<ModelBackend<mlpack::MeanSquaredError>>(layers);
    }

    return true;
}

void FFN::train(const TypedArray<PackedFloat32Array> &inputs, const TypedArray<PackedFloat32Array> &targets,
                     bool use_optimizer, int epochs, double learning_rate, double beta1, double beta2, int batch_size,
                     bool print_loss, int print_every, double tolerance, bool shuffle) {
    if (!backend) {
        UtilityFunctions::print("FFN::train called before build");
        return;
    }
    if (inputs.size() == 0 || inputs.size() != targets.size()) {
        UtilityFunctions::print("FFN::train: inputs/targets size mismatch");
        return;
    }

    PackedFloat32Array first_row = inputs[0];
    PackedFloat32Array first_target = targets[0];
    size_t n_features = static_cast<size_t>(first_row.size());
    size_t n_targets = static_cast<size_t>(first_target.size());
    size_t n_samples = static_cast<size_t>(inputs.size());

    arma::mat X(n_features, n_samples);
    arma::mat Y(n_targets, n_samples);

    for (size_t col = 0; col < n_samples; ++col) {
        PackedFloat32Array row = inputs[static_cast<int>(col)];
        PackedFloat32Array target_row = targets[static_cast<int>(col)];
        for (size_t r = 0; r < n_features; ++r) {
            X.at(r, col) = row[static_cast<int>(r)];
        }
        for (size_t r = 0; r < n_targets; ++r) {
            Y.at(r, col) = target_row[static_cast<int>(r)];
        }
    }

    TrainConfig config;
    config.epochs = static_cast<size_t>(epochs);
    config.learning_rate = learning_rate;
    config.beta1 = beta1;
    config.beta2 = beta2;
    config.batch_size = static_cast<size_t>(batch_size);
    config.print_loss = print_loss;
    config.print_every = static_cast<size_t>(print_every);
    config.tolerance = tolerance;
    config.shuffle = shuffle;
    try {
        backend->Train(X, Y, config, use_optimizer);
    } catch (std::exception &e) {
        String e_id(e.what());
        UtilityFunctions::print("MODEL: ", e_id);
    }
} 

PackedFloat32Array FFN::predict(const PackedFloat32Array &input) {
    PackedFloat32Array result;
    if (!backend) {
        UtilityFunctions::print("FFN::predict called before build");
        return result;
    }

    arma::mat in(static_cast<size_t>(input.size()), 1);
    for (int i = 0; i < input.size(); ++i) {
        in.at(static_cast<size_t>(i), 0) = input[i];
    }

    arma::mat out;
    backend->Predict(in, out);

    result.resize(static_cast<int>(out.n_rows));
    for (size_t i = 0; i < out.n_rows; ++i) {
        result[static_cast<int>(i)] = static_cast<float>(out.at(i, 0));
    }

    return result;
}

bool FFN::save_model(const String &path) {
    if (!backend) {
        return false;
    }
    return backend->Save(std::string(path.utf8().get_data()));
}

bool FFN::load_model(const String &path, int loss_type) {
    std::vector<LayerNode> empty_layers;
    if (static_cast<LOSS>(loss_type) == LOSS::CROSSENTROPY) {
        backend = std::make_unique<ModelBackend<mlpack::CrossEntropyError>>(empty_layers);
    } else {
        backend = std::make_unique<ModelBackend<mlpack::MeanSquaredError>>(empty_layers);
    }
    return backend->Load(std::string(path.utf8().get_data()));
}
