#include "model.hpp"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/image.hpp>
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
    ClassDB::bind_method(D_METHOD("add", "layer_id", "out_dims", "activation",
                                  "kernel_w", "kernel_h", "stride_w", "stride_h",
                                  "pad_w", "pad_h", "floor",
                                  "input_width", "input_height", "input_channels"), 
                         &FFN::add, 
                         DEFVAL(static_cast<int>(NONE)), DEFVAL(3), DEFVAL(3), DEFVAL(1), DEFVAL(1),
                         DEFVAL(0), DEFVAL(0), DEFVAL(true), DEFVAL(1), DEFVAL(1), DEFVAL(1));
    ClassDB::bind_method(D_METHOD("build", "loss_type"), &FFN::build);
    ClassDB::bind_method(D_METHOD("train", "inputs", "targets", "use_optimizer", "epochs", "learning_rate", "beta1", "beta2", "batch_size", "print_loss", "print_every", "tolerance", "shuffle"), &FFN::train, DEFVAL(true), DEFVAL(100), DEFVAL(0.001), DEFVAL(0.9), DEFVAL(0.999), DEFVAL(32), DEFVAL(false), DEFVAL(10), DEFVAL(1e-8), DEFVAL(true));
    ClassDB::bind_method(D_METHOD("predict", "input"), &FFN::predict);
    ClassDB::bind_method(D_METHOD("save_model", "path"), &FFN::save_model);
    ClassDB::bind_method(D_METHOD("load_model", "path", "loss_type"), &FFN::load_model);
    ClassDB::bind_method(D_METHOD("summary"), &FFN::summary);
    ClassDB::bind_static_method("FFN", D_METHOD("image_to_packedarray", "image", "channels"), &FFN::image_to_packedarray);
}

static bool check_layers_dictionary(const Array &layer_dicts) {
    for (int i = 0; i < layer_dicts.size(); ++i) {
        Dictionary d = layer_dicts[i];
        if (!d.has("layer_id")) {
            return false;
        }
        switch (static_cast<int>(d.get("layer_id", -1))) {
            case LINEAR_LAYER:
                if (!d.has("out_dims")) {
                    return false;
                }
                break;
            case ACTIVATION_LAYER:
                if (!d.has("activation")) {
                    return false;
                }
                break;
            case CONV2D_LAYER:
                if ((!d.has("kernel_w")) || (!d.has("kernel_h")) ||
                    (!d.has("stride_w")) || (!d.has("stride_h")) ||
                    (!d.has("pad_w")) || (!d.has("pad_h")) || (!d.has("out_dims"))
                ) {
                    return false;
                }
                break;
            case MAXPOOL2D_LAYER:
                if ((!d.has("kernel_w")) || (!d.has("kernel_h")) ||
                    (!d.has("stride_w")) || (!d.has("stride_h")) ||
                    (!d.has("floor"))
                ) {
                    return false;
                }
                break;
            case INPUT_LAYER:
            case OUTPUT_LAYER:
                break;
            default:
                return false;
        }
    }
    return true;
}

bool FFN::build_from_dictionary_array(const Array &layer_dicts, int loss_type) {
    if (!check_layers_dictionary(layer_dicts)) {
        UtilityFunctions::print(" build_from_dictionary_array(): invalid layer dictionary array");
        return false;
    }
    std::vector<LayerNode> layers;
    layers.reserve(static_cast<size_t>(layer_dicts.size()));

    for (int i = 0; i < layer_dicts.size(); ++i) {
        Dictionary d = layer_dicts[i];
        LayerNode spec;
        spec.layer_id = static_cast<LAYER>(static_cast<int>(d.get("layer_id", -1)));
        spec.in_dims = static_cast<size_t>(static_cast<int64_t>(d.get("in_dims", 0)));
        spec.out_dims = static_cast<size_t>(static_cast<int64_t>(d.get("out_dims", 0)));
        
        spec.kernel_w = static_cast<size_t>(static_cast<int64_t>(d.get("kernel_w", 3)));
        spec.kernel_h = static_cast<size_t>(static_cast<int64_t>(d.get("kernel_h", 3)));
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

bool FFN::add(int layer_id, size_t out_dims, int activation, 
                    int kernel_w, int kernel_h, int stride_w, int stride_h,
                    int pad_w, int pad_h, bool floor,
                    int input_width, int input_height, int input_channels) {
    Dictionary layer;
    layer["layer_id"] = static_cast<LAYER>(layer_id);
    layer["in_dims"] = 0;
    layer["out_dims"] = out_dims;
    
    if (layer_id == CONV2D_LAYER || layer_id == MAXPOOL2D_LAYER) {
        layer["kernel_w"] = kernel_w;
        layer["kernel_h"] = kernel_h;
        layer["stride_w"] = stride_w;
        layer["stride_h"] = stride_h;
        layer["pad_w"] = pad_w;
        layer["pad_h"] = pad_h;
        layer["floor"] = floor;
        layer["input_width"] = input_width;
        layer["input_height"] = input_height;
        layer["input_channels"] = input_channels;
    }

    FFN::layer_dict_build.append(layer);
    if (activation != NONE) {
        Dictionary activation_layer;
        activation_layer["layer_id"] = ACTIVATION_LAYER;
        activation_layer["activation"] = static_cast<ACTIVATION>(activation);
        FFN::layer_dict_build.append(activation_layer);
    }
    return true;
}

bool FFN::build(int loss_type) {
    if (FFN::layer_dict_build.is_empty()) { 
        UtilityFunctions::print(" build(): Layers are not set for building");
        return false; 
    }
    return FFN::build_from_dictionary_array(FFN::layer_dict_build, loss_type);
}

void FFN::train(const TypedArray<PackedFloat32Array> &inputs, const TypedArray<PackedFloat32Array> &targets,
                     bool use_optimizer, int epochs, double learning_rate, double beta1, double beta2, int batch_size,
                     bool print_loss, int print_every, double tolerance, bool shuffle) {
    if (!backend) {
        UtilityFunctions::print(" train(): train called before build");
        return;
    }
    if (inputs.size() == 0 || inputs.size() != targets.size()) {
        UtilityFunctions::print(" train(): inputs/targets size mismatch");
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
        UtilityFunctions::print(" train(): ", e_id);
    }
} 

PackedFloat32Array FFN::predict(const PackedFloat32Array &input) {
    PackedFloat32Array result;
    if (!backend) {
        UtilityFunctions::print(" predict(): predict called before build");
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
    } else if (static_cast<LOSS>(loss_type) == LOSS::NEGATIVELOGLIKELIHOOD) {
        backend = std::make_unique<ModelBackend<mlpack::NegativeLogLikelihood>>(empty_layers);
    } else {
        backend = std::make_unique<ModelBackend<mlpack::MeanSquaredError>>(empty_layers);
    }
    return backend->Load(std::string(path.utf8().get_data()));
}

String FFN::summary() {
    std::string s = "model\n";
    for (int i = 0; i < FFN::layer_dict_build.size(); ++i) {
        Dictionary d = FFN::layer_dict_build[i];
        int id = static_cast<int>(d["layer_id"]);
        size_t out_dims = static_cast<size_t>(d["out_dims"]);
        if (id == LINEAR_LAYER) {
            s += std::format(" -linear({})\n", out_dims);
        } else if (id == CONV2D_LAYER) {
            size_t kw = static_cast<size_t>(d.get("kernel_w", 3));
            size_t kh = static_cast<size_t>(d.get("kernel_h", 3));
            size_t sw = static_cast<size_t>(d.get("stride_w", 1));
            size_t sh = static_cast<size_t>(d.get("stride_h", 1));
            size_t pw = static_cast<size_t>(d.get("pad_w", 0));
            size_t ph = static_cast<size_t>(d.get("pad_h", 0));
            s += std::format(" -conv2d({}, {}, {}, {}, {}, {})\n", kw, kh, sw, sh, pw, ph);
        } else if (id == MAXPOOL2D_LAYER) {
            size_t kw = static_cast<size_t>(d.get("kernel_w", 3));
            size_t kh = static_cast<size_t>(d.get("kernel_h", 3));
            size_t sw = static_cast<size_t>(d.get("stride_w", 1));
            size_t sh = static_cast<size_t>(d.get("stride_h", 1));
            bool floor = static_cast<bool>(d.get("floor", true));
            s += std::format(" -maxpool2d({}, {}, {}, {}, ", kw, kh, sw, sh);
            if (floor) { s += "true)\n"; }
            else { s += "false)\n"; }
        } else if (id == ACTIVATION_LAYER) {
            int act = d.get("activation", NONE);
            switch (act) {
                case ACTIVATION::RELU:
                    s += " -relu\n";
                    break;
                case ACTIVATION::TANH:
                    s += " -tanh\n";
                    break;
                case ACTIVATION::SOFTMAX:
                    s += " -softmax\n";
                    break;
                case ACTIVATION::SIGMOID:
                    s += " -sigmoid\n";
                    break;
                case ACTIVATION::LEAKYRELU:
                    s += " -leakyrelu\n";
                    break;
                case ACTIVATION::LOGSOFTMAX:
                    s += " -logsoftmax\n";
                    break;
                default:
                    break;
            }
        }
    }
    return String(s.c_str());
}
PackedFloat32Array FFN::image_to_packedarray(const Ref<Image> &image, int channels) {
    PackedFloat32Array result;

    if (image.is_null()) {
        UtilityFunctions::print(" image_to_packedarray(): image is null");
        return result;
    }

    Ref<Image> img = image->duplicate();

    switch (channels) {
        case 1:
            img->convert(Image::FORMAT_L8);
            break;
        case 3:
            img->convert(Image::FORMAT_RGB8);
            break;
        case 4:
            img->convert(Image::FORMAT_RGBA8);
            break;
        default:
            UtilityFunctions::print(" image_to_packedarray(): unsupported channel count ", channels, " (expected 1, 3, or 4)");
            return result;
    }

    int width = img->get_width();
    int height = img->get_height();

    result.resize(static_cast<int64_t>(width) * height * channels);

    for (int ch = 0; ch < channels; ++ch) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                Color pixel = img->get_pixel(x, y);
                float value = 0.0f;
                switch (ch) {
                    case 0: value = pixel.r; break;
                    case 1: value = pixel.g; break;
                    case 2: value = pixel.b; break;
                    case 3: value = pixel.a; break;
                    default: value = 0.0f;
                }
                // Column-major cube layout: width fastest, height next, channel slowest.
                int index = x + y * width + ch * width * height;
                result[index] = value;
            }
        }
    }

    return result;
}

