#pragma once
#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/linear.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/loss_functions/sigmoid_cross_entropy_error.hpp>
#include <mlpack/methods/ann/loss_functions/negative_log_likelihood.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/layer/linear.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/layer/leaky_relu.hpp>
#include <mlpack/methods/ann/layer/softmax.hpp>
#include <mlpack/methods/ann/layer/log_softmax.hpp>
#include <string>
#include <vector>
#include "../utils/connections.hpp"

namespace godot {
class GodotLossCallback {
  public:
    GodotLossCallback(bool p_enabled, size_t p_print_every)
        : enabled(p_enabled), print_every(p_print_every == 0 ? 1 : p_print_every) {}

    template <typename OptimizerType, typename FunctionType, typename MatType>
    bool EndEpoch(OptimizerType & /*optimizer*/, FunctionType & /*function*/,
                  const MatType & /*coordinates*/, size_t epoch, double objective) {
        if (enabled && (epoch % print_every == 0)) {
            UtilityFunctions::print("[AirNN] epoch ", static_cast<int64_t>(epoch), " loss ", objective);
        }
        return true;
    }

  private:
    bool enabled;
    size_t print_every;
};

struct TrainConfig {
    size_t epochs = 100;
    double learning_rate = 0.001; // Adam default
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;
    size_t batch_size = 32;
    bool print_loss = false;
    size_t print_every = 10;
    double tolerance = 1e-8;
    bool shuffle = true;
};

class IModelBackend {
  public:
    virtual ~IModelBackend() = default;
    virtual void Train(const arma::mat &inputs, const arma::mat &targets, const TrainConfig &config, const bool use_optimizer) = 0;
    virtual void Predict(const arma::mat &input, arma::mat &output) = 0;
    virtual bool Save(const std::string &path) = 0;
    virtual bool Load(const std::string &path) = 0;
};
template <typename LossT>
class ModelBackend: public IModelBackend {
    public:
    explicit ModelBackend(const std::vector<LayerNode> &layers) {
        for (size_t i = 0; i < layers.size(); ++i) {
            const LayerNode &layer = layers.at(i);

            if (layer.layer_id == LAYER::LINEAR_LAYER) {
                network.template Add<mlpack::Linear<>>(layer.out_dims);
            } else if (layer.layer_id == LAYER::ACTIVATION_LAYER) {
                switch (layer.activation) {
                    case godot::ACTIVATION::RELU:
                        network.template Add<mlpack::ReLU<>>();
                        break;
                    case godot::ACTIVATION::SIGMOID:
                        network.template Add<mlpack::Sigmoid<>>();
                        break;
                    case godot::ACTIVATION::TANH:
                        network.template Add<mlpack::TanH<>>();
                        break;
                    case godot::ACTIVATION::LEAKYRELU:
                        network.template Add<mlpack::LeakyReLU<>>();
                        break;
                    case godot::ACTIVATION::SOFTMAX:
                        network.template Add<mlpack::Softmax<>>();
                        break;
                    case godot::ACTIVATION::LOGSOFTMAX:
                        network.template Add<mlpack::LogSoftMax<>>();
                        break;
                    case godot::ACTIVATION::NONE:
                    default:
                        break;
                }
            }
        }
    }
    void Train(const arma::mat &inputs, const arma::mat &targets, const TrainConfig &config, const bool use_optimizer) override {
        if (use_optimizer) {
            ens::Adam optimizer(
                config.learning_rate,
                config.batch_size,
                config.beta1,
                config.beta2,
                config.eps,
                inputs.n_cols * config.epochs,
                config.tolerance,
                config.shuffle
            );
            GodotLossCallback callback(config.print_loss, config.print_every);

            network.Train(inputs, targets, optimizer, callback);
        } else {
            network.Train(inputs, targets);
        }
    }
    void Predict(const arma::mat &input, arma::mat &output) override {
        network.Predict(input, output);
    }
    bool Save(const std::string &path) override {
        bool success = mlpack::data::Save(path, "airnn_model", network);
        UtilityFunctions::print("Model Saved to ", String(path.c_str()));
        return success;
    }

    bool Load(const std::string &path) override {
        bool success = mlpack::data::Load(path, "airnn_model", network);
        UtilityFunctions::print("Model Loaded from ", String(path.c_str()));
        return success;
    }
    private:
    mlpack::FFN<LossT, mlpack::RandomInitialization> network;
};

}
