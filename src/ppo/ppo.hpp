#pragma once

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/array.hpp>
#include "../environment/environment.hpp"
#include "../model/model_backend.hpp"
#include "ppo_loss.hpp"
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <random>

namespace godot {

class PPO : public RefCounted {
    GDCLASS(PPO, RefCounted)

  protected:
    static void _bind_methods();

  public:
    PPO();
    ~PPO();

    bool build(const Array &actor_layers, const Array &critic_layers);
    void set_environment(Ref<PPOEnvironment> p_env);
    void train(int total_timesteps, int rollout_steps, int epochs_per_update,
               int minibatch_size, double clip_eps, double gamma, double gae_lambda,
               double actor_lr, double critic_lr, bool print_loss, int print_every);
    int64_t get_action(const PackedFloat32Array &state);

  private:
    mlpack::FFN<PPOClipLoss, mlpack::RandomInitialization> actor;
    mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization> critic;
    Ref<PPOEnvironment> env;
    std::mt19937 rng{std::random_device{}()};
    bool built = false;

    void build_ffn_layers(auto &network, const Array &layer_dicts);
    arma::vec forward_actor_probs(const arma::vec &state);
};

} // namespace godot
