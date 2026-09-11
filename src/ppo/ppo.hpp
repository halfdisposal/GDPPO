/// @file ppo.hpp
/// @brief PPO: discrete-action Proximal Policy Optimization trainer.
///        Hand-rolled (mlpack has no built-in PPO) -- see ppo_loss.hpp
///        for the clipped-surrogate gradient, which should be validated
///        against a reference implementation before production use.
#pragma once
#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/array.hpp>
#include "../environment/environment.hpp"
#include "../model/model_backend.hpp"
#include "ppo_loss.hpp"
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <random>



namespace godot {
/// Discrete-action PPO agent. Holds its own actor (policy, softmax
/// output) and critic (value function) FFNs. Call build() then
/// set_environment() before train().
class PPO : public Node {
    GDCLASS(PPO, Node)

  protected:
    static void _bind_methods();

  public:
    PPO();
    ~PPO();

    /// Constructs actor/critic networks from layer dictionaries.
    /// actor_layers must end in a Linear(action_count) + SOFTMAX.
    /// critic_layers must end in a Linear(1) with no output activation.
    bool build(const Array &actor_layers, const Array &critic_layers);
    
    /// Sets the environment used by train(). p_env must be a
    /// PPOEnvironment (checked via cast; non-matching nodes are rejected
    /// with a logged message). Stores env_instance_id to detect the node
    /// being freed mid-training.
    void set_environment(Node *p_env);

    /// Runs PPO for up to total_timesteps environment steps, collecting
    /// rollout_steps transitions per update, updating actor/critic for
    /// epochs_per_update passes each time. Blocks the calling thread --
    /// run on a background Thread to avoid freezing the game
    void train(int total_timesteps, int rollout_steps, int epochs_per_update,
               int minibatch_size, double clip_eps, double gamma, double gae_lambda,
               double actor_lr, double critic_lr, bool print_loss, int print_every);

    /// Samples a discrete action from the current policy given a state.
    int64_t get_action(const PackedFloat32Array &state);

    /// Save the actor of the PPO Agent
    /// @param actor_path can be a json, xml or bin path
    bool save_actor(const String &actor_path);

    /// Save the critic of the PPO Agent
    /// @param critic_path can be a json, xml or bin path
    bool save_critic(const String &critic_path);


    /// Load the actor of the PPO Agent
    /// @param actor_path can be a json, xml or bin path
    bool load_actor(const String &actor_path);

    /// Load the critic of the PPO Agent
    /// @param critic_path can be a json, xml or bin path
    bool load_critic(const String &critic_path);

  private:
    mlpack::FFN<PPOClipLoss, mlpack::RandomInitialization> actor;
    mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization> critic;
    PPOEnvironment *env = nullptr;
    uint64_t env_instance_id = 0;
    std::mt19937 rng{std::random_device{}()};
    bool built = false;

    bool env_is_valid() const;
    void build_ffn_layers(auto &network, const Array &layer_dicts);
    arma::vec forward_actor_probs(const arma::vec &state);
};

} // namespace godot
