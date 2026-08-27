#include "ppo.hpp"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/classes/object.hpp>
#include <mlpack/methods/ann/layer/linear.hpp>
#include <mlpack/methods/ann/layer/softmax.hpp>

using namespace godot;

PPO::PPO() {}
PPO::~PPO() {}

void PPO::_bind_methods() {
    ClassDB::bind_method(D_METHOD("build", "actor_layers", "critic_layers"), &PPO::build);
    ClassDB::bind_method(D_METHOD("set_environment", "env"), &PPO::set_environment);
    ClassDB::bind_method(D_METHOD("train", "total_timesteps", "rollout_steps", "epochs_per_update",
                                   "minibatch_size", "clip_eps", "gamma", "gae_lambda",
                                   "actor_lr", "critic_lr", "print_loss", "print_every"),
                          &PPO::train,
                          DEFVAL(128), DEFVAL(4), DEFVAL(32), DEFVAL(0.2),
                          DEFVAL(0.99), DEFVAL(0.95), DEFVAL(0.0003), DEFVAL(0.001),
                          DEFVAL(false), DEFVAL(1));
    ClassDB::bind_method(D_METHOD("get_action", "state"), &PPO::get_action);
    ClassDB::bind_method(D_METHOD("save_actor", "actor_path"), &PPO::save_actor);
    ClassDB::bind_method(D_METHOD("save_critic", "critic_path"), &PPO::save_critic);
    ClassDB::bind_method(D_METHOD("load_actor", "actor_path"), &PPO::load_actor);
    ClassDB::bind_method(D_METHOD("load_critic", "critic_path"), &PPO::load_critic);
}

void PPO::build_ffn_layers(auto &network, const Array &layer_dicts) {
    for (int i = 0; i < layer_dicts.size(); ++i) {
        Dictionary d = layer_dicts[i];
        auto layer_id = static_cast<LAYER>(static_cast<int>(d.get("layer_id", -1)));
        auto out_dims = static_cast<size_t>(static_cast<int64_t>(d.get("out_dims", 0)));

        if (layer_id == LAYER::LINEAR_LAYER) {
            network.template Add<mlpack::Linear<>>(out_dims);
        } else if (layer_id == LAYER::ACTIVATION_LAYER) {
            auto act = static_cast<ACTIVATION>(static_cast<int>(d.get("activation", 0)));
            switch (act) {
                case ACTIVATION::RELU: network.template Add<mlpack::ReLU<>>(); break;
                case ACTIVATION::TANH: network.template Add<mlpack::TanH<>>(); break;
                case ACTIVATION::SOFTMAX: network.template Add<mlpack::Softmax<>>(); break;
                default: break;
            }
        }
    }
}

bool PPO::build(const Array &actor_layers, const Array &critic_layers) {
    build_ffn_layers(actor, actor_layers);
    build_ffn_layers(critic, critic_layers);
    built = true;
    return true;
}

void PPO::set_environment(Node *p_env) {
    env = Object::cast_to<PPOEnvironment>(p_env);
    if (env == nullptr) {
        UtilityFunctions::print("PPO::set_environment: node is not a PPOEnvironment");
        env_instance_id = 0;
        return;
    }
    env_instance_id = env->get_instance_id();
}

bool PPO::env_is_valid() const {
    if (env == nullptr || env_instance_id == 0) {
        return false;
    }
    return UtilityFunctions::is_instance_id_valid(static_cast<int64_t>(env_instance_id));
}

arma::vec PPO::forward_actor_probs(const arma::vec &state) {
    arma::mat out;
    actor.Predict(state, out);
    return out.col(0);
}

int64_t PPO::get_action(const PackedFloat32Array &state) {
    arma::vec s(static_cast<size_t>(state.size()));
    for (int i = 0; i < state.size(); ++i) s.at(static_cast<size_t>(i)) = state[i];

    arma::vec probs = forward_actor_probs(s);
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng);
}

void PPO::train(int total_timesteps, int rollout_steps, int epochs_per_update,
                 int minibatch_size, double clip_eps, double gamma, double gae_lambda,
                 double actor_lr, double critic_lr, bool print_loss, int print_every) {
    if (!built || !env_is_valid()) {
        UtilityFunctions::print("PPO::train called before build()/set_environment(), or environment was freed");
        return;
    }

    size_t state_dims = static_cast<size_t>(env->get_state_dims());
    int collected = 0;
    int update = 0;

    PackedFloat32Array cur_state = env->reset();

    while (collected < total_timesteps) {
        if (!env_is_valid()) {
            UtilityFunctions::print("PPO::train: environment was freed mid-training, stopping");
            return;
        }

        arma::mat states(state_dims, static_cast<arma::uword>(rollout_steps));
        std::vector<int> actions(static_cast<size_t>(rollout_steps));
        std::vector<double> log_probs(static_cast<size_t>(rollout_steps));
        std::vector<double> rewards(static_cast<size_t>(rollout_steps));
        std::vector<double> values(static_cast<size_t>(rollout_steps));
        std::vector<bool> dones(static_cast<size_t>(rollout_steps));

        for (int t = 0; t < rollout_steps; ++t) {
            if (!env_is_valid()) {
                UtilityFunctions::print("PPO::train: environment was freed mid-rollout, stopping");
                return;
            }

            arma::vec s(state_dims);
            for (size_t i = 0; i < state_dims; ++i) s.at(i) = cur_state[static_cast<int>(i)];
            states.col(static_cast<arma::uword>(t)) = s;

            arma::vec probs = forward_actor_probs(s);
            std::discrete_distribution<int> dist(probs.begin(), probs.end());
            int action = dist(rng);
            actions[static_cast<size_t>(t)] = action;
            log_probs[static_cast<size_t>(t)] = std::log(std::max(probs.at(static_cast<size_t>(action)), 1e-8));

            arma::mat v_out;
            critic.Predict(s, v_out);
            values[static_cast<size_t>(t)] = v_out.at(0, 0);

            Dictionary step_result = env->step(action);
            rewards[static_cast<size_t>(t)] = static_cast<double>(step_result.get("reward", 0.0));
            dones[static_cast<size_t>(t)] = static_cast<bool>(step_result.get("done", false));
            cur_state = step_result.get("state", cur_state);

            if (dones[static_cast<size_t>(t)]) cur_state = env->reset();
        }

        std::vector<double> advantages(static_cast<size_t>(rollout_steps)), returns(static_cast<size_t>(rollout_steps));
        double gae = 0.0;
        double next_value = 0.0;
        for (int t = rollout_steps - 1; t >= 0; --t) {
            double mask = dones[static_cast<size_t>(t)] ? 0.0 : 1.0;
            double delta = rewards[static_cast<size_t>(t)] + gamma * next_value * mask - values[static_cast<size_t>(t)];
            gae = delta + gamma * gae_lambda * mask * gae;
            advantages[static_cast<size_t>(t)] = gae;
            returns[static_cast<size_t>(t)] = advantages[static_cast<size_t>(t)] + values[static_cast<size_t>(t)];
            next_value = values[static_cast<size_t>(t)];
        }

        arma::vec adv_vec(advantages);
        double adv_mean = arma::mean(adv_vec);
        double adv_std = arma::stddev(adv_vec) + 1e-8;
        for (auto &a : advantages) a = (a - adv_mean) / adv_std;

        arma::mat actor_target(3, static_cast<arma::uword>(rollout_steps));
        arma::mat critic_target(1, static_cast<arma::uword>(rollout_steps));
        for (int t = 0; t < rollout_steps; ++t) {
            actor_target.at(0, static_cast<arma::uword>(t)) = static_cast<double>(actions[static_cast<size_t>(t)]);
            actor_target.at(1, static_cast<arma::uword>(t)) = log_probs[static_cast<size_t>(t)];
            actor_target.at(2, static_cast<arma::uword>(t)) = advantages[static_cast<size_t>(t)];
            critic_target.at(0, static_cast<arma::uword>(t)) = returns[static_cast<size_t>(t)];
        }

        ens::Adam actor_opt(actor_lr, static_cast<size_t>(minibatch_size), 0.9, 0.999, 1e-8,
                             static_cast<size_t>(rollout_steps * epochs_per_update), 1e-8, true);
        ens::Adam critic_opt(critic_lr, static_cast<size_t>(minibatch_size), 0.9, 0.999, 1e-8,
                              static_cast<size_t>(rollout_steps * epochs_per_update), 1e-8, true);

        GodotLossCallback callback(false, static_cast<size_t>(print_every));
        actor.Train(states, actor_target, actor_opt, callback);
        critic.Train(states, critic_target, critic_opt);

        collected += rollout_steps;
        update++;
        if (print_loss) {
            UtilityFunctions::print("[PPO] update ", update, " timesteps ", collected, " last reward ", rewards.back());
        }
    }
}

bool PPO::save_actor(const String &actor_path) {
    bool state = mlpack::Save(actor_path.utf8().get_data(), actor);
    return state;
}

bool PPO::save_critic(const String &critic_path) {
    bool state = mlpack::Save(critic_path.utf8().get_data(), critic);
    return state;
}

bool PPO::load_actor(const String &actor_path) {
    bool state = mlpack::Load(actor_path.utf8().get_data(), actor);
    return state;
}

bool PPO::load_critic(const String &critic_path) {
    bool state = mlpack::Load(critic_path.utf8().get_data(), critic);
    return state;
}