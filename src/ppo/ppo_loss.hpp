#pragma once

#include <armadillo>
#include <cmath>
#include <algorithm>

namespace godot {

class PPOClipLoss {
  public:
    explicit PPOClipLoss(double p_clip_eps = 0.2) : clip_eps(p_clip_eps) {}

    template <typename PredictionType, typename TargetType>
    double Forward(const PredictionType &pred, const TargetType &target) {
        double total = 0.0;
        for (size_t col = 0; col < pred.n_cols; ++col) {
            int action = static_cast<int>(target.at(0, col));
            double old_log_prob = target.at(1, col);
            double advantage = target.at(2, col);

            double prob = std::max(pred.at(static_cast<size_t>(action), col), 1e-8);
            double ratio = prob / std::exp(old_log_prob);
            double unclipped = ratio * advantage;
            double clipped = std::clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage;

            total += -std::min(unclipped, clipped); // negative: optimizer minimizes
        }
        return total / static_cast<double>(pred.n_cols);
    }

    template <typename PredictionType, typename TargetType, typename OutputType>
    void Backward(const PredictionType &pred, const TargetType &target, OutputType &output) {
        output.zeros(arma::size(pred));
        double n = static_cast<double>(pred.n_cols);

        for (size_t col = 0; col < pred.n_cols; ++col) {
            int action = static_cast<int>(target.at(0, col));
            double old_log_prob = target.at(1, col);
            double advantage = target.at(2, col);

            double prob = std::max(pred.at(static_cast<size_t>(action), col), 1e-8);
            double pi_old = std::exp(old_log_prob);
            double ratio = prob / pi_old;

            double unclipped = ratio * advantage;
            double clipped_ratio = std::clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps);
            double clipped = clipped_ratio * advantage;

            // Subgradient: only the "active" (minimum) branch contributes,
            // and the clipped branch contributes 0 once clamped (its
            // derivative w.r.t. ratio is 0 outside the clip window).
            double d_surrogate_d_ratio = 0.0;
            if (unclipped <= clipped) {
                d_surrogate_d_ratio = advantage;
            } else if (ratio > (1.0 - clip_eps) && ratio < (1.0 + clip_eps)) {
                d_surrogate_d_ratio = advantage; // inside clip window, clipped branch tracks ratio
            }
            // else: ratio clamped and clipped branch is active -> 0 gradient

            double d_ratio_d_prob = 1.0 / pi_old;
            double grad = -(d_surrogate_d_ratio * d_ratio_d_prob) / n;

            output.at(static_cast<size_t>(action), col) = grad;
        }
    }

  private:
    double clip_eps;
};

} // namespace godot
