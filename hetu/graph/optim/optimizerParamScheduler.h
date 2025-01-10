#pragma once

#include <iostream>
#include <cmath>
#include <cassert>
#include <string>
#include <vector>
#include <unordered_map>
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/variable.h"

namespace hetu {
namespace graph {

class OptimizerParamScheduler {
public:
    OptimizerParamScheduler() {}
    
    OptimizerParamScheduler(double init_lr, double max_lr, double min_lr,
        int64_t lr_warmup_steps, int64_t lr_decay_steps, std::string lr_decay_style,
        double start_wd, double end_wd, int wd_incr_steps, std::string wd_incr_style) 
        : init_lr(init_lr), max_lr(max_lr), min_lr(min_lr),
        lr_warmup_steps(lr_warmup_steps), lr_decay_steps(lr_decay_steps), lr_decay_style(lr_decay_style),
        start_wd(start_wd), end_wd(end_wd), wd_incr_steps(wd_incr_steps), wd_incr_style(wd_incr_style){

        // 参数检查
        HT_ASSERT(min_lr >= 0.0);
        HT_ASSERT(max_lr >= min_lr);
        HT_ASSERT(init_lr <= max_lr);
        HT_ASSERT(lr_decay_steps > 0);
        HT_ASSERT(lr_warmup_steps < lr_decay_steps);
        HT_ASSERT(start_wd >= 0.0);
        HT_ASSERT(end_wd >= start_wd);

    }

    OptimizerParamScheduler(const OptimizerParamScheduler& other)
        : init_lr(other.init_lr), max_lr(other.max_lr), min_lr(other.min_lr),
          lr_warmup_steps(other.lr_warmup_steps), lr_decay_steps(other.lr_decay_steps),
          lr_decay_style(other.lr_decay_style), start_wd(other.start_wd),
          end_wd(other.end_wd), wd_incr_steps(other.wd_incr_steps),
          wd_incr_style(other.wd_incr_style) {}


    double get_wd(int64_t num_steps) const {

        if (num_steps > wd_incr_steps) return end_wd;

        if (wd_incr_style == "constant") {
            assert(start_wd == end_wd);
            return end_wd;
        }

        double incr_ratio = static_cast<double>(num_steps) / wd_incr_steps;
        assert(incr_ratio >= 0.0 && incr_ratio <= 1.0);
        double delta_wd = end_wd - start_wd;

        double coeff;
        if (wd_incr_style == "linear") {
            coeff = incr_ratio;
        } else if (wd_incr_style == "cosine") {
            coeff = 0.5 * (std::cos(M_PI * (1 - incr_ratio)) + 1.0);
        } else {
            throw std::runtime_error(wd_incr_style + " weight decay increment style is not supported.");
        }

        return start_wd + coeff * delta_wd;
    }

    double get_lr(int64_t num_steps) const {
        

        // 线性预热
        if (lr_warmup_steps > 0 && num_steps <= lr_warmup_steps) {
            return init_lr + ((max_lr - init_lr) * num_steps) / lr_warmup_steps;
        }

        // 常数学习率
        if (lr_decay_style == "constant") return max_lr;

        // 衰减到最小学习率
        if (num_steps > lr_decay_steps) return min_lr;

        // 其他衰减策略
        int64_t num_steps_ = num_steps - lr_warmup_steps;
        int64_t decay_steps_ = lr_decay_steps - lr_warmup_steps;
        double decay_ratio = static_cast<double>(num_steps_) / decay_steps_;
        assert(decay_ratio >= 0.0 && decay_ratio <= 1.0);
        double delta_lr = max_lr - min_lr;

        double coeff;
        if (lr_decay_style == "linear") {
            coeff = 1.0 - decay_ratio;
        } else if (lr_decay_style == "cosine") {
            coeff = 0.5 * (std::cos(M_PI * decay_ratio) + 1.0);
        } else if (lr_decay_style == "inverse-square-root") {
            int warmup_steps = std::max(lr_warmup_steps, static_cast<int64_t>(1));
            int effective_steps = std::max(num_steps, static_cast<int64_t>(1));
            double lr = max_lr * std::sqrt(warmup_steps) / std::sqrt(effective_steps);
            return std::max(min_lr, lr);
        } else {
            throw std::runtime_error(lr_decay_style + " decay style is not supported.");
        }

        return min_lr + coeff * delta_lr;
    }

    bool operator==(const OptimizerParamScheduler& rhs) const {
        return std::fabs(init_lr - rhs.init_lr) < 1e-6 &&
               std::fabs(max_lr - rhs.max_lr) < 1e-6 &&
               std::fabs(min_lr - rhs.min_lr) < 1e-6 &&
               lr_warmup_steps == rhs.lr_warmup_steps &&
               lr_decay_steps == rhs.lr_decay_steps &&
               lr_decay_style == rhs.lr_decay_style &&
               std::fabs(start_wd - rhs.start_wd) < 1e-6 &&
               std::fabs(end_wd - rhs.end_wd) < 1e-6 &&
               wd_incr_steps == rhs.wd_incr_steps &&
               wd_incr_style == rhs.wd_incr_style;
    }

private:
    double init_lr, max_lr, min_lr;
    int64_t lr_warmup_steps, lr_decay_steps;
    std::string lr_decay_style;
    double start_wd, end_wd;
    int wd_incr_steps;
    std::string wd_incr_style;
    bool use_checkpoint_opt_param_scheduler;
    bool override_opt_param_scheduler;
};


}
}