#include "kernsearch_env_loader.h"

namespace minizero::env::kernsearch {
//* ------------------KernEnvLoader------------------------------
float KernEnvLoader::transformValue(float value) const{
  // reference: Observe and Look Further: Achieving Consistent Performance on Atari, page 11
  const float epsilon = 0.001;
  const float sign_value = (value > 0.0f ? 1.0f : (value == 0.0f ? 0.0f : -1.0f));
  // value = sign_value * (sqrt(fabs(value) + 1) - 1) + epsilon * value;
  value = sign_value * (std::pow(fabs(value) + 1, pow_) - 1) + epsilon * value;
  return value;
}
std::vector<float> KernEnvLoader::getActionFeatures(const int pos, 
  utils::Rotation rotation /* = utils::Rotation::kRotationNone */) const{
    //todo only for mu zero  
    throw std::invalid_argument( "some err msg, only for muzero" );
    const KernAction& action = action_pairs_[pos].first;
    // std::vector<float> action_features(N_ * N_, 0.0f);
    std::vector<float> action_features(N_, 0.0f);
    int action_id = ((pos < static_cast<int>(action_pairs_.size())) ? 
           action.getActionID() : utils::Random::randInt() % action_features.size());
    action_features[action_id] = 1.0f;
    return action_features;
}

float KernEnvLoader::calculateNStepValue(const int pos) const{
    assert(pos < static_cast<int>(action_pairs_.size()));
    const int n_step = config::learner_n_step_return;
    const float discount = config::actor_mcts_reward_discount;
    size_t bootstrap_index = pos + n_step;
    float value = 0.0f;
    float n_step_value = ((bootstrap_index < action_pairs_.size()) ? std::pow(discount, n_step) * BaseEnvLoader::getValue(bootstrap_index)[0] : 0.0f);
    for (size_t index = pos; index < std::min(bootstrap_index, action_pairs_.size()); ++index) {
        float reward = BaseEnvLoader::getReward(index)[0];
        value += std::pow(discount, index - pos) * reward;
    }
    value += n_step_value;
    return value;
}

std::vector<float> KernEnvLoader::toDiscreteValue(float value) const{
    std::vector<float> discrete_value(kPuzzle2048DiscreteValueSize, 0.0f);
    int value_floor = floor(value);
    int value_ceil = ceil(value);
    // int shift = kPuzzle2048DiscreteValueSize / 2;
    int value_floor_shift = std::min(std::max(value_floor + shift_, 0), kPuzzle2048DiscreteValueSize - 1);
    int value_ceil_shift = std::min(std::max(value_ceil + shift_, 0), kPuzzle2048DiscreteValueSize - 1);
    if (value_floor == value_ceil) {
        discrete_value[value_floor_shift] = 1.0f;
    } else {
        discrete_value[value_floor_shift] = value_ceil - value;
        discrete_value[value_ceil_shift] = value - value_floor;
    }
    return discrete_value;
}

int KernEnvLoader::getRotateAction(int action_id, utils::Rotation rotation) const{ 
  //* this is called by envLoader::getPolicy
  if (col_per_feat == false){//*without colpermute 
    return action_id; 
  }else{
    throw std::invalid_argument( "some err msg" );
    int a2 = getPermAction(action_id, rotation, RotateMap, N_);
    return a2;
  }
   throw std::invalid_argument( "some err msg" );
};
}