#include "kernsearch.h"
#include "random.h"
#include "sgf_loader.h"
#include <algorithm>
#include <string>
#include <cmath>
#include <iostream>
#include <random>
#include <cstdarg> 
#include "comp_defines.h"
#include "comp_utils.h"
#include "kern_search_utils.h"
#include "kernsearch_env_loader.h"
#include <assert.h>
#include "d_kernels.h"
#include "configuration.h"
#include <boost/algorithm/string.hpp>
using namespace std;
namespace minizero::env::kernsearch {
using namespace minizero::utils;
float transform_comp_reward(int comp, float gamma, float x_min, float x_max,float y_min, float y_max){
  if (comp > x_max){  comp = x_max; }
  if (comp < x_min){  comp = x_min; }

  ASSERT_THROW((x_min <= comp) && (comp <= x_max), "comp should be in range x_min x_max");  
  float trans_v = y_min + (y_max - y_min) * std::pow((x_max - comp)/(x_max - x_min), gamma);
  ASSERT_THROW( ((trans_v <= y_max) && (trans_v >= y_min)), "trans v should be in range y_min y_max");
  return trans_v;
}
bool KernEnv::act(const KernAction& action){//*done just put a stone
  if (!isLegalAction(action)) { return false; }
  assert (foundKern_.size() == 0 && "foundKern should only be triggered once?");  
  float c_val; float alpha;
  if (N_ == 12){ alpha = 5;  c_val = 0.1; 
  }else{alpha = 10;  c_val = 0.1; } 
  actions_.push_back(action);
  reward_ = - c_val;//-1;   //* -1 //*0
  count+=1;
  turn_ = action.nextPlayer();
  int act_id = action.getActionID(); 
  assert(action.getPlayer() == Player::kPlayer1 && "should only have one player");
  assert(act_id < N_ && "action id should < N");
  int action_int = std::pow(2,act_id);
  curKern_[curRow_] |= action_int;
  assert(pdp_[curRow_] >= __builtin_popcount(curKern_[curRow_]));
  if (pdp_[curRow_] ==__builtin_popcount(curKern_[curRow_])){  
    cwBuffer_[std::pow(2,curRow_)] = curKern_[curRow_];
    bool res = CheckPD(cwBuffer_,curRow_,pdp_[curRow_]);
    if (res){
      curRow_ += 1;  
      //* this is very important
      reward_ = alpha;  //5
      if(curRow_ < N_){curKern_[curRow_] = 0;
      }else if (curRow_ == N_){  //*Terminal  
        vector<int> curKern_rev = curKern_;  
        std::reverse(curKern_rev.begin(),curKern_rev.end());
        foundKern_ = comp_vec2mat(curKern_rev,N_);

        complexity_ = 100; //todo  this should be the complexity of your decoder   

        float tmp_reward; 
        tmp_reward = max_comp_ - complexity_; 
        float gamma = 2; 
        float tmp_reward2 = transform_comp_reward(complexity_, gamma, min_comp_,max_comp_, 0, max_comp_ - min_comp_);  
        if (gamma == 1){
          ASSERT_THROW(tmp_reward2-tmp_reward<=0.1 && tmp_reward2-tmp_reward>=-0.1, "tmp_r 2 and tmp_r shoud be close if gamma = 1 ");
        }
        tmp_reward = tmp_reward2; 
        if (tmp_reward < alpha){  //160
          tmp_reward = alpha;  
        }
        reward_ = tmp_reward; 
      }else{ throw std::invalid_argument( "some err msg" );}
    }else{
      curKern_[curRow_] = 0;
    }
  }
  total_reward_ += reward_; 
  return true;
}
//* done?
//* put a stone by string ex . in console  play "black A8" <- action string
bool KernEnv::act(const std::vector<std::string>& action_string_args){
    return act(KernAction(action_string_args));
}
//*half, only used in mode_handler.cpp runEnvTest()
std::vector<KernAction> KernEnv::getLegalActions() const{
  if (turn_ != Player::kPlayer1){ throw std::invalid_argument( "some err msg" );}
  std::vector<KernAction> actions;
  for (int pos = 0; pos < N_; ++pos) {
      KernAction action(pos, turn_);
      if (!isLegalAction(action)) { continue; }
      actions.push_back(action);
  }
  return actions;
}
//* half
bool KernEnv::isLegalAction(const KernAction& action) const{
    //* half check a action is legal
    assert(action.getActionID() >= 0 &&  
      action.getActionID() < N_ );
    assert(action.getPlayer() == Player::kPlayer1);

    if (curRow_ < N_){
       assert ( comp_vec2mat(curKern_,N_)[curRow_][action.getActionID()]==((curKern_[curRow_]>>action.getActionID())&1));
       if (((curKern_[curRow_]>>action.getActionID())&1) == 1){return false;
       }else{return true;}
    }
    return true;
}
bool KernEnv::isTerminal() const{//*done
    // terminal: any player wins or board is filled
    if (count > game_len_){return true;}
    if (curRow_ == N_){ return true;//todo  add a count constraint  
    }else{ return false;}
}
//* done?
float KernEnv::getEvalScore(
  bool is_resign /*= false*/) const{
    return total_reward_;// return 1.0f;
}
//* half
std::vector<float> KernEnv::getFeatures(  //* if this feature size is wrong , error will throw : RuntimeError: CUDA error: invalid device ordinal
  utils::Rotation rotation /*= utils::Rotation::kRotationNone*/) const{
  if (col_per_feat == false){//*without colpermute 
    if (network_type_ == "conv"){
      std::vector<float> features;
      for (int row = 0; row < N_; row++) {
        int row_int = curKern_[row];
        for (int col =0; col < N_; col ++){
          features.push_back(((row_int & (1 << col)) ? 1.0f : 0.0f));
        }
      }
      return features;

    }else if (network_type_ == "transformer"){
      std::vector<float> features;
      for (int i =0; i< N_; i++){
        for (int row = 0; row < N_; row++) {
          int row_int = curKern_[row];
          for (int col =0; col < N_; col ++){
            features.push_back(((row_int & (1 << col)) ? 1.0f : 0.0f));
          }
        }
      }
      return features; 
    }else{
       throw std::invalid_argument( "some err msg" );
    }
  }else{//* col permute
     throw std::invalid_argument( "some err msg" );
    //todo wait but the problem is we shouldn't col permute to increase features because col permute will affect complexity 
    int rot = static_cast<int>(rotation);   
    assert(0 <= rot && rot < 8  && "rot should < rot size");
    std::vector<int> col_permute = RotateMap[rot];
    mat Kern_perm;  
    mat Kern_cur_mat = comp_vec2mat(curKern_,N_); 
    ///col_perm  3210
    std::vector<float> features;
    for (int row = 0; row < N_; row++) {
      // int row_int = curKern_[row];
      for (int col =0; col < N_; col ++){
        // features.push_back(((row_int & (1 << col)) ? 1.0f : 0.0f));
        features.push_back(Kern_cur_mat[row][col_permute[col]]);
      }
    }
    return features;
  }
   throw std::invalid_argument( "some err msg" );

}
std::vector<float> KernEnv::getActionFeatures(const KernAction& action, 
  utils::Rotation rotation /*= utils::Rotation::kRotationNone*/) const{
    // std::vector<float> action_features(N_ * N_, 0.0f);
    throw std::invalid_argument( "some err msg, this is only for muzero" );
    std::vector<float> action_features(N_ , 0.0f);
    action_features[action.getActionID()] = 1.0f;
    return action_features;
}
//* half
std::string KernEnv::toString() const{
  std::ostringstream oss;
  for (int row = 0; row < N_; row++) {
    int row_int = curKern_[row];
    oss << row + 1 << " ";
    for (int col = 0; col < N_; ++col) {
      oss<<(row_int & (1 << col) ? "1" : "0")<<" ";
    }
    if (row == curRow_){oss<<"<<curRow";}
    oss << std::endl;
  }
  oss<<"---- reward: "<<reward_<<", total reward: "<<total_reward_
     <<", terminal:"<<isTerminal()<<",count:"<<count<<endl;
  if (curRow_ == N_){
    oss<<"complexity_:"<<complexity_<<endl;
  }
  return oss.str();
}

int KernEnv::getRotateAction(int action_id, utils::Rotation rotation) const{ 
  if (col_per_feat == false){//*without colpermute 
    return action_id; 
  }else{
    throw std::invalid_argument( "some err msg" );
    int a2 = getPermAction(action_id, rotation, RotateMap, N_);
    return a2;
  }
  throw std::invalid_argument( "some err msg" );
}
} 
