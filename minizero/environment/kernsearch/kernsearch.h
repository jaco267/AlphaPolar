#pragma once
#include "base_env.h"
#include "comp_defines.h"
#include "random.h"  //* minizero::utils::random
#include <string>
#include <vector>
#include <random>
#include "kern_search_utils.h"
#include "configuration.h"
using std::vector;
namespace minizero::env::kernsearch {
using namespace minizero::utils;
using namespace minizero::config;
const std::string kTicTacToeName = "kernsearch";
const int kPuzzle2048DiscreteValueSize = 601;  //*neccessary for reward game  (2048)
const int kTicTacToeNumPlayer = 1;
//* 8/10 works pretty well    
const bool col_per_feat = false;  //* col perm for data aug
   
const bool random_start =  true;  //todo delete random_start in kernsearch.h and kernsearch.cpp  
//todo they is a 100% rand start in kernsearch.cpp
class KernAction : public BaseAction {//* almost done
public:
    KernAction() : BaseAction() {}
    KernAction(int action_id, Player player) : BaseAction(action_id, player) {}
    KernAction(const std::vector<std::string>& action_string_args,
         int board_size = minizero::config::env_board_size){
        assert(action_string_args.size() == 2); assert(action_string_args[0].size() == 1);
        // action_string_args  [black, 0] or [black,backOneRow]
        // cout<<"aaaa"<<action_string_args<<endl;
        player_ = Player::kPlayer1;
        action_id_ = atoi(action_string_args[1].c_str());
        assert(static_cast<int>(player_) > 0 && static_cast<int>(player_) <= 1); // assume kPlayer1 == 1, kPlayer2 == 2, ...
        // assert(action_id_ >= 0 && action_id_ < board_size);
    }

    inline Player nextPlayer() const override { return getNextPlayer(getPlayer(), 1); }
    inline std::string toConsoleString() const override { 
        // return minizero::utils::SGFLoader::actionIDToBoardCoordinateString(
        //     getActionID(), minizero::config::env_board_size);
        return std::to_string(getActionID()); 
    }
};

class KernEnv : public BaseEnv<KernAction> {
public:
    KernEnv() : complexity_(10000),
      N_(config::env_board_size),
      game_len_(config::kern_search_game_len), 
      network_type_(config::network_type), 
      init_seed_(0), dist_(0,1),
      rand_init_step_(config::rand_init_step),
      seed_rotate_(2){ 
        assert(N_ > 0);
        RotateMap = getRotateMap( rng_rotate_, N_,seed_rotate_);
        reset(); 
    }
    inline int getBoardSize() const { return N_; }
    inline int getNumActionFeatureChannels() const override { return 1; }
    inline int getInputChannelHeight() const override { return getBoardSize(); }
    inline int getInputChannelWidth() const override { return getBoardSize(); }
    inline int getHiddenChannelHeight() const override { return getBoardSize(); }
    inline int getHiddenChannelWidth() const override { return getBoardSize(); }
    inline int getDiscreteValueSize() const override { return kPuzzle2048DiscreteValueSize; } //todo change 1  
    //* random
    void reset() override {reset(utils::Random::randInt());};
    void reset(int seed);
    inline int getSeed() const {return init_seed_;}
    //* --------- 
    void set_row(int row_id);
    bool act_raw(int act_id); 
    bool act(const KernAction& action) override;
    bool act(const std::vector<std::string>& action_string_args) override;
    std::vector<KernAction> getLegalActions() const override;
    bool isLegalAction(const KernAction& action) const override;
    bool isTerminal() const override;
    float getReward() const override { return reward_; }  
    float getEvalScore(bool is_resign = false) const override;
    std::vector<float> getFeatures(utils::Rotation rotation = 
      utils::Rotation::kRotationNone) const override;
    std::vector<float> getActionFeatures(const KernAction& action,
        utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    inline int getNumInputChannels() const override { //todo channel should > 1
        if (network_type_ == "conv"){ return 1; 
        }else if (network_type_ == "transformer"){return N_;
        }else{throw std::invalid_argument( "some err msg" );}
    }  
    inline int getPolicySize() const override { return getBoardSize(); }//{ return getBoardSize() * getBoardSize(); }
    std::string toString() const override;
    inline std::string name() const override { return kTicTacToeName; }
    inline int getNumPlayer() const override { return kTicTacToeNumPlayer; }
    inline int getRotatePosition(int position, utils::Rotation rotation) const override { // in tictactoe this is called by get features, useless in here
         throw std::invalid_argument( "some err msg at get Rotate position" );
        return  position; };
    int getRotateAction(int action_id, utils::Rotation rotation) const override;
public:
    vector<int> curKern_;
    int complexity_; 
private:
    int count;    //* reset 0 
    int curRow_;  
    int N_; 
    vector<int> pdp_;  //* wait if we use vector<unsigned int> we will have segmentation fault??   but if we use vector<int> we dont have segmentation fault 
    vector<int> cwBuffer_;  
    float reward_;   float total_reward_;
    mat foundKern_;
    
    int game_len_;
    std::string network_type_;
    //*-----random init state----    // different starting point to avoid having a same start kern as nn converge
    int init_seed_;
    std::mt19937 rng_rand_init_;  //
    std::uniform_real_distribution<float> dist_; 
    int rand_init_step_; 
    //*-------reward shaping----------
    float max_comp_; float min_comp_; //min_comp max_comp from random agent
    // float alpha_; float c_val_;   
    //------for rand permutation, useless because col perm will make complexity different , useless for data aug
    int seed_rotate_; std::mt19937 rng_rotate_; mat RotateMap;
    

};
}
