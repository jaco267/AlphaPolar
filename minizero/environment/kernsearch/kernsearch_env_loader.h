#pragma once
#include "base_env.h"
#include "comp_defines.h"
#include <string>
#include <vector>
#include <random>

#include "kern_search_utils.h"
#include "kernsearch.h"
using std::vector;
using std::string;
namespace minizero::env::kernsearch {
using namespace minizero::utils; 
using namespace minizero::config;
class KernEnvLoader : public BaseEnvLoader<KernAction, KernEnv> {
public:
    //* baseBoard Env  
    KernEnvLoader() : N_(config::env_board_size), 
     //kpuzzle2048DiscreteVlaueSize/2 = 601/2    default pow(0.5) shift 300
    //pow_(0.5), shift_(300),   //* todo pow 0.85  shift 50  
    pow_(0.5), shift_(300),  
    seed_rotate_(2) {
        assert(N_ > 0);
        RotateMap = getRotateMap( rng_rotate_, N_,seed_rotate_);
    }
    virtual ~KernEnvLoader() = default;
    void loadFromEnvironment(const KernEnv& env, 
        const std::vector<std::vector<
          std::pair<std::string, std::string>
        >>& action_info_history = {}) override{
        string kern_string0;  
        // BaseEnvLoader<KernAction, KernEnv>::loadFromEnvironment(env, action_info_history);
        int env_seed = env.getSeed();  
        KernEnv env_copy;
        env_copy.reset(env_seed);  
        for (int row_int : env_copy.curKern_){kern_string0+=std::to_string(row_int) + ":";}
        //*---------base env loadFromEnv
        reset();
        for (size_t i = 0; i < env.getActionHistory().size(); ++i) {
            addActionPair(env.getActionHistory()[i], action_info_history.size() > i ? ActionInfo(action_info_history[i]) : ActionInfo());
        }
        addTag("ini_kern", kern_string0);
        addTag("RE", std::to_string(env.getEvalScore()));
        // add observations
        std::string observations;
        for (const auto& obs : env.getObservationHistory()) { observations += obs; }
        addTag("OBS", utils::compressString(observations));
        assert(observations == utils::decompressString(getTag("OBS")));
        //*-------------------------------------------------------------------
        string kern_string;
        for (int row_int : env.curKern_){kern_string+=std::to_string(row_int) + ":";}
        addTag("kern", kern_string);  //* final kernel
        //* store size & seed into sgf file so later we can reconstruct the state just by reset() and act()
        addTag("SZ", std::to_string(env.getBoardSize())); //* will be used in addTag
        addTag("SD", std::to_string(env.getSeed()));   //* will be called by getSeed() and getFeatures's reset
        addTag("kernComp",std::to_string(env.complexity_));
    }
    inline int getSeed() const { return std::stoi(getTag("SD"));}
    inline void addTag(const std::string& key, const std::string& value) override{
        BaseEnvLoader<KernAction, KernEnv>::addTag(key, value);
        if (key == "SZ") { N_ = std::stoi(value); }
    }
    inline int getBoardSize() const { return N_; }
    //* -----random ----
    vector<float> getFeatures(const int pos, utils::Rotation rotation = utils::Rotation::kRotationNone) const override{
        KernEnv env;  
        env.reset(getSeed());  
        const auto& action_pairs__ = action_pairs_;
        for(int i =0; i < std::min(pos, static_cast<int>(action_pairs__.size())); i++){
            env.act(action_pairs__[i].first);
        }  
        return env.getFeatures(rotation); 
    }
    //* ------------------------------------------------
    float transformValue(float value) const ;
    std::vector<float> getActionFeatures(const int pos, 
       utils::Rotation rotation = utils::Rotation::kRotationNone) const override;
    // inline std::vector<float> getValue(const int pos) const { return {getReturn()}; }
    std::vector<float> getValue(const int pos) const override {   //todod becuae we change getDiscreteValueSize to 601
        return toDiscreteValue(pos < static_cast<int>(action_pairs_.size()) ? 
           transformValue(calculateNStepValue(pos)) : 0.0f);    
        // return {getReturn()}; 
    }
    std::vector<float> getReward(const int pos) const override {
        return toDiscreteValue(pos < static_cast<int>(action_pairs_.size()) ? 
        transformValue(BaseEnvLoader::getReward(pos)[0]) : 0.0f); }
    float getPriority(const int pos) const override {return 
      fabs(calculateNStepValue(pos) - BaseEnvLoader::getValue(pos)[0]); }
    
    inline std::string name() const override { return kTicTacToeName; }
    inline int getPolicySize() const override { return getBoardSize(); }//{ return getBoardSize() * getBoardSize(); }
    
    inline int getRotatePosition(int position, utils::Rotation rotation) const override { 
        throw std::invalid_argument( "some err msg rot pos" );
        return  position; };
    int getRotateAction(int action_id, utils::Rotation rotation) const override;
private:
    float calculateNStepValue(const int pos) const;
    std::vector<float> toDiscreteValue(float value) const;
protected:
    int N_;
    //* ---for transform value---  
    float pow_;  
    int shift_; 
    //* ---rotate----
    int seed_rotate_; 
    std::mt19937 rng_rotate_; 
    mat RotateMap;
};
 
}
