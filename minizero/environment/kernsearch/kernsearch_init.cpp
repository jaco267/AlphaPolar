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
void KernEnv::set_row(int row_id){
    curKern_[curRow_] = row_id;
    cwBuffer_[std::pow(2,curRow_)] = curKern_[curRow_];
    bool res = CheckPD(cwBuffer_,curRow_,pdp_[curRow_]);  
    ASSERT_THROW(res == true , "set row should satisfy");  
    curRow_+=1; 
}
bool KernEnv::act_raw(int act_id){
    int action_int = std::pow(2,act_id);
    curKern_[curRow_] |= action_int;
    //todo  not sure whether we can use reward at here 
    ASSERT_THROW(pdp_[curRow_] >= __builtin_popcount(curKern_[curRow_]), "pdp should >= weight");
    if (pdp_[curRow_] ==__builtin_popcount(curKern_[curRow_])){ 
        cwBuffer_[std::pow(2,curRow_)] = curKern_[curRow_];
        bool res = CheckPD(cwBuffer_,curRow_,pdp_[curRow_]);
        if (res){
        curRow_ += 1;  
        ASSERT_THROW(curRow_<N_, "tooo many init steps");  
        curKern_[curRow_] = 0; 
        }else{ //search again
        curKern_[curRow_] = 0;
        }
    }
    return true;
}
void KernEnv::reset(int seed){ //* done just reset the board
    //we need this seed since we want to replicate the same result from sgf file ,using seed reset and action to reconstruct the state  
    // init_seed_ = seed; random_.seed(init_seed_);   // Seed the random number generator with 'seed_'
    rng_rand_init_.seed(init_seed_ = seed);   //* this see will then be put into tag_ in sgf file   because init_seed_ = seed;   get_seed{return init_seed_}
    curRow_ = 0;
    assert((N_==4 || (N_ >= 8 && N_ <= 16)) && "N_ should be 4 or 16");
    if (N_ == 4){       pdp_ = {1,2,2,4};   
    }else if (N_ == 8){ pdp_ = {1,2,2,2,4,4,4,8};
    }else if (N_ == 9){ pdp_ = {1,2,2,2,2,4,4,6,6};
    }else if (N_ == 10){pdp_ = {1,2,2,2,2,4,4,4,6,8};
    }else if (N_ == 11){pdp_ = {1,2,2,2,2,4,4,4,6,6,8};
    }else if (N_ == 12){pdp_ = {1,2,2,2,2,4,4,4,4,6,6,12};
    }else if (N_ == 13){pdp_ = {1,2,2,2,2,4,4,4,4,6,6,8,10};
    }else if (N_ == 14){pdp_ = {1,2,2,2,2,4,4,4,4,6,6,8,8,8};
    }else if (N_ == 15){pdp_ = {1,2,2,2,2,4,4,4,4,6,6,8,8,8,8};
    }else if(N_ == 16){ pdp_ = {1,2,2,2,2,4,4,4,4,6,6,8,8,8,8,16};
    }else{ throw std::invalid_argument( "some err msg" );}
    std::reverse(pdp_.begin(),pdp_.end());
    int cw_size = std::pow(2, N_); 
    cwBuffer_.resize(cw_size, 0);
    for (unsigned i =0; i<cwBuffer_.size(); i++){cwBuffer_[i]= 0;}
    curKern_.resize(N_,0);
    for (unsigned i =0; i<curKern_.size(); i++){curKern_[i]= 0;}
    //* -------------------
    total_reward_ = 0; reward_ = 0; turn_ = Player::kPlayer1;
    count = 0;
    actions_.clear();
    foundKern_.clear();
    //*--------------------
    bool hand_craft = false;  if (N_ == 16){hand_craft = true; } 
    // if (hand_craft){ 
    //   auto comp_vec = mat2comp_vec(R16);
    //   std::reverse(comp_vec.begin(), comp_vec.end());  
    //   //* 16  8   8    8    8      6
    //   //65535 255 3855 13107 21845   
    //   for (int i = 0; i<5; i++){
    //     ASSERT_THROW(curRow_ == i ,"we should be in ith row");
    //     set_row(comp_vec[i]);
    //   }
    // }else{//normal start 
      if (pdp_[0] == N_){
        ASSERT_THROW(curRow_ == 0 , "currow at init should be 0");
        set_row(std::pow(2,N_) - 1);
        ASSERT_THROW(curRow_ == 1 ,"we should be in 2nd row");
      }else{
         cout<< "should not come here N should == pdp[0]" <<endl;
      }
    // }
    //*-----min/max comp estimate from random agent----
    std::vector<std::string> tokens;// Split the string by ':'
    boost::split(tokens, config::min_max_comp, boost::is_any_of(","));
    int min = -1; int max=-1;
    if (tokens.size() == 2) {
      min = std::stoi(tokens[0]);  max = std::stoi(tokens[1]);
    } else {
      std::cout << "Error: Invalid format! for min_max_comp" << std::endl;
    }
    if ((min > 0) && (max > 0)){
      min_comp_ = (float) min;  max_comp_ = (float) max;  
    }else{
      if (N_ == 16){    
        if (hand_craft){
          // max_comp_ = 4200;  min_comp_ = 1200; //wrong comp
          max_comp_ = 5000;  min_comp_ = 1250;
        }else{
          max_comp_ = 5000; min_comp_ = 1300;   //comp 3482~5666   hand craft 1384
        }
      }else if (N_ == 12){
        max_comp_ = 1500; min_comp_ = 700;
        // max_comp_ = 1500; min_comp_ = 600; //todo wrong comp //comp 936~1422 ;   
      }else if (N_ == 8){ 
        max_comp_ = 350 ; min_comp_ = 140; //todo wrong comp   //* 10*8 = 80;  comp = 168~280
      }else if (N_ == 4){  
        max_comp_ = 100;  min_comp_ = 20;  //todo wrong comp  //* 10*4 = 40;  comp 32~40  100-40 = 60
      }else{ cout<<"havent implement this reward  N = "<<N_<<endl; 
        throw std::invalid_argument( "some err msg" );
      }
    }
    // cout<<"config min max: "<<config::min_max_comp<<endl;
    if (random_start){  //randomly put some stone on the board in the begining with some probability
      float rand_value = dist_(rng_rand_init_);   //0.9
      ASSERT_THROW(rand_value <= 1 && rand_value >= 0 , "rand value should be between 0~1");
      float rand_portion = 1;//0.5; //todo  
      if (rand_value  > rand_portion){//* start with zeroboard 
      }else{  //just randomly generate some move 
        vector<KernAction> legal_actions;
        for (int s=0 ; s< rand_init_step_; s++){ 
          int act_id = std::uniform_int_distribution<int> ()(rng_rand_init_) % N_;
          //todo  we need to do legal action.... see z_cpp_kern_search/b_my
          act_raw(act_id);// init state cant get reward (no action pair),so we use act_raw instead of act() 
        }
      }
    }
  }
}