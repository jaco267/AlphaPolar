#pragma once

// #include "kernsearch.h"
#include "base_env.h"
#include "comp_defines.h"
#include <string>
#include <vector>
#include <random>
// #include <config.h>
#include "kern_search_utils.h"
namespace minizero::env::kernsearch {
    using namespace minizero::utils;
    bool CheckPD(std::vector<int>& m_pWordBuffer,int CurrentRow, int RequiredPD);
    
    mat getRotateMap(std::mt19937 & rng_rotate,int N,int seed);
    int getPermAction(int action_id, utils::Rotation rotation,
        const mat & RotateMap, int N); 
}