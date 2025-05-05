#include "kern_search_utils.h"
namespace minizero::env::kernsearch {
    using namespace minizero::utils;
    bool CheckPD(std::vector<int>& m_pWordBuffer,int CurrentRow, int RequiredPD){
      if (CurrentRow == 0) return true;
      int n = 1u << CurrentRow;
      bool isSucceed = true;
      for (int i = 1; i < (int)(1u << CurrentRow); ++i) {
        int &res = m_pWordBuffer[i + n];
        res = m_pWordBuffer[i] ^ m_pWordBuffer[n];
        // unsigned weight = _mm_popcnt_u32(res);
        int weight = __builtin_popcount(res);//todo
        if (weight < RequiredPD) { isSucceed = false;break;}
      }
      return isSucceed;
    }
    mat getRotateMap(std::mt19937 & rng_rotate,int N,int seed){
      rng_rotate.seed(seed);   //* so that if the seed is the same we get same rotation map
      mat rotation_map;
      int rotate_size = static_cast<int> (utils::Rotation::kRotateSize); 
      rotation_map.resize(rotate_size, std::vector<int>(N));  
      for (int i=0; i< rotate_size; i++){
        std::vector<int> a(N);  
        std::iota(a.begin(),a.end(),0);  
        if (i>0){ //rotate 0 doesn't rotate (doesn't permute column)
          std::shuffle(a.begin(),a.end(), rng_rotate);
        }
        rotation_map[i] = a;
      }
      return rotation_map; 
    }
    int getPermAction(int action_id, utils::Rotation rotation,const mat & RotateMap, int N){ 
      // return action_id;  //* old version
      int rot = static_cast<int>(rotation);   
      assert(0 <= rot && rot < 8  && "rot should < rot size");
      std::vector<int> col_permute = RotateMap[rot];
      // cout<<"rot: "<<rot<<endl;print_mat(RotateMap); 
      // cout<<"vec"<<endl;  print_vec(col_permute);  //todo move it as a utils function
      int action_rot = -1;  
      for(int i =0; i< N; i++){
        if(col_permute[i] == action_id){
          action_rot = i;
          break;
        }
      }
      return action_rot; 
    }
}