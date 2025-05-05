#pragma once
#include <vector>
// using namespace std;  //todo wait this is weird if I uncomment this some nasty error occurs
//todo avoid using namespace std, otherwise it will cause conflict between 
//todo std::nullopt (from <optional>) and c10::nullopt 
typedef std::vector<std::vector<int>> mat;


#define ASSERT_THROW(cond, msg) if (!(cond)) throw std::runtime_error(msg);