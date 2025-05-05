#include "tictactoe.h"
#include "random.h"
#include "sgf_loader.h"
#include <algorithm>
#include <string>
#include <iostream>
#include <random>
#include <cstdarg> //for format
#include <torch/torch.h>
using namespace std;
//todo  how can I use otherclass in here
namespace minizero::env::tictactoe {
using namespace minizero::utils;
int plus(int a, int b){ //* this is dumb
  return 0;
}
void TicTacToeEnv::reset(){ //* done just reset the board
    turn_ = Player::kPlayer1;
    actions_.clear();
    board_.resize(kTicTacToeBoardSize * kTicTacToeBoardSize);
    fill(board_.begin(), board_.end(), Player::kPlayerNone);
}
bool TicTacToeEnv::act(const TicTacToeAction& action){//*done just put a stone
    int a[] = {10};
    a[0] = 3;
    a[0] = plus(a[0],3);
    // torch::Tensor tensor = torch::rand({2, 3});
    // torch::Tensor T2 = tensor.reshape({3,2});
    // std::cout << tensor << std::endl;
    //todo why cout have error, maybe try to do this when return console str
    //the tostring function //todo we cant use cout use oss instead
    if (!isLegalAction(action)) { return false; }
    actions_.push_back(action);
    board_[action.getActionID()] = action.getPlayer();
    turn_ = action.nextPlayer();
    return true;
}
//* put a stone by string ex . in console  play "black A8" <- action string
bool TicTacToeEnv::act(const std::vector<std::string>& action_string_args){
    return act(TicTacToeAction(action_string_args));
}
//*this only used in mode_handler.cpp runEnvTest()
std::vector<TicTacToeAction> TicTacToeEnv::getLegalActions() const{
    std::vector<TicTacToeAction> actions;
    for (int pos = 0; pos < kTicTacToeBoardSize * kTicTacToeBoardSize; ++pos) {
        TicTacToeAction action(pos, turn_);
        if (!isLegalAction(action)) { continue; }
        actions.push_back(action);
    }
    return actions;
}
bool TicTacToeEnv::isLegalAction(const TicTacToeAction& action) const{
    //* half check a action is legal
    assert(action.getActionID() >= 0 && 
          action.getActionID() < kTicTacToeBoardSize * kTicTacToeBoardSize);
    assert(action.getPlayer() == Player::kPlayer1 || action.getPlayer() == Player::kPlayer2);
    return (action.getActionID() >= 0 && action.getActionID() < kTicTacToeBoardSize * kTicTacToeBoardSize 
              && board_[action.getActionID()] == Player::kPlayerNone);
}
bool TicTacToeEnv::isTerminal() const{//*done
    // terminal: any player wins or board is filled
    return (eval() != Player::kPlayerNone || 
    std::find(board_.begin(), board_.end(), Player::kPlayerNone) == board_.end());
}
//todo
float TicTacToeEnv::getEvalScore(
  bool is_resign /*= false*/) const{
    Player result = (is_resign ? 
      getNextPlayer(turn_, kTicTacToeNumPlayer) : eval());
    switch (result) {
        case Player::kPlayer1: return 1.0f;
        case Player::kPlayer2: return -1.0f;
        default: return 0.0f;
    }
}

std::vector<float> TicTacToeEnv::getFeatures(
  utils::Rotation rotation /*= utils::Rotation::kRotationNone*/) const{
  /* 4 channels:
    0~1. own/opponent position
    2. Nought turn
    3. Cross turn
  */
  std::vector<float> features;
  for (int channel = 0; channel < 4; ++channel) {
    for (int pos = 0; pos < kTicTacToeBoardSize * kTicTacToeBoardSize; ++pos) {
      int rotation_pos = getRotatePosition(pos, utils::reversed_rotation[static_cast<int>(rotation)]);
      if (channel == 0) {
          features.push_back((board_[rotation_pos] == turn_ ? 1.0f : 0.0f));
      } else if (channel == 1) {
          features.push_back((board_[rotation_pos] == getNextPlayer(turn_, kTicTacToeNumPlayer) ? 1.0f : 0.0f));
      } else if (channel == 2) {
          features.push_back((turn_ == Player::kPlayer1 ? 1.0f : 0.0f));
      } else if (channel == 3) {
          features.push_back((turn_ == Player::kPlayer2 ? 1.0f : 0.0f));
      }
    }
  }
  return features;
}
//todo
std::vector<float> TicTacToeEnv::getActionFeatures(const TicTacToeAction& action, 
  utils::Rotation rotation /*= utils::Rotation::kRotationNone*/) const{
    std::vector<float> action_features(kTicTacToeBoardSize * kTicTacToeBoardSize, 0.0f);
    action_features[getRotateAction(action.getActionID(), rotation)] = 1.0f;
    return action_features;
}
std::string TicTacToeEnv::toString() const{
  std::ostringstream oss;  
  // cout<<"hey"<<c<<std::endl; //* if you cout you get a error 
  torch::Tensor tensor = torch::rand({2, 3});
  // cout<<tensor<<"hahaha"<<endl;
  tensor = tensor.unsqueeze(0).unsqueeze(0);  
  // board_2 = Board();
  oss << "   A  B  C" << std::endl;
  for (int row = kTicTacToeBoardSize - 1; row >= 0; --row) {
    oss << row + 1 << " ";
    for (int col = 0; col < kTicTacToeBoardSize; ++col) {
        if (board_[row * kTicTacToeBoardSize + col] == Player::kPlayerNone) {
            oss << " . ";
        } else if (board_[row * kTicTacToeBoardSize + col] == Player::kPlayer1) {
            oss << " O ";
        } else { oss << " X ";}
    }
    oss << " " << row + 1 << std::endl;
  }
  oss << "   A  B  C" << std::endl;
  return oss.str();
}

Player TicTacToeEnv::eval() const {
  int c;
  for (int i = 0; i < kTicTacToeBoardSize; ++i) {
    // rows
    c = 3;
    for (int j = 0; j < kTicTacToeBoardSize; ++j) { 
      c &= static_cast<int>(board_[i * kTicTacToeBoardSize + j]); }
    if (c != static_cast<int>(Player::kPlayerNone)) { 
      return static_cast<Player>(c); }
    // columns
    c = 3;
    for (int j = 0; j < kTicTacToeBoardSize; ++j) { 
      c &= static_cast<int>(board_[j * kTicTacToeBoardSize + i]); }
    if (c != static_cast<int>(Player::kPlayerNone)) { return static_cast<Player>(c); }
  }
  // diagonal (left-up to right-down)
  c = 3;
  for (int i = 0; i < kTicTacToeBoardSize; ++i) { 
    c &= static_cast<int>(board_[i * kTicTacToeBoardSize + i]); }
  if (c != static_cast<int>(Player::kPlayerNone)) { return static_cast<Player>(c); }
  // diagonal (right-up to left-down)
  c = 3;
  for (int i = 0; i < kTicTacToeBoardSize; ++i) { 
    c &= static_cast<int>(board_[i * kTicTacToeBoardSize + 
    (kTicTacToeBoardSize - 1 - i)]); }
  if (c != static_cast<int>(Player::kPlayerNone)) { return static_cast<Player>(c); }
  return Player::kPlayerNone;
}
std::vector<float> TicTacToeEnvLoader::getActionFeatures(const int pos, 
  utils::Rotation rotation /* = utils::Rotation::kRotationNone */) const{
    const TicTacToeAction& action = action_pairs_[pos].first;
    std::vector<float> action_features(kTicTacToeBoardSize * kTicTacToeBoardSize, 0.0f);
    int action_id = ((pos < static_cast<int>(action_pairs_.size())) ? getRotateAction(action.getActionID(), rotation) : utils::Random::randInt() % action_features.size());
    action_features[action_id] = 1.0f;
    return action_features;
}

} // namespace minizero::env::tictactoe
