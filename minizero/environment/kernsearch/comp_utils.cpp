#include <vector>
#include <iostream>
#include "comp_utils.h"
#include <cmath>
#include <bitset>
#include "comp_defines.h"
using std::vector;
using std::cout;
using std::endl;
//* ----np func----
vector<int> arange(int start, int end, int step) {
    std::vector<int> result;
    for (int i = start; i < end; i += step) {
        result.push_back(i);
    }
    return result;
}
mat eye(int k){
  mat e;
  for (int i=0;i<k;i++){
    vector<int> row;  
    row.resize(k,0);  
    row[i] = 1;  
    e.push_back(row);
  } 
  return e;
}

mat hstack3(const mat & A, const mat & B, const mat& C) {
    mat result;
    for (size_t i = 0; i < A.size(); ++i) {
        vector<int> row = A[i];
        row.insert(row.end(), B[i].begin(), B[i].end());
        row.insert(row.end(), C[i].begin(), C[i].end());
        result.push_back(row);
    }
    return result;
}

vector<int> setdiff1d(const vector<int> &input, const vector<int> &diff_list){
 vector<int> ret_rows;
 bool flag; 
 for(unsigned i = 0; i<input.size(); i++){
     flag = true; 
     for (unsigned j = 0;j<diff_list.size(); j++){
       if(input[i] == diff_list[j]){flag = false;}
     }
     if (flag){
       ret_rows.push_back(input[i]);
     }
 }
 return ret_rows;
}
void setdiff1d_rev(vector<int> &ones_rows, vector<int> &fixed_rows_list 
     ,vector<int> &del_rows){
  bool flag; 
  for(int i = (int) ones_rows.size()-1; i>=0; i--){
      flag = true; 
      for (unsigned j = 0;j<fixed_rows_list.size(); j++){
        if(ones_rows[i] == fixed_rows_list[j]){flag = false;}
      }
      if (flag){
        del_rows.push_back(ones_rows[i]);
      }
  }
}
//* -------basic-----------
int bin_to_dec(const std::vector<int>& bin_list) {
  int dec = 0;
  int n = (int) bin_list.size();
  for (int i = 0; i < n; ++i) {
      dec += bin_list[i] * std::pow(2, n - 1 - i);
  }
  return dec;
}
vector<int> dec_to_bin(int n, int x){
  std::string binary = std::bitset<32>(n).to_string(); // Convert to 32-bit binary string
  binary = binary.substr(32 - x); // Keep only the last x bits
  std::vector<int> result;
  for (char c : binary) {
      result.push_back(c - '0'); // Convert char to int
  }
  return result;
}
vector<int> mat2comp_vec(const mat& g){
  vector<int> g_list;  
  for (auto row : g){
    int val = 0;  
    for (unsigned i =0; i< row.size(); i++){
        if (row[i] == 1){
            val += std::pow(2,i);
        }
    }
    g_list.push_back(val);
  }
  return g_list;
}
mat comp_vec2mat(const vector<int> & v,int size_w){
  mat g;  
  for(unsigned i=0; i < v.size(); i ++){
    vector<int> row; 
    for(int j=0; j < size_w; j++ ){
      row.push_back((v[i]>>j)&1);
    }
    g.push_back(row);
  }
  return g;
}
void swap_row(mat &a, int i, int j){
  vector<int> tmp = a[i];  
  a[i] = a[j];  a[j] = tmp;
}
void add_row(mat &a, int i, int j){
  for (unsigned col_i =0; col_i < a[0].size(); col_i ++){
    a[i][col_i] ^= a[j][col_i];
  } 
}
int bin_mat_sum(const mat &A){
  int sum_val = 0;  
  for (auto row : A){
    for (int val : row){
      sum_val += val; 
    }
  } 
  return sum_val;
}
mat bin_mat_mul(const mat &A, const mat & B){
  mat out;  
  ASSERT_THROW (A.size()>0 && B.size()>0 , " mat should have row num > 0");
  int m = A.size();  
  int n = A[0].size();  
  ASSERT_THROW ((int) B.size()==n , "2nd dim of A should be same as 1st dim in B in bin mat_mul");
  // (m,n)@(n,k) --> out m,k
  int k = (int) B[0].size();  
  for(int r = 0; r<m; r++){
    vector<int> row;  
    for(int c = 0; c<k; c++){
      int val = 0;  
      for(int j = 0; j<n; j++){
        val += (A[r][j]* B[j][c]);
      }
      row.push_back(val%2);  
    }
    out.push_back(row);
  }
  return out;
}

