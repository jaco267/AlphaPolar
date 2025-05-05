#pragma once
#include <vector>
#include <iostream>
#include "comp_defines.h"
using std::vector;
using std::cout;
using std::endl;


//* ----np func----
vector<int> arange(int start, int end, int step = 1);
mat eye(int k);

mat hstack3(const mat & A, const mat & B, const mat& C);

vector<int> setdiff1d(const vector<int> &input, const vector<int> &diff_list);
void setdiff1d_rev(vector<int> &ones_rows, vector<int> &fixed_rows_list 
     ,vector<int> &del_rows);
//* -----basic---------
int bin_to_dec(const vector<int>& bin_list);
vector<int> dec_to_bin(int n, int x);
vector<int> mat2comp_vec(const mat& g);
mat comp_vec2mat(const vector<int> & v,int size_w);

void swap_row(mat &a, int i, int j);
void add_row(mat &a, int i, int j);

// void print_mat(mat G);
template <typename T>
void print_mat(vector<vector<T>> G){
    for (int row =0 ; row<G.size(); row++){
        for (int col=0; col<G[0].size(); col++){cout<<G[row][col]<<" ";} cout<<endl;
    }
}

// void print_vec(vector<int> v);
template <typename T>
void print_vec(vector<T> v){
//   for (int row =0 ; row<v.size(); row++){cout<<v[row]<<" "; } cout<<endl;
  for (const T& elem : v) {cout << elem << " ";}cout << endl;
} 

int bin_mat_sum(const mat &A);
mat bin_mat_mul(const mat &A, const mat & B);

