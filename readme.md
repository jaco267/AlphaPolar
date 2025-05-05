The AlphaZero source code is based on [minizero](https://github.com/rlglab/minizero.git), please read the readme in [minizero](https://github.com/rlglab/minizero.git) to setup the training enviroment.  

The AlphaPolar environment can be found in minizero/environment/kernsearch folder,
Currently, we didn't opensource our decoding algorithm (RMLD). You should replace the 
```c
complexity_ = 100; 
```
in minizero/environment/kernsearch.cpp with the complexity of your polar kernel decoder.   
We might opensource the viterbi decoding algorihtm in the future.   

To see the kernel generate by the AlphaPolar Agent, 
please setup the environemnt in [minizero](https://github.com/rlglab/minizero.git) 
And run the following command   

```sh
podman run --device nvidia.com/gpu=all  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --network=host --ipc=host --rm -it -v .:/workspace kds285/minizero:latest

#train a 4x4 polar kernel 
tools/quick-run.sh train kernsearch gaz 50 -conf_str \
  env_board_size=4:actor_num_simulation=16:kern_search_game_len=20:network_type=conv:rand_init_step=0 -gen kernsearch_gaz.cfg
tools/quick-run.sh train kernsearch kernsearch_gaz.cfg 100

tools/quick-run.sh console kernsearch kernsearch_gaz_1bx256_n16-19e696-dirty_A16/
# showboard  
# genmove black
# gen_multi_move black
# ctrl + c
```

Note that we've modify kernsearch_gaz_1bx256_n16-19e696-dirty_A16/kernsearch_gaz_1bx256_n16-19e696-dirty.cfg  
line 70   
```c
kern_search_game_len=1000
```
to make it work in console mode   
but in training mode this number should be set to around 250, that is   

```sh
#* ------16x16 board ----
tools/quick-run.sh train kernsearch gaz 50 -conf_str \
  env_board_size=16:actor_num_simulation=16:kern_search_game_len=250:network_type=conv:rand_init_step=7\
   -gen kernsearch_gaz.cfg
#* ------12x12board-----   
tools/quick-run.sh train kernsearch gaz 50 -conf_str   env_board_size=12:actor_num_simulation=16:kern_search_game_len=144\
:network_type=conv:rand_init_step=5 -gen kernsearch_gaz.cfg
```
