The AlphaZero source code is based on [minizero](https://github.com/rlglab/minizero.git), please read the readme in [minizero](https://github.com/rlglab/minizero.git) to setup the training enviroment.  

The AlphaPolar environment can be found in environment/kernsearch folder,
Currently, we didn't opensource our decoding algorithm (RMLD),    


```sh
podman run --device nvidia.com/gpu=all  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --network=host --ipc=host --rm -it -v .:/workspace kds285/minizero:latest

tools/quick-run.sh train kernsearch gaz 50 -conf_str \
  env_board_size=4:actor_num_simulation=16:kern_search_game_len=20:network_type=conv:rand_init_step=0 -gen kernsearch_gaz.cfg
tools/quick-run.sh train kernsearch kernsearch_gaz.cfg 100

tools/quick-run.sh console kernsearch kernsearch_gaz_1bx256_n16-19e696-dirty_A16/
# showboard  
# genmove black
# gen_multi_move black
# ctrl + c
```