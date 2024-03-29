# PIE-G

Codebase for NeurIPS 2022: "Pre-Trained Image Encoder for Generalizable Visual
Reinforcement Learning"



<p align="left">
    <img src="https://anonymous.4open.science/r/PIE-G-A80B/figures/ball_video_hard_882.gif" width = "190" height = "190" >
    <img src="https://anonymous.4open.science/r/PIE-G-A80B/figures/cheetah_video_hard_600.gif" width = "190" height = "190"  >
    <img src="https://anonymous.4open.science/r/PIE-G-A80B/figures/stand_video_hard_958.gif" width = "190" height = "190"  >
    <img src="https://anonymous.4open.science/r/PIE-G-A80B/figures/walk_video_hard_801.gif" width = "190" height = "190"  >
    <img src="https://anonymous.4open.science/r/PIE-G-A80B/figures/finger_video_hard_905.gif" width = "190" height = "190"  >
</p>

## Setup

The MuJoCo license and instruction can be found at https://github.com/deepmind/mujoco;The Mujoco Version should be `2.1.1` for training.

The DeepMind Control license and instruction can be found at https://github.com/deepmind/dm_control

For training, the dependencies are based on DrQ-v2. You can install them with the following commands:

```
conda env create -f conda_env.yml
```
Detailed installation instructions can be found at: https://github.com/facebookresearch/drqv2


For generalization testing, we use the DMControl Gneralization Benchmark.  You can run the commands as follows:

```
cd dmcontrol-generalization-benchmark/
conda env create -f setup/dmcgb.yml
conda activate dmcgb
sh setup/install_envs.sh
```

The place365 dataset can be downloaded by running:
```
wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
```
After downloading and extracting the data, add your dataset directory to the `config.cfg`. 

Detailed installation instructions can be found at: https://github.com/nicklashansen/dmcontrol-generalization-benchmark






## Training and Evaluation

### Training

`pieg` conda environment is served for training, so you should activate this conda env at first:

```
conda activate pieg
bash pieg_train.sh task=walker_walk seed=1
```
`cd` to the `exp_local` file and move the trained model to the test file:
```
mv snapshot.pt ~/PIE-G/dmcontrol-generalization-benchmark/logs/walker_walk/pieg/1
```



### Evaluation

```
cd ~/PIEG/dmcontrol-generalization-benchmark/
conda activate dmcgb
bash script/eval/pieg.sh 1 video_hard walker walk
```





## License

The majority of PIE-G is licensed under the MIT license, however portions of the project are available under separate license terms: DeepMind is licensed under the Apache 2.0 license.

