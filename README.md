# PIE-G

Codebase for NIPS2022: "Pre-Trained Image Encoder for Generalizable Visual
Reinforcement Learning"



## Code Stucture





## Setup

The MuJoCo license and instruction can be found at https://github.com/deepmind/mujoco

The DeepMind Control license and instruction can be found athttps://github.com/deepmind/dm_control

For training, the dependencies are based on DrQ-v2. You can install them with the following commands:

```
conda env create -f resdrqv2/conda.yml
```







For generalization testing, we use the DMControl Gneraliztaion Benchmark.  You can run the commands as follows:

```
cd dmcontrol-generalization-benchmark/
conda env create -f setup/dmcgb.yml
conda activate dmcgb
sh setup/install_envs.sh
```



## Datasets

The same with prior methods, we choose the Place dataset for data augmentation:

```
wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
```

After downloading and extracting the data, add your dataset directory to the `datasets` list in `setup/config.cfg`.



## Training and Evaluation

### Training

resdrqv2 conda environment is served for training, so you should activate this conda env at first:

```
conda activate pieg
bash script/pieg-train.sh
```

resdrqv2 with augmentation:

```
bash script/pieg-train.sh
```



### Evaluation

```
conda activate dmcgb
bash eval/script/pieg.sh 1 video_hard walker walk
```





## License

The majority of PIE-G is licensed under the MIT license, however portions of the project are available under separate license terms: DeepMind is licensed under the Apache 2.0 license.

