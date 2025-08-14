# HEAD: Hand-Eye Autonomous Delivery: Learning Humanoid Navigation, Locomotion and Reaching


Sirui Chen*, Yufei Ye*, Zi-Ang Cao*, Jennifer Lew, Pei Xu, C. Karen Liu, in CoRL 2025

[Project Page](https://stanford-tml.github.io/HEAD/)
| [Arxiv](https://arxiv.org/abs/2508.03068) | [Code of Other Compoenents](https://github.com/Stanford-TML/HEAD_release)

This repo contains code to train/test/deploy navigation module and instructions to collect human data from aria glasses. 


## Installation

```
bash scripts/one_click.sh
```

## Inference
Download our model and put it under `weights/`. 
| Training Data | Model |
| --- | --- |
| All Data (Robot+Human+ADT) | [gdrive]() |


We provide demo robot data collected in the Kitchen room. 


```
python -m demo -m  expname=release/mix_data num=-1  vis=True ds=g1-kit 
```
You should be see similar output.
![image]()


## Deploy to G1
[ ] todo 


## Collect Your Own Aria Data and Train Your Own Model
### Data Processing
This section instructs you how to collect your own data from [Aria Glasses](https://www.projectaria.com/).
- **Record your data**. Follow Aria Glasses tutorial. Set fps to 10, with SLAM and eye gaze turned on. Copy `*.vrs` and request MPS and put them under `data/your_own_data_name/raw`
    ```
    data/your_own_data_name/raw/
        seq1.vrs
        seq1.vrs.json
        mps_seq1_vrs/
        ...
    ```
- **Extract video frames**. Decode images from vrs
    ```
    python -m preprocess.decode_image --raw_dir data/your_own_data_name/raw
    ```

- **Mark key frames**.  We provide a simple UI to split the videos to only contain valid clips. A valid clip is a consective navigation trajectory.  You need a display to run this. It can run on CPUs such as Mac laptop.
    ```
    python preprocess/anno_clip.py
    ```
    For example, here are two clips from the same videos.
    | clip 1 | clip 1 |
    |---|---|
    | | |

- **Convert to ready-to-train data**.
    + For each clip, it uses SLAM, gaze, vrs data and convert them to format that are ready to train

    + Approximate goal by gaze.
    ```
    python cvt_clip.py --data_dir data/your_own_data_name/ --goal_mode pc --vis True
    ```
    + Augment images by fixed pitch angles. 
    ```
    python cvt_clip.py --data_dir data/your_own_data_name/ --goal_mode aug  --vis True
    ```

    + --vis True visualizes preprocessed data for sanity check. You should be able to see output similar to this.
    ![images]()
    
    + **Create split**. 
    ```
    python cvt_clip.py --mode split --data_dir data/your_own_data_name/  --val_seq_list seq1,seq2  # validation sequence index, separate with comma
    ```


### Start Training
- Register your new datasets.
    + Register your dataset path in [`dataset_config/datasets.yaml`](dataset_config/datasets.yaml)
    + Mix it into training datasets. [`config/dataset/mix_all.yaml`](config/dataset/mix_all.yaml)


- Now you are ready to train your model! 
    ```
    python -m train -m expname=dev/new_model dataset=mix_all [checkpoint=PATH_TO_PRETRAIN_MODEL/checkpoints/last.ckpt]
    ```