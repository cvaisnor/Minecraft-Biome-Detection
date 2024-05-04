## Identifying Minecraft Biomes with Transfer Learning and Fine-tuning

### By: Chris Vaisnor

### System Setup
- OS: Ubuntu 22.04.2 LTS
- Python Version: 3.10.12
- PyTorch Version:  2.1.0+cu121

### Repository Structure
- `frames_biomes/`: Contains the frames used to train the model
- `preprocessing.py`: Script to convert video to frames
    - If you would like to make the data yourself by using screen recordings, you can use this script to convert the video to frames. The following structure is required for the videos:
```
Original Video Structure
- /data
    - /category1
        - category1_v0.mkv
        ....
        - category1_v1.mkv
    - /cateogry2
        - category2_v0.mkv
        ....
        - category2_v1.mkv
    - ....
```

The preprocessing script will convert the videos to frames and store them in the following structure:
```
Frame Storage Structure
- /frames
    - train
        - /category1
            - frame0.jpg
            ....
            - frameN.jpg
        - /category2
            - frame0.jpg
            ....
            - frameN.jpg
        - ....
    - test
        - /category1
            - frame0.jpg
            ....
            - frameN.jpg
        - /category2
            - frame0.jpg
            ....
            - frameN.jpg
        - ....
```

Once the frames are stored in the correct structure, you can use main.ipynb to walk through the process of training and testing the model.

Various ResNet models can be choosen as the base with their pretraining weights initialized.

Transfer learning and fine-tuning is used to train the model on distinct 10 Minecraft biomes. The dataset consists of 10 classes:
- Desert
- Birch Forest
- Jungle 
- Crimson Forest
- Dark Forest
- End Midlands
- Savanna
- Snowy Taiga
- Swamp
- Plains. 

200 frames are uniformly sampled from each video to create the dataset. The dataset is split into 80% training and 20% testing. The model is trained on the training set and tested on the testing set. The model is evaluated using the accuracy metric.# Minecraft-Biome-Detection-with-ResNet
