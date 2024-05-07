## Minecraft Biome Detection w/ Transfer Learning and Fine-tuning

### By: Chris Vaisnor

### System Setup
- OS: Ubuntu 22.04.2 LTS
- Python Version: 3.10.12
- PyTorch Version:  2.1.0+cu121

### Repository Structure
- `biome_videos/`: Contains the raw videos for the dataset
```
Original Video Structure
- /biome_videos
    - /category1
        - video.mkv
    - /cateogry2
        - video.mkv
    - ....
```
- This format should work if multiple videos are present for each category.
- `preprocessing.py`: Script to convert videos to frames and store them in the following structure
```
Frame Storage Structure
- /frames
    - train
        - /category1
            - frame0.jpg
            ....
            - frameN.jpg
        - /category2
        - ....
    - test
        - /category1
            - frame0.jpg
            ....
            - frameN.jpg
        - /category2
        - ....
```

Example Usage:
```bash
python preprocessing.py -dd biome_videos -fd frames -nf 200
```
Frames are uniformly sampled from each video to create the dataset. The dataset is split into 80% training and 20% testing.
Once the frames are stored in the correct structure, you can use main.ipynb to walk through the process of training and testing the model.
- Various ResNet models can be initialized with pretrained weights from the ImageNet dataset.

Transfer learning and fine-tuning is used to train the model on 10 distinct Minecraft biomes. The dataset consists of 10 classes:
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