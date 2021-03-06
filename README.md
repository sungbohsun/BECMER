# BECMER: A Fusion Model Using BERT and CNN for Music Emotion Recognition

#### Music emotion analysis has been an ever-growing field of research in music information retrieval. To solve the cold start problem of content-based recommendation systems, a method for automatic music labeling is needed. Due to recent advances, neural networks can be used to extract audio features for a wide variety of tasks. When humans listen to a song, it is the music or the lyrics that touch the heartthe most. Therefore, this study will try to predict the type of music emotion based on the audio signal and the lyrics information. For model building, convolutional neural networks (CNNs) will be used on the audio signals and natural language processing (NLP) models on the lyrics. A new dataset ABP is compiled from three datasets of Western pop music where each song contains valence and arousal values judged by humans. The type of music emotion will be categorized based on the four quadrants formed by the valence and arousal axes. It is confirmed in the experiment that use of audio and lyrics information to classify the emotions of songs has a better classification performance than use of the audio-only learning methods in previous  studies. Compared with a related work, this study has improved the accuracy of the audio model and the lyrics model by 8~16%.

![GITHUB](https://github.com/sungbohsun/BECMER/blob/main/BECMER.png)

## Datasets
* 4Q audio emotion dataset (Russell's model) (2018) http://mir.dei.uc.pt/downloads.html
* Bi-modal (audio and lyrics) emotion dataset (Russell's model) (2016) http://mir.dei.uc.pt/downloads.html
* PMEmo: A Dataset For Music Emotion Computing https://github.com/HuiZhangDB/PMEmo

```bash
data
├── MER_bi
│   ├── chorus
│   ├── MER-Audio-dataset.xls
│   ├── MER-Bimodal-Dataset.xls
│   └── MER_lyrics_dataset.xls
├── PME
│   ├── annotations
│   ├── chorus
│   ├── comments
│   ├── EDA
│   ├── features
│   ├── lyrics
│   ├── metadata.csv
│   └── netease_soundcloud.csv
├── Q4_MER
│   ├── chorus
│   ├── panda_dataset_taffc_annotations.csv
│   ├── panda_dataset_taffc_metadata.csv
│   ├── Q1
│   ├── Q2
│   ├── Q3
│   ├── Q4
│   └── readme.txt
```
## Data proccess
* setup_Bi.ipynb
* setup_PME&4Q.ipynb
* setup_lyrics.ipynb

## Proccessed data

https://drive.google.com/drive/folders/1xC72Ul4Qx1FvCkhg208rNwrqXcJAd3bq?usp=sharing

```bash
data
├── dic_Q4.pt
├── dic_Q4_lyrics.pt
├── dic_PME.pt
├── dic_PME_lyrics.pt
├── dic_Bi.pt
├── dic_Bi_lyrics.pt
```

## Pretrained model 

https://drive.google.com/drive/folders/1SXmfYa3eDrU_mcPWwTj0Z9HGxEIVLWUz?usp=sharing

```bash
model
├── ALL-Audio_4Q_Cnn10_fold-0
│   ├── best_net.pt
├── ALL-Lyrics_4Q_ALBERT_fold-0
│   ├── best_net.pt
├── ALL-MIX_4Q_Cnn10_ALBERT_fold-0
│   ├── best_net.pt
```
## Training Demo
* CNN model: Cnn6/Cnn10 (model.py)
* BERT model: BERT/ALBERT (model_bert.py)
* mode:  Ar/Va/4Q 
* ![GITHUB](https://github.com/sungbohsun/BECMER/blob/main/ar-va.png)
* k-fold: number
* Cross validation: ALL/PME
* ![GITHUB](https://github.com/sungbohsun/BECMER/blob/main/cv.png)
```python
#CNN
CUDA_VISIBLE_DEVICES=0, python train_cnn.py \
--model Cnn10 \
--mode 4Q \
--fold 0 \
--CV ALL

#BERT
CUDA_VISIBLE_DEVICES=0, python train_bert.py \
--model BERT \
--mode 4Q \
--fold 0 \
--CV ALL

#Fusion
CUDA_VISIBLE_DEVICES=0, python train_mix.py \
--mode Va \
--path1 ./model/ALL-Audio_4Q_Cnn10_fold-0/best_net.pt \
--path2 ./model/ALL-Lyrics_4Q_ALBERT_fold-0/best_net.pt
```

## Test Demo
```python
CUDA_VISIBLE_DEVICES=2, python test_cnn.py  --path ./model/ALL-Audio_4Q_Cnn10_fold-0/best_net.pt
CUDA_VISIBLE_DEVICES=2, python test_bert.py --path ./model/ALL-Lyrics_4Q_ALBERT_fold-0/best_net.pt
CUDA_VISIBLE_DEVICES=2, python test_mix.py  --path ./model/ALL-MIX_4Q_Cnn10_ALBERT_fold-0/best_net.pt
```
