# CUDA_VISIBLE_DEVICES=0, python train_cnn.py \
# --model Cnn6 \
# --mode Va \
# --fold 0 \
# --CV ALL

# CUDA_VISIBLE_DEVICES=0, python train_bert.py \
# --model BERT \
# --mode Va \
# --fold 0 \
# --CV ALL

# CUDA_VISIBLE_DEVICES=0, python train_mix.py \
# --mode Va \
# --path1 ./model/ALL-Audio_Va_Cnn6_fold-0/best_net.pt \
# --path2 ./model/ALL-Lyrics_Va_BERT_fold-0/best_net.pt
