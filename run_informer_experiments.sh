#!/bin/bash

MODEL="informer"
DATA="custom"
ROOT_PATH="/root/repo/Informer2020/data_cleaned"
DATA_PATH="sentiment_all_stock_price.csv"
SEQ_LEN=96
LABEL_LEN=48
BATCH_SIZE=32
TRAIN_EPOCHS=6
LEARNING_RATE=0.0001
D_MODEL=512
N_HEADS=8
E_LAYERS=2
D_LAYERS=1
D_FF=2048
FACTOR=5
DROPOUT=0.05
ATTN="prob"
EMBED="timeF"
FREQ="b"
GPU=0
OT="Apple_Price"

LOG_DIR="./logs"
mkdir -p $LOG_DIR

# # univariate
# echo "Running univariate forecasting experiments..."
# FEATURES="S"
# ENC_IN=1
# DEC_IN=1
# C_OUT=1

# for PRED_LEN in 24 ; do
#     echo "Running experiment with pred_len=$PRED_LEN (Univariate)"
#     python main_informer.py \
#         --model $MODEL \
#         --data $DATA \
#         --features $FEATURES \
#         --target $OT \
#         --seq_len $SEQ_LEN \
#         --label_len $LABEL_LEN \
#         --pred_len $PRED_LEN \
#         --enc_in $ENC_IN \
#         --dec_in $DEC_IN \
#         --c_out $C_OUT \
#         --d_model $D_MODEL \
#         --n_heads $N_HEADS \
#         --e_layers $E_LAYERS \
#         --d_layers $D_LAYERS \
#         --d_ff $D_FF \
#         --factor $FACTOR \
#         --dropout $DROPOUT \
#         --attn $ATTN \
#         --embed $EMBED \
#         --freq $FREQ \
#         --batch_size $BATCH_SIZE \
#         --train_epochs $TRAIN_EPOCHS \
#         --learning_rate $LEARNING_RATE \
#         --root_path $ROOT_PATH \
#         --data_path $DATA_PATH \
#         --gpu $GPU \
#         > $LOG_DIR/univariate_pred_len_${PRED_LEN}.log
# done

# multivariate
echo "Running multivariate forecasting experiments..."
FEATURES="MS"
ENC_IN=30
DEC_IN=30
C_OUT=1

for PRED_LEN in 24 ; do
    echo "Running experiment with pred_len=$PRED_LEN (Multivariate)"
    python main_informer.py \
        --model $MODEL \
        --data $DATA \
        --features $FEATURES \
        --target $OT \
        --seq_len $SEQ_LEN \
        --label_len $LABEL_LEN \
        --pred_len $PRED_LEN \
        --enc_in $ENC_IN \
        --dec_in $DEC_IN \
        --c_out $C_OUT \
        --d_model $D_MODEL \
        --n_heads $N_HEADS \
        --e_layers $E_LAYERS \
        --d_layers $D_LAYERS \
        --d_ff $D_FF \
        --factor $FACTOR \
        --dropout $DROPOUT \
        --attn $ATTN \
        --embed $EMBED \
        --freq $FREQ \
        --batch_size $BATCH_SIZE \
        --train_epochs $TRAIN_EPOCHS \
        --learning_rate $LEARNING_RATE \
        --root_path $ROOT_PATH \
        --data_path $DATA_PATH \
        --gpu $GPU \
        > $LOG_DIR/multivariate_pred_len_${PRED_LEN}.log
done

echo "All experiments completed. Logs are saved in $LOG_DIR."