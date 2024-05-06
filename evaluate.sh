#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
HOME_DIR=`realpath .`

function evaluate () {
    EVAL_DATASET=$1
    echo $MODEL_DIR/${EVAL_DATASET}
    python -W ignore utils/evaluate.py \
        --src_dir ${DATA_DIR_PREFIX}/${EVAL_DATASET}/fairseq \
        --file_prefix $MODEL_DIR/${EVAL_DATASET}/kp20k \
        --tgt_dir $MODEL_DIR/${EVAL_DATASET} \
        --log_file $EVAL_DATASET \
        --k_list 5 M;
}


DATA_DIR_PREFIX=${HOME_DIR}/data/scikp
MODEL_DIR=${HOME_DIR}/save_models/bart/triplet-sa_scikp
evaluate kp20k


DATA_DIR_PREFIX=${HOME_DIR}/data/scikp
MODEL_DIR=${HOME_DIR}/save_models/bart/triplet-aa_scikp-5epoch
evaluate kp20k