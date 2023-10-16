#!/bin/bash
export PYTHONUNBUFFERED=1
pretrain_path='bert-base-uncased'
train_file='benchmark/nyt10m/nyt10m_train.txt'
val_file='benchmark/nyt10m/nyt10m_val.txt'
test_file='benchmark/nyt10m/nyt10m_test.txt'
rel2id_file='benchmark/nyt10m/nyt10m_rel2id.json'
batch_size=16
max_length=258
max_epoch=3
seed=772
name=$pretrain_path'_'$max_length'_'$batch_size'_'$max_epoch'_'$lr'_'$seed'_nyt10m_sep_na'
echo $name
python main.py --pretrain_path $pretrain_path --train_file $train_file --val_file $val_file --test_file $test_file --rel2id_file $rel2id_file --batch_size $batch_size --max_epoch $max_epoch --max_length $max_length --ckpt $name --devs 0 --seed $seed --only_test
