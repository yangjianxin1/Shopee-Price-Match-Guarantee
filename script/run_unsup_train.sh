python train.py \
    --device gpu \
    --output_path output \
    --lr 3e-5 \
    --dropout 0.2 \
    --epochs 10 \
    --batch_size_train 64 \
    --batch_size_eval 256 \
    --num_workers 0 \
    --eval_step 50 \
    --max_len 150 \
    --seed 42 \
    --train_file data/shopee/train.csv \
    --dev_file data/shopee/dev.csv \
    --test_file data/shopee/test.csv \
    --pretrain_model_path pretrain_model/bert-base-uncased \
    --pooler cls \
    --train_mode unsupervise \
    --overwrite_cache \
    --do_train \
    --do_predict