if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model=informer
loss=mse # RI_loss

root_path=./data/ETT/
data_path=ETTh2.csv
data=ETTh2

for pred_len in  96 192 336 720
do
      python -u main_informer.py \
        --root_path $root_path \
        --data_path $data_path \
        --model $model \
        --data $data \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len\
        --e_layers 2 \
        --d_layers 1 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --itr 1 \
        --loss $loss >logs/LongForecasting/$model'_'96'_'48'_'$pred_len'_'$loss.log
done
