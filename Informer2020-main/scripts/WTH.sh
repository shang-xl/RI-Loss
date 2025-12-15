if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model=informer
loss=mse  #RI-Loss
seq_len=96
root_path=./data/WTH/
data_path=weather.csv
data=WTH

for pred_len in 96 192 336 720
do
    python -u main_informer.py \
      --root_path $root_path \
      --data_path $data_path \
      --model $model \
      --data $data \
      --features M \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 3 \
      --d_layers 2 \
      --factor 5 \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --des 'Exp' \
      --itr 1 \
      --loss $loss >logs/LongForecasting/$model'_'WTH'_'$seq_len'_'48'_'$pred_len'_'$loss.log
done