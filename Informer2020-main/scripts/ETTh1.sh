if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model=informer
loss=mse # RI_loss
seq_len=96
root_path=./data/ETT/

python -u main_informer.py \
  --root_path $root_path \
  --data_path ETTh1.csv \
  --model $model \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --attn prob \
  --factor 3 \
  --loss $loss \
  --des 'Exp' --itr 1 >logs/LongForecasting/$model'_'Etth1_96_48_96_$my_loss.log

python -u main_informer.py \
  --root_path $root_path \
  --data_path ETTh1.csv \
  --model $model \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --attn prob \
  --loss $loss \
  --des 'Exp' --itr 1 >logs/LongForecasting/$model'_'Etth1_96_48_192_$my_loss.log

python -u main_informer.py \
  --root_path $root_path \
  --data_path ETTh1.csv \
  --model $model \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --attn prob \
  --loss $loss \
  --des 'Exp' --itr 1 >logs/LongForecasting/$model'_'Etth1_96_48_336_$my_loss.log

python -u main_informer.py \
  --root_path $root_path \
  --data_path ETTh1.csv \
  --model $model \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --attn prob \
  --loss $loss \
  --des 'Exp' --itr 1 >logs/LongForecasting/$model'_'Etth1_96_48_720_$my_loss.log