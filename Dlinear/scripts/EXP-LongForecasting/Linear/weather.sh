
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=DLinear
loss=MSE # RI_loss

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 21 \
  --des 'Exp' \
  --loss $loss \
  --itr 1 --batch_size 16 --learning_rate 0.005  >logs/LongForecasting/$model_name'_'Weather_$seq_len'_'96_$loss.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 21 \
  --des 'Exp' \
  --loss $loss \
  --itr 1 --batch_size 16 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'Weather_$seq_len'_'192_$loss.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/WTH \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 21 \
  --des 'Exp' \
  --loss $loss \
  --itr 1 --batch_size 16 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'Weather_$seq_len'_'336_$loss.log

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 21 \
  --des 'Exp' \
  --loss $loss \
  --itr 1 --batch_size 16 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'Weather_$seq_len'_'720_$loss.log
