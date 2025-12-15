model_name=RAFT
loss=MSE # RI_loss
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data weather \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --loss $loss \
  --itr 1


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --model $model_name \
  --data weather \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --loss $loss \
  --itr 1


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model $model_name \
  --data weather \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --loss $loss \
  --itr 1


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model $model_name \
  --data weather \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --loss $loss \
  --itr 1

