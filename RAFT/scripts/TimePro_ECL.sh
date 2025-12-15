model_name=RAFT
loss=MSE # RI_loss

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id electricity_96_96 \
  --model $model_name \
  --data ECL \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --batch_size 32 \
  --itr 1 \
  --loss $loss \
  --learning_rate 0.0001

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id electricity_96_192 \
  --model $model_name \
  --data ECL \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --batch_size 32 \
  --itr 1 \
  --loss $loss \
  --learning_rate 0.0001

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id electricity_96_336 \
  --model $model_name \
  --data ECL \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --batch_size 32 \
  --itr 1 \
  --loss $loss \
  --learning_rate 0.0001

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id electricity_96_720 \
  --model $model_name \
  --data ECL \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --batch_size 32 \
  --itr 1 \
  --loss $loss \
  --learning_rate 0.0001