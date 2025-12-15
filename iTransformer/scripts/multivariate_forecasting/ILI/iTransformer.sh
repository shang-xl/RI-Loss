model_name=iTransformer
loss=MSE  # RI_loss

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --model_id ili_36_24 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 36 \
  --pred_len 24 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --loss $loss \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --model_id ili_36_36 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 36 \
  --pred_len 36 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --loss $loss \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --model_id ili_36_48 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 36 \
  --pred_len 48 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --loss $loss \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --model_id ili_36_60 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 36 \
  --pred_len 60 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --loss $loss \
  --itr 1