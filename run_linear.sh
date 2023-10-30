if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
  mkdir ./logs/LongForecasting
fi
gpu=0
station_type=adaptive
features=M

for model_name in DLinear; do
  for pred_len in 96 192 336 720; do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/exchange_rate \
      --data_path exchange_rate.csv \
      --model_id exchange_336_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len 336 \
      --label_len 168 \
      --pred_len $pred_len \
      --enc_in 8 \
      --dec_in 8 \
      --c_out 8 \
      --des 'Exp' \
      --gpu $gpu \
      --station_type $station_type \
      --period_len 6 \
      --learning_rate 0.001 \
      --station_lr 0.001 \
      --itr 3 >logs/LongForecasting/$model_name'_exchange_rate_'$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/electricity \
      --data_path electricity.csv \
      --model_id electricity_336_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len 336 \
      --label_len 168 \
      --pred_len $pred_len \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --gpu $gpu \
      --station_type $station_type \
      --period_len 24 \
      --learning_rate 0.001 \
      --station_lr 0.0001 \
      --itr 3 >logs/LongForecasting/$model_name'_electricity_'$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/traffic \
      --data_path traffic.csv \
      --model_id traffic_336_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len 336 \
      --label_len 168 \
      --pred_len $pred_len \
      --enc_in 862 \
      --dec_in 862 \
      --c_out 862 \
      --des 'Exp' \
      --itr 3 \
      --gpu $gpu \
      --period_len 24 \
      --learning_rate 0.05 \
      --station_lr 0.0001 \
      --station_type $station_type >logs/LongForecasting/$model_name'_traffic_'$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/weather \
      --data_path weather.csv \
      --model_id weather_336_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len 336 \
      --label_len 168 \
      --pred_len $pred_len \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --des 'Exp' \
      --itr 3 \
      --gpu $gpu \
      --period_len 12 \
      --learning_rate 0.0001 \
      --station_lr 0.0001 \
      --station_type $station_type >logs/LongForecasting/$model_name'_weather_'$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/ETT-small \
      --data_path ETTh1.csv \
      --model_id ETTh1_336_$pred_len \
      --model $model_name \
      --data ETTh1 \
      --features $features \
      --seq_len 336 \
      --label_len 168 \
      --pred_len $pred_len \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --gpu $gpu \
      --station_type $station_type \
      --period_len 24 \
      --learning_rate 0.005 \
      --station_lr 0.0001 \
      --itr 3 >logs/LongForecasting/$model_name'_Etth1_'$pred_len.log
    #
    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/ETT-small \
      --data_path ETTh2.csv \
      --model_id ETTh2_336_$pred_len \
      --model $model_name \
      --data ETTh2 \
      --features $features \
      --seq_len 336 \
      --label_len 168 \
      --pred_len $pred_len \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --gpu $gpu \
      --station_type $station_type \
      --period_len 24 \
      --learning_rate 0.05 \
      --station_lr 0.00005 \
      --itr 3 >logs/LongForecasting/$model_name'_Etth2_'$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/ETT-small \
      --data_path ETTm1.csv \
      --model_id ETTm1_336_$pred_len \
      --model $model_name \
      --data ETTm1 \
      --features $features \
      --seq_len 336 \
      --label_len 168 \
      --pred_len $pred_len \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --gpu $gpu \
      --station_type $station_type \
      --period_len 12 \
      --learning_rate 0.0001 \
      --station_lr 0.0001 \
      --itr 3 >logs/LongForecasting/$model_name'_Ettm1_'$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/ETT-small \
      --data_path ETTm2.csv \
      --model_id ETTm2_336_$pred_len \
      --model $model_name \
      --data ETTm2 \
      --features $features \
      --seq_len 336 \
      --label_len 168 \
      --pred_len $pred_len \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --gpu $gpu \
      --station_type $station_type \
      --period_len 12 \
      --learning_rate 0.01 \
      --station_lr 0.00005 \
      --itr 3 >logs/LongForecasting/$model_name'_Ettm2_'$pred_len.log
  done
done

for model_name in DLinear; do
  for pred_len in 24 36 48 60; do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/illness \
      --data_path national_illness.csv \
      --model_id ili_96_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $pred_len \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --period_len 6 \
      --gpu $gpu \
      --station_type None \
      --learning_rate 0.01 \
      --station_lr 0.0005 \
      --itr 3 >logs/LongForecasting/$model_name'_ili_None_'$pred_len.log
  done
done
