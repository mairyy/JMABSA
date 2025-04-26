for sl in  '7.5e-5' #You can change the sl to find the best hyperparameter.
do
		echo ${sl}
		python MAESC_training.py \
          --dataset twitter15 ./src/data/jsons/twitter15_info.json \
          --checkpoint_dir ./train15 \
          --model_config config/pretrain_base.json \
          --log_dir 15_aesc \
          --num_beams 4 \
          --eval_every 1 \
          --lr ${sl} \
          --train_batch_size 8  \
          --dev_batch_size 8  \
          --test_batch_size 8  \
          --epochs 40 \
          --grad_clip 5 \
          --warmup 0.1 \
          --seed 57 \
          --rank 2 \
          --trc_pretrain_file ./src/AoM-ckpt/Twitter2015/pytorch_model.bin \
          --nn_attention_on \
          --nn_attention_mode 0\
          --gcn_on \
          --dep_mode 2\
          --gpu_num 2\
          --gcn_layer_num 2\
          --text_encoder bart\
          --aesc_enabled true\
          --img_num 49\
          --trc_on\
          --task AESC\
          --w_l 0.8
done
