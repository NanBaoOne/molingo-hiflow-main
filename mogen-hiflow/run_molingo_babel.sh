torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  mogen-hiflow/train_molingo_babel.py \
  --dataset_name babel \
  --vae sae_bs512_lr1p5e4_l2_2_32_1024_d3_kl_1e-05_zero_cos_0.001 \
  --batch_size 128 \
  --exp_name molingo_bs128 \
  --model_size large \
  --sample_steps 32 \
  --adapter_layers 6 \
  --t5_max_len 64 \
  --cfg 4.0 \
  --temperature 1.0 \
  --acc_ratio 5.0 \
  --eval_every_e 5 \
  --save_every_e 5 \
  --second_stage molingo \
  --grad_checkpointing
