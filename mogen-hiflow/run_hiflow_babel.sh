torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  mogen-hiflow/train_molingo_babel.py \
  --dataset_name babel \
  --vae sae_bs512_lr1p5e4_l2_2_32_1024_d3_kl_1e-05_zero_cos_0.001 \
  --batch_size 64 \
  --exp_name hiflow_bs64_D0729 \
  --model_size large \
  --second_stage hiflow \
  --checkpoints_dir checkpoints \
  --hiflow_scales 0.3,0.6,1.0 \
  --eval_every_e 10 \
  --eval_start_epoch 200 \
  --time_patch 2
