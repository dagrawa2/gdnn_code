#!/usr/bin/bash
set -e

NS=(25 50 75 100)
EPOCHS=500

for SEED in {0..23}
do
	for N in ${NS[@]}
	do
		# type 1
		python ../../train.py \
			--seed ${SEED} \
			--percentage ${N} \
			--channels 32 64 128 40 \
			--subgroup_idxs 30 15 10 1 \
			--batch_norm \
			--batch_size 64 \
			--epochs ${EPOCHS} \
			--lr 1e-2 \
			--lr_decay 0.99 \
			--lr_step 1 \
			--val_batch_size 512 \
			--val_interval 1 \
			--output_dir results/N${N}/type1/seed${SEED} \
			--exist_ok \
			--load_patterns patterns/type1.pkl \
			--device cpu
		# mixed
		python ../../train.py \
			--seed ${SEED} \
			--percentage ${N} \
			--channels 16 32 64 40 \
			--subgroup_idxs 30 15 10 1 \
			--batch_norm \
			--batch_size 64 \
			--epochs ${EPOCHS} \
			--lr 1e-2 \
			--lr_decay 0.99 \
			--lr_step 1 \
			--val_batch_size 512 \
			--val_interval 1 \
			--output_dir results/N${N}/mixed/seed${SEED} \
			--exist_ok \
			--load_patterns patterns/mixed.pkl \
			--device cpu
		# unraveled
		python ../../train.py \
			--seed ${SEED} \
			--percentage ${N} \
			--channels 16 32 64 40 \
			--subgroup_idxs 30 15 10 1 \
			--batch_norm \
			--batch_size 64 \
			--epochs ${EPOCHS} \
			--lr 1e-2 \
			--lr_decay 0.99 \
			--lr_step 1 \
			--val_batch_size 512 \
			--val_interval 1 \
			--output_dir results/N${N}/unraveled/seed${SEED} \
			--exist_ok \
			--load_patterns patterns/unraveled.pkl \
			--device cpu
	done
done

echo All done!
