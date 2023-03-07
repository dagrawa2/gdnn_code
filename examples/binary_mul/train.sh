#!/usr/bin/bash
set -e

EPOCHS=5

for SEED in {0..23}
do
	# type 1
	python ../../train.py \
		--seed ${SEED} \
		--channels 1 1 1 2 1 \
		--subgroup_idxs 2 2 2 1 1 \
		--batch_size 64 \
		--epochs ${EPOCHS} \
		--lr 1e-2 \
		--lr_decay 0.99 \
		--lr_step 1 \
		--val_train \
		--val_batch_size 500 \
		--val_interval 1 \
		--output_dir results/type1/seed${SEED} \
		--exist_ok \
		--load_reps data/reps.pkl \
		--load_patterns patterns/type1.pkl \
		--save_params \
		--device cpu
	# type 2
	python ../../train.py \
		--seed ${SEED} \
		--channels 1 1 1 2 1 \
		--subgroup_idxs 2 2 2 1 1 \
		--batch_size 64 \
		--epochs ${EPOCHS} \
		--lr 1e-2 \
		--lr_decay 0.99 \
		--lr_step 1 \
		--val_train \
		--val_batch_size 500 \
		--val_interval 1 \
		--output_dir results/type2/seed${SEED} \
		--exist_ok \
		--load_reps data/reps.pkl \
		--load_patterns patterns/type2.pkl \
		--save_initial \
		--save_params \
		--device cpu
	# unraveled (random init)
	python ../../train.py \
		--seed ${SEED} \
		--channels 1 1 1 2 1 \
		--subgroup_idxs 2 2 2 1 1 \
		--batch_size 64 \
		--epochs ${EPOCHS} \
		--lr 1e-2 \
		--lr_decay 0.99 \
		--lr_step 1 \
		--val_train \
		--val_batch_size 500 \
		--val_interval 1 \
		--output_dir results/unraveled_init-random/seed${SEED} \
		--exist_ok \
		--load_reps data/reps.pkl \
		--load_patterns patterns/unraveled.pkl \
		--save_params \
		--device cpu
	# unraveled (random init)
	python ../../train.py \
		--seed ${SEED} \
		--channels 1 1 1 2 1 \
		--subgroup_idxs 2 2 2 1 1 \
		--batch_size 64 \
		--epochs ${EPOCHS} \
		--lr 1e-2 \
		--lr_decay 0.99 \
		--lr_step 1 \
		--val_train \
		--val_batch_size 500 \
		--val_interval 1 \
		--output_dir results/unraveled_init-type2/seed${SEED} \
		--exist_ok \
		--load_reps data/reps.pkl \
		--load_patterns patterns/unraveled.pkl \
		--save_params \
		--special_init results/type2/seed${SEED}/initial.pth \
		--device cpu
done

echo All done!
