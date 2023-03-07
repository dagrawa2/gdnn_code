#!/usr/bin/bash
set -e

# type 1
echo Generating type 1 patterns . . .
python ../../train.py \
	--tunnel_reps \
	--seed 0 \
	--channels 1 1 1 2 1 \
	--subgroup_idxs 2 2 2 1 1 \
	--batch_size 64 \
	--epochs 0 \
	--val_interval 0 \
	--output_dir temp \
	--load_reps data/reps.pkl \
	--save_patterns patterns/type1.pkl \
	--device cpu \
	> /dev/null

rm -r temp


# type 2
echo Generating type 2 patterns . . .
python ../../train.py \
	--seed 0 \
	--channels 1 1 1 2 1 \
	--subgroup_idxs 2 2 2 1 1 \
	--batch_size 64 \
	--epochs 0 \
	--val_interval 0 \
	--output_dir temp \
	--load_reps data/reps.pkl \
	--save_patterns patterns/type2.pkl \
	--device cpu \
	> /dev/null

rm -r temp


# unraveled
echo Generating unraveled patterns . . .
python ../../train.py \
	--unravel_reps \
	--seed 0 \
	--channels 1 1 1 2 1 \
	--subgroup_idxs 2 2 2 1 1 \
	--batch_size 64 \
	--epochs 0 \
	--val_interval 0 \
	--output_dir temp \
	--load_reps data/reps.pkl \
	--save_patterns patterns/unraveled.pkl \
	--device cpu \
	> /dev/null

rm -r temp


echo All done!
