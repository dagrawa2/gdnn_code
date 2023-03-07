#!/usr/bin/bash
set -e

# type 1
echo Generating type 1 patterns . . .
printf "0\n0\n0\n0" | python ../../train.py \
	--seed 0 \
	--percentage 25 \
	--channels 1 1 1 40 \
	--subgroup_idxs 30 15 10 1 \
	--batch_size 64 \
	--epochs 0 \
	--val_interval 0 \
	--output_dir temp \
	--save_patterns patterns/type1.pkl \
	--device cpu

rm -r temp


# mixed
echo Generating mixed patterns . . .
printf "0, 1\n0, 1\n0, 1\n0" | python ../../train.py \
	--seed 0 \
	--percentage 25 \
	--channels 1 1 1 40 \
	--subgroup_idxs 30 15 10 1 \
	--batch_size 64 \
	--epochs 0 \
	--val_interval 0 \
	--output_dir temp \
	--save_patterns patterns/mixed.pkl \
	--device cpu

rm -r temp


# unraveled
echo Generating unraveled patterns . . .
printf "0, 1\n0, 1\n0, 1\n0" | python ../../train.py \
	--unravel_reps \
	--seed 0 \
	--percentage 25 \
	--channels 1 1 1 40 \
	--subgroup_idxs 30 15 10 1 \
	--batch_size 64 \
	--epochs 0 \
	--val_interval 0 \
	--output_dir temp \
	--save_patterns patterns/unraveled.pkl \
	--device cpu

rm -r temp


echo All done!
