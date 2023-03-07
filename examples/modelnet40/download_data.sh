#!/bin/bash
DATADIR=data/raw

# generate data
if [ ! -d $DATADIR ]; then
	echo "[!] Data files do not exist. Preparing data..."
	# make data directories
	mkdir -p $DATADIR
	mkdir -p $DATADIR/modelnet40_test
	mkdir -p $DATADIR/modelnet40_train
	# download modelnet data
	wget -P $DATADIR/ https://lmb.informatik.uni-freiburg.de/resources/datasets/ORION/modelnet40_manually_aligned.tar

	# setup modelnet40
	mkdir -p $DATADIR/modelnet40
	tar -xf $DATADIR/modelnet40_manually_aligned.tar -C $DATADIR/modelnet40
	find $DATADIR/modelnet40 -mindepth 3 | grep off | grep train | xargs mv -t $DATADIR/modelnet40_train
	find $DATADIR/modelnet40 -mindepth 3 | grep off | grep test | xargs mv -t $DATADIR/modelnet40_test
	rm -rf $DATADIR/modelnet40
	rm $DATADIR/modelnet40_manually_aligned.tar
	find $DATADIR/modelnet40_train -mindepth 1 | grep annot | xargs rm
	find $DATADIR/modelnet40_test -mindepth 1 | grep annot | xargs rm
fi

echo Done!
