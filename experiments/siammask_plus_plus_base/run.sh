ROOT=`git rev-parse --show-toplevel`
export PYTHONPATH=$ROOT:$PYTHONPATH


mkdir -p logs

python -u $ROOT/tools/train_siammask_pp.py \
    --config=config.json -b 16 \
    -j 4 \
    --epochs 20 \
    --log logs/log.txt \
    2>&1 | tee logs/train.log

