#!/bin/bash

GPUS=(0 1 2 3 4 5 6 7)
GPU_IDX=0

SEED=42
EPOCHS=200
ARCH="resnet18"
BATCH=128

for OPT in adam sgd; do
    if [ "$OPT" == "adam" ]; then
        LRS=(0.1 0.01 0.001)
        MOMENTUMS=(0.0)
    else
        LRS=(0.1 0.05 0.01)
        MOMENTUMS=(0.9 0.99)
    fi

    for LR in "${LRS[@]}"; do
        for WD in 0.0 0.0005 0.001; do
            for MOM in "${MOMENTUMS[@]}"; do
                for L1 in 0.0 1e-5; do
                    for RAUG in "" "--rand_aug"; do
                        for SAM in "none" "0.05" "0.1"; do
                            for SCHED in "" "--lr_schedule"; do

                                CMD="CUDA_VISIBLE_DEVICES=${GPUS[$GPU_IDX]} python3 cifar10_learn_configurable.py \
                                    --arch $ARCH --opt $OPT --lr $LR --wd $WD --seed $SEED \
                                    --batch_size $BATCH --epochs $EPOCHS \
                                    --l1_lambda $L1 --l2_lambda 0.0 $RAUG $SCHED"

                                if [ "$OPT" != "adam" ]; then
                                    CMD+=" --momentum $MOM"
                                else
                                    CMD+=" --momentum 0.0"
                                fi

                                if [ "$SAM" != "none" ]; then
                                    CMD+=" --sam --sam_rho $SAM"
                                fi

                                echo "Launching: $CMD"
                                eval "$CMD &"

                                GPU_IDX=$(( (GPU_IDX + 1) % ${#GPUS[@]} ))

                                while [ $(jobs -r | wc -l) -ge ${#GPUS[@]} ]; do
                                    sleep 10
                                done

                            done
                        done
                    done
                done
            done
        done
    done
done

wait
echo "All models trained."
