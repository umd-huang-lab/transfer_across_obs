#!/bin/bash
ENVNAME="cart"
RUNS=20
for (( i=0; i<${RUNS}; i++ ))
do
    #### IN SOURCE TASK
    # source task learning with auxiliary task
    python source_dqn.py --env-name ${ENVNAME} --name source_l2_run${i} --episodes 200 \
        --head-layers 1 --feature-size 16 -detach-next

    #### IN TARGET TASK
    # single task learning without auxiliary task
    python pixel_dqn.py --env-name ${ENVNAME}_pixel --name single_run${i} --episodes 200 \
        -no-reg -detach-next --head-layers 1 --feature-size 16
    # single task learning with auxiliary task
    python pixel_dqn.py --env-name ${ENVNAME}_pixel --name single_auxiliary_run${i} --episodes 200 \
        -detach-next --head-layers 1 --feature-size 16
    # transfer from source task
    python pixel_dqn.py --env-name ${ENVNAME}_pixel --name transfer_run${i} --episodes 200 \
        -transfer -detach-next --head-layers 1 --feature-size 16 \
        --load-from learned_models/${ENVNAME}/source_l2_run${i}.pt
    # transfer with one layer head
    python pixel_dqn.py --env-name ${ENVNAME}_pixel --name transfer_linear_run${i} --episodes 200 \
        -transfer -detach-next --head-layers 0 --feature-size 16 \
        --load-from learned_models/${ENVNAME}/source_l2_run${i}.pt
    # learn with randomly initialized model
    python pixel_dqn.py --env-name ${ENVNAME}_pixel --name transfer_random_run${i} --episodes 200 \
        -transfer -detach-next --head-layers 1 --feature-size 16 
done