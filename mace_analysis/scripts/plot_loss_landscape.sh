#!/bin/bash

# Set default values
SEED_LIST=({4..6})
SHUFFLE_SEED_LIST=({1..1})

for SEED in "${SEED_LIST[@]}"; do
    for SHUFFLE_SEED in "${SHUFFLE_SEED_LIST[@]}"; do
        MODEL_PATH="/scratch/st-ortner-1/jerry528/ETN/train_benchmark_results/results_MACEwater_firstlayer_new/water-non_symmetric_cp-L0-seed-$SEED-c64/MACEwater_baseline_run-${SEED}_lastepoch.model"
        #MODEL_PATH="/scratch/st-ortner-1/jerry528/ETN/train_benchmark_results/results_MACEwater_firstlayer_new/water-symmetric_cp-L0-seed-${SEED}-c64/MACEwater_baseline_run-${SEED}_lastepoch.model"
        CONFIG_PATH="/scratch/st-ortner-1/jerry528/ETN/data/dataset_water/water_test.xyz"
        OUTPUT_DIR="/scratch/st-ortner-1/jerry528/ETN/PyHessian/mace_analysis/results/water/nonsym_cp_${SEED}"
        DEVICE="cuda"
        BATCH_SIZE=10
        NDATA=-1

        # Run the Python script
        python /scratch/st-ortner-1/jerry528/ETN/PyHessian/mace_analysis/mace_pyhessian_loss_landscape.py --model_path "$MODEL_PATH" --config_path "$CONFIG_PATH" --output_dir "$OUTPUT_DIR" --device "$DEVICE" --batch_size "$BATCH_SIZE" --shuffle_seed "$SHUFFLE_SEED" --Ndata "$NDATA"
    done
done