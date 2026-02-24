# Public retrain hyper-parameters for MIL GMA training
export NUM_EPOCHS=120
export LEARNING_RATE=1e-4
export WEIGHT_DECAY=1e-4
export BATCH_SIZE_TRAIN=1
export NUM_WORKERS=4
export CV_FOLDS=5
export AGGREGATION_METHOD=GMA
# Validate only at specific epochs via EXTRA_VAL_EPOCHS; set VAL_INTERVAL high to avoid extras
export VAL_INTERVAL=999
export EXTRA_VAL_EPOCHS=2,5,10,20,50,80,120
export PATIENCE=999
