RUN_DIR=$PROJECT_DIR/runs/d5t2_rerun_2

export DATA_DIR=/root/autodl-tmp/MSCOCO
export PROJECT_DIR=/root/VTI

source $PROJECT_DIR/scripts/load_api.sh

# if acceleration needed, use hf-mirror
# export HF_ENDPOINT=https://hf-mirror.com

mkdir -p $RUN_DIR
touch $RUN_DIR/run_log.txt
touch $RUN_DIR/eval_log.txt

python -u $PROJECT_DIR/experiments/eval/run_mmhal_vti.py \
    --alpha_image 0.9 \
    --alpha_text 0.9 \
    --seed 42 \
    --image-folder $DATA_DIR/val2014/ \
    --data-file $DATA_DIR/ \
    --answers-file $RUN_DIR/MMHal_answer.jsonl \
    --num_demos 5 \
    --mask_ratio 0.99 \
    --num_trials 2 \
    2>&1 | tee $RUN_DIR/run_log.txt

python -u $PROJECT_DIR/experiments/eval/eval_mmhal.py \
    --response $RUN_DIR/MMHal_answer.jsonl \
    --evaluation $RUN_DIR/MMHal_answer_evaluation.json \
    --api-base $OPENAI_API_BASE \
    --api-key $OPENAI_API_KEY \
    --gpt-model $OPENAI_API_MODEL \
    2>&1 | tee $RUN_DIR/eval_log.txt
