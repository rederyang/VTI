export DATA_DIR=/path/to/your/data
export PROJECT_DIR=/path/to/your/project
export OPENAI_API_KEY=your_openai_api_key
# if acceleration needed, use hf-mirror
# export HF_ENDPOINT=https://hf-mirror.com

mkdir -p $PROJECT_DIR/results

python -u $PROJECT_DIR/experiments/eval/run_mmhal_vti.py \
    --alpha_image 0.9 \
    --alpha_text 0.9 \
    --seed 42 \
    --image-folder $DATA_DIR/val2014/ \
    --data-file $DATA_DIR/ \
    --answers-file $PROJECT_DIR/results/MMHal_answer.jsonl \
    --num_demos 5 \
    --mask_ratio 0.99 \
    --num_trials 2 \
    2>&1 | tee $PROJECT_DIR/results/run_log.txt

python -u $PROJECT_DIR/experiments/eval/eval_mmhal.py \
    --response $PROJECT_DIR/results/MMHal_answer.jsonl \
    --evaluation $PROJECT_DIR/results/MMHal_answer_evaluation.json \
    --api-key $OPENAI_API_KEY \
    2>&1 | tee $PROJECT_DIR/results/eval_log.txt
