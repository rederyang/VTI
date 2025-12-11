export DATA_DIR=/workspace/data/COCO2014
export PROJECT_DIR=/workspace/VTI

RUN_DIR=$PROJECT_DIR/runs/vti_fix/d50t50_i0.9

source $PROJECT_DIR/scripts/load_api.sh

# if acceleration needed, use hf-mirror
# export HF_ENDPOINT=https://hf-mirror.com

mkdir -p $RUN_DIR
touch $RUN_DIR/run_log.txt
touch $RUN_DIR/eval_log.txt

python -u $PROJECT_DIR/experiments/eval/run_mmhal_vti.py \
    --alpha_image 0.9 \
    --alpha_text 0.0 \
    --use_fix_vti \
    --head_wise_alpha_image 0.0 \
    --seed 42 \
    --image-folder $DATA_DIR/val2014/ \
    --data-file $DATA_DIR/ \
    --answers-file $RUN_DIR/MMHal_answer.jsonl \
    --num_demos 50 \
    --mask_ratio 0.99 \
    --num_trials 50 \
    --visual_direction_path $RUN_DIR/visual_direction.pt \
    --textual_direction_path $RUN_DIR/textual_direction.pt \
    2>&1 | tee $RUN_DIR/run_log.txt

    # --competitor smoothing \
    # --competitor_position encoder \
    # --smooth_kernel 3 \
    # --grid_size 24 \

    # --competitor clipping \
    # --competitor_position encoder \
    # --clip_percentile 95 \
    # --clip_mode global \

python -u $PROJECT_DIR/experiments/eval/eval_mmhal.py \
    --response $RUN_DIR/MMHal_answer.jsonl \
    --evaluation $RUN_DIR/MMHal_answer_evaluation.json \
    --api-base $OPENAI_API_BASE \
    --api-key $OPENAI_API_KEY \
    --gpt-model $OPENAI_API_MODEL \
    2>&1 | tee $RUN_DIR/eval_log.txt
