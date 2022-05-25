CUDA_VISIBLE_DEVICES=0 python3 src/reseval.py \
                      --algorithm pieg \
                      --eval_episodes 100 \
                      --seed $1 \
                      --eval_mode $2 \
                      --action_repeat 2 \
                      --domain_name $3 \
                      --task_name $4