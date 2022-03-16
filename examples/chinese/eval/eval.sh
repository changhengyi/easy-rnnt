#!/usr/bin/env bash
ps aux | grep eval.py | awk '{print $2}' | xargs kill -9

starttime=$(date +%Y-%m-%d\ %H:%M:%S)
resume=/home/changhengyi/rnnt-decode_test/model.epoch-27
log_path=/home/changhengyi/rnnt-decode_test/$(date +%Y-%m-%d-%H-%M-%S)
beam_size=5
mkdir -p ${log_path}
echo ============================================================================ >> ${log_path}/final.res
echo "Time: $starttime" >> ${log_path}/final.res
echo "Model: ${resume}" >> ${log_path}/final.res
echo "Beam size: ${beam_size}" >> ${log_path}/final.res
echo ---------------------------------------------------------------------------- >> ${log_path}/final.res
echo "Use beamserch find min wer in word aspect" >> ${log_path}/final.res
echo ---------------------------------------------------------------------------- >> ${log_path}/final.res

CUDA_VISIBLE_DEVICES=4 python /home/changhengyi/easy-rnnt/rnnt/bin/eval.py \
            config=/home/changhengyi/easy-rnnt/examples/chinese/eval/eval.yaml \
            testset_path=/home/changhengyi/rnnt-decode_test/speechio_1.tiny.tsv \
            resume=${resume} \
            beam_size=${beam_size} \
            log_path=${log_path}

# gpu_dict=(["1"]="1" ["2"]="2" ["3"]="6" ["4"]="7")
# for x in 1 2 3 4; do
#     echo "----------Test_set ${x} use gpu ${gpu_dict[x]}"
#     CUDA_VISIBLE_DEVICES=${gpu_dict[x]} python /home/changhengyi/easy-rnnt/rnnt/bin/eval.py \
#             config=/home/changhengyi/easy-rnnt/examples/chinese/eval/eval.yaml \
#             testset_path=/home/changhengyi/rnnt-decode_test/speechio_${x}.tsv \
#             resume=${resume} \
#             beam_size=${beam_size} \
#             log_path=${log_path} &
# done
