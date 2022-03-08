#!/usr/bin/env bash
starttime=$(date +%Y-%m-%d\ %H:%M:%S)
resume=/home/changhengyi/rnnt-decode_test/model.epoch-23
log_path="speechIO.wer"
beam_size=10
echo ============================================================================ >> ${log_path}
echo "Time: $starttime" >> ${log_path}
echo "Model: ${resume}" >> ${log_path}
echo "Beam size: ${beam_size}" >> ${log_path}
echo ---------------------------------------------------------------------------- >> ${log_path}
echo "Use beamserch find min wer in pinyin aspect" >> ${log_path}
echo ---------------------------------------------------------------------------- >> ${log_path}


for x in "1" "2" "3" "4"; do
    CUDA_VISIBLE_DEVICES=${x} python /home/changhengyi/easy-rnnt/rnnt/bin/eval.py \
            config=/home/changhengyi/easy-rnnt/examples/chinese/eval.yaml \
            testset_path=/home/changhengyi/rnnt-decode_test/speechio_${x}.tsv \
            resume=${resume} \
            beam_size=${beam_size} \
            log_path=${log_path} &
done