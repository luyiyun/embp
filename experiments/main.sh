set -e # 一旦出现错误，立即停止运行，并打印出错误信息。

nrepeat=1000
ncore=20
# runtool="uv run"
runtool="python"

# ================ scenario3003: binary, without Z ================
# seed=0
# ncore=20
# nrepeat=100
# num_samples=(100 200)
# prevalences=(0.1)
# ORs=(1.25 1.5 1.75 2 2.25 2.5 2.75 3)
# for n in ${num_samples[@]}; do
#     for pr in ${prevalences[@]};do
#         for OR in ${ORs[@]}; do
#             seed=$((seed+1))
#             echo "<==========3003> n=$n, pr=$pr, OR=$OR, sigma2_x=10, rx=0.2, seed=$seed"
#             data_dir=./scenario3003/data/binary_wo_z_${n}_${pr}_${OR}
#             ana_dir=./scenario3003/lap/binary_wo_z_${n}_${pr}_${OR}
#             eval_fn=eval_results.csv
#             $runtool main.py simulate -ot binary -od $data_dir --seed $seed --n_samples $n --ratio_observed_x 0.2 --OR $OR -nr $nrepeat -pr $pr --sigma2_x 10 --sigma2_e 10
#             $runtool main.py analyze -ot binary -dd $data_dir -od $ana_dir -nc $ncore --binary_solve lap
#             $runtool main.py evaluate -ad $ana_dir -of $eval_fn
#         done
#     done
# done

# $runtool main.py summarize -efp "./scenario3003/lap/binary_wo_z_*/eval_results.csv" -of ./scenario3003/summary_lap.xlsx \
#     -sp  prevalence n_knowX_per_studies OR
