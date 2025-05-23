#!bin/bash

function get_jobdep() {
if [[ "$1" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
    exit 0
else
    echo "submission failed"
    exit 1
fi
}

#Run first batch
sb="$(sbatch setup.qs 1)"
dep="$(get_jobdep "$sb")"
sb="$(sbatch -D batch_1 --dependency=afterok:$dep --kill-on-invalid-dep=yes mdrun.qs)"
dep="$(get_jobdep "$sb")"
sb="$(sbatch --dependency=afterany:$dep --kill-on-invalid-dep=yes post.qs 1)"
dep="$(get_jobdep "$sb")"

#run other batches
for i in {2..5}
do
    sb="$(sbatch --dependency=afterok:$dep --kill-on-invalid-dep=yes setup.qs $i)"
    dep="$(get_jobdep "$sb")"
    sb="$(sbatch -D "batch_$i" --dependency=afterok:$dep --kill-on-invalid-dep=yes mdrun.qs)"
    dep="$(get_jobdep "$sb")"
    sb="$(sbatch --dependency=afterany:$dep --kill-on-invalid-dep=yes post.qs $i)"
    dep="$(get_jobdep "$sb")"
done
