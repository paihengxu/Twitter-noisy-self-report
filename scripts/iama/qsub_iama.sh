num_jobs=30
for i in $(seq -f '%02g' 0 29); do
    qsub -N iama_${i} -cwd -o ./log -e ./log ./run_iama.sh collect ${i} ${num_jobs}
done
qsub -hold_jid "iama_*" -N iama_agg -cwd -o ./log -e ./log ./run_iama.sh aggregate ${i} ${num_jobs} 
