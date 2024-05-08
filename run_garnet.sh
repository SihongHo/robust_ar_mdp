#!/bin/bash
# 定义算法列表
algorithms=("robust-our" "robust-base" "non-robust")

# 外层循环：算法
for alg in "${algorithms[@]}"
do
    echo "Starting runs for algorithm: $alg"
    
    # 内层循环：运行次数
    for i in {1..10}
    do
        # 创建一个文件名包含算法名、运行编号和时间戳的日志文件
        log_file="logs/${alg}_run_${i}_$(date +%Y%m%d_%H%M%S).log"
        echo "Run $i for $alg starting..." | tee -a $log_file
        
        # 将Python命令的输出和错误都重定向到日志文件
        python main.py --env garnet_instance --alg $alg --save_path "results/${alg}/garnet_instance_3_2/" --garnet_instance_path "envs/garnet_instance_3_2.pkl" --warnings_stop --training_steps 100 --S_n 3 --A_n 2 --uncertainty 0.1 --alpha 0.1 --max_iterations 2000 >> $log_file 2>&1
        
        echo "Run $i for $alg completed." | tee -a $log_file
        sleep 10  # Waits for 10 seconds before the next run
    done
    
    echo "All runs completed for $alg."
done

echo "All tasks completed."
