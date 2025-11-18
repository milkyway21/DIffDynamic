# 总结：
# - 根据传入参数在多节点环境中均匀分配采样任务，每个任务调用 `scripts.sample_diffusion` 进行模型采样。
# - 支持指定配置文件、输出目录、总节点数、当前节点编号以及起始任务索引。

TOTAL_TASKS=100  # 定义任务总数，即模型采样将生成的样本数量。
BATCH_SIZE=50    # 每个任务调用 `sample_diffusion` 时使用的采样批大小。

if [ $# != 5 ]; then
    echo "Error: 5 arguments required."  # 参数不足时给出提示并退出。
    exit 1
fi

CONFIG_FILE=$1
RESULT_PATH=$2
NODE_ALL=$3
NODE_THIS=$4
START_IDX=$5

for ((i=$START_IDX;i<$TOTAL_TASKS;i++)); do
    NODE_TARGET=$(($i % $NODE_ALL))  # 根据总节点数计算任务所属节点。
    if [ $NODE_TARGET == $NODE_THIS ]; then
        echo "Task ${i} assigned to this worker (${NODE_THIS})"
        python -m scripts.sample_diffusion ${CONFIG_FILE} -i ${i} --batch_size ${BATCH_SIZE} --result_path ${RESULT_PATH}
    fi
done
