
taskset -c 0-72 python main.py --cfg configs/PE_Ablation/zinc-CKGConvMLP-BN-BS32-RWSE.yaml mlflow.use True accelerator "cuda:4" seed 0 &
taskset -c 0-72 python main.py --cfg configs/PE_Ablation/zinc-CKGConvMLP-BN-BS32-RWSE.yaml mlflow.use True accelerator "cuda:5" seed 1 &
taskset -c 0-72 python main.py --cfg configs/PE_Ablation/zinc-CKGConvMLP-BN-BS32-RWSE.yaml mlflow.use True accelerator "cuda:6" seed 2 &
taskset -c 0-72 python main.py --cfg configs/PE_Ablation/zinc-CKGConvMLP-BN-BS32-RWSE.yaml mlflow.use True accelerator "cuda:7" seed 3 &
wait



#
#taskset -c 0-72 python main.py --cfg configs/PE/zinc-CKGConvMLP-BN-BS32-RWSE.yaml mlflow.use False accelerator "cuda:4" seed 0







