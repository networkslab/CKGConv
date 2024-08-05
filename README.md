# Read Me  
> This is the official implementation for CKGConv.
> 
> Ma, L., Pal, S., Zhang, Y., Zhou, J., Zhang, Y., & Coates, M. CKGConv: General Graph Convolution with Continuous Kernels. In *Forty-first International Conference on Machine Learning* (ICML2024). [[ICML]](https://proceedings.mlr.press/v235/ma24k.html)  [[arXiv]](https://arxiv.org/abs/2404.13604)

> The code base is built upon the code of [GRIT](https://github.com/LiamMa/GRIT) and [GraphGPS](https://github.com/rampasek/GraphGPS).
> 

### Python environment setup with Conda
```bash
conda create -n ckgconv python=3.9
conda activate ckgconv

# please change the cuda/device version as you need; 


pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118 --trusted-host download.pytorch.org 
# we use the pytorch version 2.1.2 in the experiments
# --- up to torch_geometric==2.5.3; torch_geometric will automatically adjust the version for torch==2.1.2
pip install torch_geometric --trusted-host data.pyg.org 
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html --trusted-host data.pyg.org


# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
conda install openbabel fsspec rdkit -c conda-forge
# alternative if conda doesn't work: 
## pip install rdkit

pip install torchmetrics==0.9.1 
## later version of torchmetrics might lead to incompatibility 


## ----- the versions of the following package typically will not lead to conflicts -----
pip install ogb
pip install tensorboardX
pip install yacs
pip install opt_einsum
pip install graphgym 
pip install pytorch-lightning # required by graphgym 
## --------------------------------------------------


pip install setuptools==59.5.0
# distuitls has conflicts with pytorch with latest version of setuptools

pip install timm
pip install einops

# pip install wandb # the support of wandb is implemented in the pipeline; but we did not use it in CKGConv; please verify the usability before using.
## we use mlflow as an alternative
pip install mlflow 
# To use MLFLOW, excute:  mlflow server --backend-store-uri mlruns --port 5000

# conda clean --all
```


### Running CKGConv 
```bash
# Run

# > python main.py --cfg configs/{{str:the_config_file}}.yaml mlflow.use {{bool:use_mlflow}} accelerator {{str:'cpu'|'cuda:0'}} seed {{int:seed}} dataset.dir {{data_dir}} 

# example:
python main.py --cfg configs/CKGConvMLP/zinc-CKGConvMLP-BN-BS32.yaml mlflow.use True accelerator "cuda:0" seed 0


# replace 'cuda:0' with the device to use
# data will be downloaded to "./datasets" without specifying dataset.dir 
```

