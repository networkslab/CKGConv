from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_mlflow')
def set_cfg_mlflow(cfg):
    """
    MLflow tracker configuration.
    """

    # MLflow group
    cfg.mlflow = CN()

    # Use MLflow or not
    cfg.mlflow.use = False

    # # Wandb entity name, should exist beforehand
    # cfg.mlflow.entity = "gtransformers"

    # MLflow project name, will be created in your team if doesn't exist already
    # cfg.mlflow.project = "gtblueprint"

    # Optional run name
    cfg.mlflow.project = " "
    cfg.mlflow.name = " "
