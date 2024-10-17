import os
from .clip_encoder import CLIPVisionTower
from .qformer_encoder import QFormerWrapper
from .retriever import Retriever
from llava.model import *


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("google") or vision_tower.startswith("facebook") or "ShareGPT4V" in vision_tower:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')


def build_qformer(qformer_cfg):
    return QFormerWrapper(qformer_cfg)


def build_retriever(cfg=None):
    return Retriever(input_dim=4096, num_heads=8, ff_dim=2048, num_layers=12, dropout_rate=0.1)