import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
# from .dinov2_encoder import Dinov2VisionTower, Dinov2VisionTowerS2
from .dinov2_encoder_mrf import Dinov2VisionTower
from .hybrid_encoder import HybridVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)

    # 支持hybrid模型: hybridmodel-facebook/dinov2-base-&&&-siglip/CLIP-ViT-SO400M-14-384
    if vision_tower.startswith("hybridmodel-"):
        # raise ValueError(f'Unknown vision tower: {vision_tower}')

        return HybridVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    if vision_tower.startswith("facebook/dinov2"):
        # if use_s2:
        #     return Dinov2VisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        # else:
        #     return Dinov2VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        return Dinov2VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    # 支持siglip和CLIP模型
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or vision_tower.startswith("siglip") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')

