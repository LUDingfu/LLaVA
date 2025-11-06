import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model, Dinov2Config, AutoImageProcessor
from open_clip import create_model_from_pretrained

from ezcolorlog import root_logger as logger


class ProcessorWrapper:
    """SigLIP processor wrapper"""
    def __init__(self, transform, height=224, width=224, image_mean=[0.48145466, 0.4578275, 0.40821073]):
        self._crop_size = {
            "height": height,
            "width": width,
        }
        self._transforms = transform
        self.image_mean = image_mean

    @property
    def crop_size(self):
        return self._crop_size

    def preprocess(self, image, return_tensors='pt'):
        output = {}
        output['pixel_values'] = [self._transforms(image)]
        return output


def extract_siglip_model_info(vision_tower_name):
    """Extract SigLIP model information"""
    valid_model_prefixes = {
        "siglip/CLIP-ViT-SO400M-14-384": "hf-hub:timm/ViT-SO400M-14-SigLIP-384",
        "timm/ViT-SO400M-14-SigLIP-384": "hf-hub:timm/ViT-SO400M-14-SigLIP-384",
        "siglip/CLIP-ViT-SO400M-14": "hf-hub:timm/ViT-SO400M-14-SigLIP",
        "timm/ViT-SO400M-14-SigLIP": "hf-hub:timm/ViT-SO400M-14-SigLIP"
    }

    res = 384 if '384' in vision_tower_name else 224
    interp = None

    for prefix in valid_model_prefixes:
        if vision_tower_name.startswith(prefix):
            base_model_name = valid_model_prefixes[prefix]
            break
    else:
        raise ValueError(f"Unknown SigLIP vision tower: {vision_tower_name}")

    parts = vision_tower_name.split("-")
    for part in parts:
        if part.startswith("res"):
            res = int(part[3:])
        elif part.startswith("interp"):
            interp = int(part[6:])

    return base_model_name, res, interp


class MRFVisionTower(nn.Module):
    """
    Multi-Resolution Fusion (MRF) Vision Tower
    融合三个尺度的DINOv2特征和一个尺度的SigLIP特征
    - DINOv2: 112 (0.5x), 224 (1x), 518 (1.5x) -> [B, N, 768*3]
    - SigLIP: 224 (1x) -> [B, N, 1152]
    统一空间分辨率到最大网格(518//14=37)后，在特征维度拼接 -> [B, 37*37, 768*3+1152]
    """
    
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.args = args
        self.unfreeze_mm_vision_tower = getattr(args, 'unfreeze_mm_vision_tower', False)
        
        # 解析模型名称: mrf-facebook/dinov2-base-&&&-siglip/CLIP-ViT-SO400M-14-384
        model_names = self.vision_tower_name.replace("mrf-", "")
        self.model_names = model_names.split("-&&&-")
        logger.warning(f"Creating MRF Vision Tower with models: {self.model_names}")
        
        # DINOv2多尺度配置
        self.dinov2_patch_size = 14
        self.dinov2_target_sizes = [112, 224, 518]  # MRF尺度: 0.5x, 1x, 1.5x (近似)
        self.select_layer = getattr(args, 'mm_vision_select_layer', -2)
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        
        # SigLIP配置
        self.siglip_target_size = 224  # 1x resolution
        
        if not delay_load or self.unfreeze_mm_vision_tower:
            self.load_model()
        else:
            # 延迟加载时只加载配置
            self.dinov2_cfg_only = Dinov2Config.from_pretrained(self.model_names[0])
            self._siglip_hidden_size = 1152
    
    def load_model(self, device_map=None):
        if self.is_loaded:
            logger.warning(f'{self.vision_tower_name} is already loaded, skipping.')
            return
        
        # 加载DINOv2模型
        dinov2_name = self.model_names[0]
        img_size = max(self.dinov2_target_sizes)
        self.dinov2_image_processor = AutoImageProcessor.from_pretrained(
            dinov2_name,
            size={"height": img_size, "width": img_size},
            crop_size={"height": img_size, "width": img_size}
        )
        self.dinov2_tower = Dinov2Model.from_pretrained(dinov2_name, device_map=device_map)
        self.dinov2_tower.requires_grad_(False)
        
        # 加载SigLIP模型
        siglip_name = self.model_names[1]
        base_siglip_name, res, interp = extract_siglip_model_info(siglip_name)
        clip_model, processor = create_model_from_pretrained(base_siglip_name)
        
        self.siglip_tower = clip_model.visual.trunk
        self.siglip_tower.output_tokens = True
        self.siglip_hidden_size = self.siglip_tower.embed_dim
        self.siglip_patch_size = self.siglip_tower.patch_embed.patch_size[0]
        self.siglip_image_processor = ProcessorWrapper(
            processor,
            height=self.siglip_target_size,
            width=self.siglip_target_size
        )
        self.siglip_tower.requires_grad_(False)
        
        # 计算总hidden_size
        dinov2_hidden_size = self.dinov2_tower.config.hidden_size
        self._hidden_size = dinov2_hidden_size * 3 + self.siglip_hidden_size  # 768*3 + 1152 = 3456
        
        # 最大网格大小 (518//14 = 37)
        self.max_grid_size = max(self.dinov2_target_sizes) // self.dinov2_patch_size
        
        self.is_loaded = True
        logger.info(f"MRF Vision Tower loaded: hidden_size={self._hidden_size}, max_grid_size={self.max_grid_size}")
    
    def dinov2_feature_select(self, image_forward_outs):
        """选择DINOv2的特征"""
        x = image_forward_outs.hidden_states[self.select_layer]  # (B, 1+N, C)
        if self.select_feature == 'patch':
            x = x[:, 1:]  # 去掉 CLS
        elif self.select_feature == 'cls_patch':
            pass
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return x  # (B, N, C)
    
    def _run_dinov2_one_scale(self, images_chw, side: int):
        """
        运行DINOv2的一个尺度
        images_chw: (B, 3, H, W) tensor in [0,1] or [0,255]
        side: 112 / 224 / 518
        returns: (B, N=grid^2, C) patch tokens at this scale
        """
        # Resize到目标尺寸
        pixel_values = F.interpolate(images_chw, size=(side, side), mode='bilinear', align_corners=False)
        
        # 通过DINOv2，取指定层的patch token
        out = self.dinov2_tower(pixel_values, output_hidden_states=True)
        feats = self.dinov2_feature_select(out).to(images_chw.dtype)  # (B, N, C)
        
        return feats  # (B, (side/14)^2, C)
    
    def _run_siglip_scale(self, images_chw):
        """
        运行SigLIP的一个尺度 (224)
        images_chw: (B, 3, H, W) tensor in [0,1] or [0,255]
        returns: (B, N=grid^2, C) patch tokens
        """
        # Resize到224
        pixel_values = F.interpolate(images_chw, size=(self.siglip_target_size, self.siglip_target_size), 
                                    mode='bilinear', align_corners=False)
        
        # 通过SigLIP (梯度控制由forward方法统一管理)
        image_features = self.siglip_tower.forward_features(
            pixel_values.to(device=self.device, dtype=self.dtype)
        )  # (B, N, C)
        
        return image_features
    
    def _tokens_to_grid(self, tokens, side, patch_size):
        """
        tokens: (B, N, C), side: 图片边长, patch_size: patch大小
        return: (B, C, H, W) with H=W=side//patch_size
        """
        B, N, C = tokens.shape
        g = side // patch_size
        assert g * g == N, f"N={N}不等于({g}^2)；side={side}, patch={patch_size}"
        grid = tokens.view(B, g, g, C).permute(0, 3, 1, 2).contiguous()  # (B, C, g, g)
        return grid
    
    def _grid_to_tokens(self, grid):
        """
        grid: (B, C, H, W)  -> (B, H*W, C)
        """
        B, C, H, W = grid.shape
        tokens = grid.permute(0, 2, 3, 1).contiguous().view(B, H*W, C)
        return tokens
    
    def forward(self, images):
        """
        前向传播融合过程:
        1. 提取三个DINOv2尺度的特征: 112, 224, 518
        2. 提取一个SigLIP尺度的特征: 224
        3. 统一空间分辨率到最大网格(37x37)
        4. 在特征维度拼接
        
        images: Tensor (B,3,H,W)，数值范围 [0,1] / [0,255]
        返回: (B, 37*37, 3456)
        """
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            # 1. 提取三个DINOv2尺度的特征
            dinov2_grids = []
            for s in self.dinov2_target_sizes:
                tokens = self._run_dinov2_one_scale(images, s)  # (B, (s/14)^2, 768)
                grid = self._tokens_to_grid(tokens, s, self.dinov2_patch_size)  # (B, 768, s/14, s/14)
                dinov2_grids.append(grid)
            
            # 2. 提取SigLIP尺度的特征
            siglip_tokens = self._run_siglip_scale(images)  # (B, (224/patch_size)^2, 1152)
            siglip_grid_size = self.siglip_target_size // self.siglip_patch_size
            siglip_grid = self._tokens_to_grid(siglip_tokens, self.siglip_target_size, self.siglip_patch_size)  # (B, 1152, grid_size, grid_size)
            
            # 3. 统一空间分辨率到最大网格 (37x37)
            up_grids = []
            
            # DINOv2的三个尺度
            for g, side in zip(dinov2_grids, self.dinov2_target_sizes):
                cur_g = side // self.dinov2_patch_size
                if cur_g != self.max_grid_size:
                    g_up = F.interpolate(g, size=(self.max_grid_size, self.max_grid_size), 
                                        mode='bilinear', align_corners=False)  # (B, 768, 37, 37)
                else:
                    g_up = g
                up_grids.append(g_up)
            
            # SigLIP尺度
            if siglip_grid_size != self.max_grid_size:
                siglip_up = F.interpolate(siglip_grid, size=(self.max_grid_size, self.max_grid_size),
                                        mode='bilinear', align_corners=False)  # (B, 1152, 37, 37)
            else:
                siglip_up = siglip_grid
            up_grids.append(siglip_up)
            
            # 4. 在通道维拼接 -> (B, 3456, 37, 37) -> 再摊平成 (B, 37*37, 3456)
            up_cat = torch.cat(up_grids, dim=1)  # (B, 3456, 37, 37)
            out = self._grid_to_tokens(up_cat)  # (B, 1369, 3456)
            
            return out
    
    @property
    def dummy_feature(self):
        """返回dummy特征用于初始化"""
        return torch.zeros(1, self.num_patches, self._hidden_size, 
                          device=self.device, dtype=self.dtype)
    
    @property
    def dtype(self):
        return self.dinov2_tower.dtype if self.is_loaded else torch.float32
    
    @property
    def device(self):
        return self.dinov2_tower.device if self.is_loaded else torch.device("cpu")
    
    @property
    def config(self):
        if self.is_loaded:
            return self.dinov2_tower.config
        else:
            return self.dinov2_cfg_only
    
    @property
    def hidden_size(self):
        return self._hidden_size
    
    @property
    def num_patches_per_side(self):
        """以最大尺度518为准"""
        return self.max_grid_size  # 37
    
    @property
    def num_patches(self):
        """总patch数量"""
        g = self.num_patches_per_side
        return g * g  # 1369
    
    @property
    def image_processor(self):
        """返回DINOv2的image processor用于兼容性"""
        return self.dinov2_image_processor if self.is_loaded else None

