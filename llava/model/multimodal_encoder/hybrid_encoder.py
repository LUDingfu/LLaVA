import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ezcolorlog import root_logger as logger


class HybridVisionTower(nn.Module):
    """
    Hybrid Vision Tower融合DINOv2和SigLIP两个编码器
    - DINOv2: facebook/dinov2-base -> [B, N1, 768]
    - SigLIP: siglip/CLIP-ViT-SO400M-14-384 -> [B, N2, 1152]
    统一空间分辨率到576 tokens (24×24)后，在特征维度拼接 -> [B, 576, 1920]
    """

    def __init__(self, vision_tower, args, delay_load=False):
        super(HybridVisionTower, self).__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.unfreeze_mm_vision_tower = getattr(args, 'unfreeze_mm_vision_tower', False)
        self.args = args

        # 解析模型名称: hybridmodel-facebook/dinov2-base-&&&-siglip/CLIP-ViT-SO400M-14-384
        model_names = self.vision_tower_name.replace("hybridmodel-", "")
        self.model_names = model_names.split("-&&&-")
        logger.warning(f"Creating a Hybrid Vision Tower with models: {self.model_names}")

        if not delay_load:
            self.load_model()
        elif self.unfreeze_mm_vision_tower:
            self.load_model()

    def load_model(self):
        """加载两个编码器模型"""
        if self.is_loaded:
            logger.warning(f'{self.vision_tower_name} is already loaded, skipping.')
            return

        # 延迟导入避免循环导入
        from .builder import build_vision_tower

        self.vision_model = "hybrid"
        
        # 使用builder加载每个vision tower
        for i, model_name in enumerate(self.model_names, start=1):
            # 创建临时args对象用于build_vision_tower
            class TempArgs:
                def __init__(self, args, vision_tower_name):
                    self.mm_vision_tower = vision_tower_name
                    self.vision_tower = vision_tower_name
                    self.mm_vision_select_layer = getattr(args, 'mm_vision_select_layer', -2)
                    self.mm_vision_select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
                    self.unfreeze_mm_vision_tower = getattr(args, 'unfreeze_mm_vision_tower', False)
                    self.s2 = False
            
            temp_args = TempArgs(self.args, model_name)
            vision_tower_module = build_vision_tower(temp_args, delay_load=False)
            setattr(self, f"vision_tower_{i}", vision_tower_module)

        # 计算总hidden_size（DINOv2: 768 + SigLIP: 1152 = 1920）
        self._hidden_size = sum([getattr(self, f"vision_tower_{i}").hidden_size for i in range(1, len(self.model_names) + 1)])
        self._image_size = 384
        self._patch_size = 16

        # 保存每个编码器的image_processor
        self.image_processor = []
        for i in range(1, len(self.model_names) + 1):
            vision_tower = getattr(self, f"vision_tower_{i}")
            self.image_processor.append(vision_tower.image_processor)

        # 设置梯度要求
        for i in range(1, len(self.model_names) + 1):
            getattr(self, f"vision_tower_{i}").requires_grad_(self.unfreeze_mm_vision_tower)

        self.is_loaded = True
        logger.info(f"Hybrid Vision Tower loaded: hidden_size={self._hidden_size}, image_size={self._image_size}, patch_size={self._patch_size}")

    def forward(self, images):
        """
        前向传播融合过程:
        1. 独立编码：每个编码器处理原始图像（使用各自的image_processor）
        2. 空间分辨率统一：统一到576 tokens (24×24)
        3. 特征维度拼接：在最后一个维度拼接 -> [B, 576, 1920]
        
        Args:
            images: 可以是以下格式之一:
                - PIL图像列表
                - 已处理的tensor [B, C, H, W]（这种情况下假设输入是预处理过的，直接使用）
        
        Returns:
            output_tensor: [B, 576, 1920] 融合后的特征
        """
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            output_images_features = []
            print("HHHHHHHH\n hhhhhhh")
            for i in range(1, len(self.model_names) + 1):
                vision_tower = getattr(self, f"vision_tower_{i}")
                processor = self.image_processor[i-1]
                
                # 处理图像输入：使用各自的processor
                if isinstance(images, list):
                    # PIL图像列表：使用对应的processor处理
                    processed_images = []
                    for img in images:
                        if hasattr(img, 'mode'):  # PIL Image
                            processed = processor(img, return_tensors="pt")['pixel_values'].squeeze(0)
                        else:  # 已经是tensor [C, H, W]
                            processed = img
                        processed_images.append(processed)
                    
                    if len(processed_images) > 1:
                        batch_tensor = torch.stack(processed_images)
                    else:
                        batch_tensor = processed_images[0].unsqueeze(0)
                else:
                    # 单个tensor输入: [B, C, H, W] 或 [C, H, W]
                    if images.dim() == 3:
                        batch_tensor = images.unsqueeze(0)
                    else:
                        batch_tensor = images
                    
                    # 如果输入已经是tensor，假设它可能需要进行一些标准化
                    # 但为了简化，我们直接使用（在实际使用中可能需要根据具体情况调整）
                    # 注意：如果输入已经是处理过的tensor，可能需要重新处理以确保兼容性
                
                # 移动到正确的设备和数据类型
                batch_tensor = batch_tensor.to(device=self.device, dtype=self.dtype)
                
                # 前向传播获取特征
                image_features = vision_tower(batch_tensor)  # [B, num_tokens, dim]
                
                # 空间分辨率统一到576 tokens (24×24)
                b, num_tokens, dim = image_features.shape
                if num_tokens != self.image_token_len:
                    target_h = target_w = int(np.sqrt(self.image_token_len))  # 24
                    h = w = int(np.sqrt(num_tokens))
                    
                    # 重塑为空间网格 [B, H, W, C]
                    image_features = image_features.view(b, h, w, dim)
                    # 转换为 [B, C, H, W] 用于插值
                    image_features = image_features.permute(0, 3, 1, 2).contiguous()
                    # 双线性插值到目标尺寸
                    image_features = F.interpolate(
                        image_features.to(torch.float32), 
                        size=(target_h, target_w), 
                        mode='bilinear', 
                        align_corners=False
                    ).to(image_features.dtype)
                    # 转换回 [B, H*W, C]
                    image_features = image_features.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
                
                output_images_features.append(image_features)
            
            # 在特征维度拼接: [B, 576, 768] + [B, 576, 1152] -> [B, 576, 1920]
            output_tensor = torch.cat(output_images_features, dim=-1)
            
            return output_tensor

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def patch_size(self):
        return self._patch_size

    @property
    def image_size(self):
        return self._image_size

    @property
    def image_token_len(self):
        return (self.image_size // self.patch_size) ** 2

    @property
    def dtype(self):
        # Dynamically infer the dtype from the first parameter, if not explicitly specified
        if hasattr(self.vision_tower_1, 'dtype'):
            return self.vision_tower_1.dtype
        else:
            params = list(self.vision_tower_1.parameters())
            return params[0].dtype if len(params) > 0 else torch.float32  # Default to torch.float32 if no parameters

    @property
    def device(self):
        # Dynamically infer the device from the first parameter, if not explicitly specified
        if hasattr(self.vision_tower_1, 'device'):
            return self.vision_tower_1.device
        else:
            params = list(self.vision_tower_1.parameters())
            return params[0].device if len(params) > 0 else torch.device("cpu")  # Default to CPU if no parameters
