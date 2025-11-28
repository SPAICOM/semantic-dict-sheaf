import timm
import torch

model_names = [
    'vit_small_patch16_224',
    'vit_small_patch16_384',
    'vit_small_patch32_224',
    'vit_small_patch32_384',
    'levit_128',
    'levit_192',
    'efficientvit_m4',
    'levit_conv_128',
    'volo_d1_224',
    'volo_d1_384',
]


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


for name in model_names:
    try:
        model = timm.create_model(name, pretrained=False)
        n_params = count_parameters(model)
        print(
            f'{name:20s} -> {n_params:>12,} params  ({n_params / 1e6:6.2f}M)'
        )
    except Exception as e:
        print(f'{name:20s} -> ERROR: {e}')
