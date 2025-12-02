import timm
import torch


def extract_vec(model, x):
    with torch.no_grad():
        try:
            y = model.forward_features(x)
        except Exception:
            y = model(x)

    # Unwrap containers
    if isinstance(y, dict):
        for k in ('pre_logits', 'x', 'features', 'feat', 'last_hidden_state'):
            if k in y and torch.is_tensor(y[k]):
                y = y[k]
                break
        else:
            # fall back: first tensor value in dict
            for v in y.values():
                if torch.is_tensor(v):
                    y = v
                    break
    if isinstance(y, list | tuple):
        # take the last tensor-like item
        for z in reversed(y):
            if torch.is_tensor(z):
                y = z
                break

    # Normalize shapes to (B, C)
    if y.ndim == 4:  # (B, C, H, W)
        y = y.mean(dim=(2, 3))
    elif y.ndim == 3:  # (B, N, C) tokens
        # prefer CLS if model has one
        if getattr(model, 'cls_token', None) is not None and y.shape[1] >= 1:
            y = y[:, 0, :]  # CLS token
        else:
            y = y.mean(dim=1)  # mean over tokens
    elif y.ndim != 2:
        raise RuntimeError(f'Unexpected feature shape: {tuple(y.shape)}')

    return int(y.shape[-1])


def make(name):
    # Pass img_size=32; if the model ignores it, that's fine.
    try:
        return timm.create_model(
            name, pretrained=False, num_classes=0, in_chans=3, img_size=32
        )
    except TypeError:
        # Some models don't accept img_size
        return timm.create_model(
            name, pretrained=False, num_classes=0, in_chans=3
        )


def main():
    x = torch.zeros(1, 3, 32, 32)
    hits = []
    for name in timm.list_models():
        try:
            m = make(name).eval()
            d = getattr(m, 'num_features', None)
            if not isinstance(d, int) or d <= 0:
                d = extract_vec(m, x)
            if d == 384:
                hits.append(name)
        except Exception:
            # skip models that can't run at 32×32
            continue

    print('\n'.join(sorted(hits)))


if __name__ == '__main__':
    main()

################################################
# Models with 384-dim features:
# cait_s24_224
# cait_s24_384
# cait_s36_384
# convnext_femto
# convnext_femto_ols
# convnextv2_femto
# crossvit_9_240
# crossvit_9_dagger_240
# deit3_small_patch16_224
# deit3_small_patch16_384
# deit_small_distilled_patch16_224
# deit_small_patch16_224
# edgenext_small_rw
# efficientformerv2_l
# efficientvit_b2
# efficientvit_m4
# efficientvit_m5
# eva02_small_patch14_224
# eva02_small_patch14_336
# flexivit_small
# gmixer_12_224
# gmixer_24_224
# lcnet_075
# levit_128
# levit_128s
# levit_192
# levit_conv_128
# levit_conv_128s
# levit_conv_192
# mobilevit_xs
# mobilevitv2_075
# nest_small
# nest_small_jx
# nest_tiny
# nest_tiny_jx
# pit_xs_224
# pit_xs_distilled_224
# regnetx_004
# repvit_m0_9
# repvit_m1
# resmlp_12_224
# resmlp_24_224
# resmlp_36_224
# sequencer2d_l
# sequencer2d_m
# sequencer2d_s
# tnt_s_patch16_224
# visformer_tiny
# vit_relpos_small_patch16_224
# vit_relpos_small_patch16_rpn_224
# vit_small_patch14_dinov2
# vit_small_patch14_reg4_dinov2
# vit_small_patch16_18x2_224
# vit_small_patch16_224
# vit_small_patch16_36x1_224
# vit_small_patch16_384
# vit_small_patch32_224
# vit_small_patch32_384
# vit_small_patch8_224
# vit_small_r26_s32_224
# vit_small_r26_s32_384
# vit_srelpos_small_patch16_224
# vitamin_small_224
# volo_d1_224
# volo_d1_384
# xcit_small_12_p16_224
# xcit_small_12_p16_384
# xcit_small_12_p8_224
# xcit_small_12_p8_384
# xcit_small_24_p16_224
# xcit_small_24_p16_384
# xcit_small_24_p8_224
# xcit_small_24_p8_384
