# models.py
# the five rungs of the ladder, all exposing the same interface: a callable that
# takes a (B, 3, H, W) float tensor and returns (B, K) logits. each model also
# declares its expected input size and normalization via get_transforms(), so the
# notebooks stay thin and the only thing that varies between them is the name.

import torch
import torch.nn as nn
import torchvision.transforms as T

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
half_mean = [0.5, 0.5, 0.5]   # google/vit-base-patch16-224 processor
half_std = [0.5, 0.5, 0.5]

# (input_size, mean, std) per model
model_input_config = {
    "cnn":      (224, imagenet_mean, imagenet_std),
    "resnet50": (224, imagenet_mean, imagenet_std),
    "vit":      (224, half_mean, half_std),
    "dinov2":   (224, imagenet_mean, imagenet_std),
    "retfound": (224, imagenet_mean, imagenet_std),
}


def get_transforms(model_name):
    # returns (train_transform, eval_transform). inputs are already paper-cropped
    # square pngs, so here we only do light augmentation + resize + normalize.
    size, mean, std = model_input_config[model_name]
    train_tf = T.Compose([
        T.Resize((size, size)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    eval_tf = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_tf, eval_tf


# ----------------------------------------------------------------------------
# rung 1: vanilla cnn (trained from scratch -- the lower-bound baseline)
# ----------------------------------------------------------------------------
class VanillaCNN(nn.Module):
    def __init__(self, num_classes, in_size=224):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


# ----------------------------------------------------------------------------
# rung 3: huggingface vit wrapper so model(x) -> logits (engine expects a tensor)
# ----------------------------------------------------------------------------
class HFViTWrapper(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def forward(self, x):
        return self.vit(pixel_values=x).logits


def load_retfound_weights(model, ckpt_path):
    # load RETFound MAE-pretrained ViT-L/16 encoder weights into a timm vit.
    # the official checkpoint stores the state dict under "model"; the decoder
    # and the (randomly initialized) classification head are dropped.
    # weights_only=False: the official checkpoint bundles an argparse.Namespace,
    # which pytorch 2.6+ refuses under the new weights_only=True default (trusted source).
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    state = {k: v for k, v in state.items()
             if not k.startswith("head") and not k.startswith("decoder")
             and "mask_token" not in k}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"retfound weights loaded. missing={len(missing)} unexpected={len(unexpected)}")
    return model


def build_model(name, num_classes, retfound_ckpt=None):
    # factory. for 'retfound', pass retfound_ckpt=path to the downloaded weights;
    # if None, the ViT-L/16 is left ImageNet-initialized (still runnable, but not
    # the domain-pretrained comparison).
    name = name.lower()
    if name == "cnn":
        return VanillaCNN(num_classes)

    if name == "resnet50":
        import torchvision.models as tvm
        model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "vit":
        from transformers import ViTForImageClassification
        vit = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        return HFViTWrapper(vit)

    if name == "dinov2":
        import timm
        # general-purpose self-supervised foundation model (patch14, 224 ok)
        return timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=True, num_classes=num_classes, img_size=224,
        )

    if name == "retfound":
        import timm
        model = timm.create_model(
            "vit_large_patch16_224", pretrained=False, num_classes=num_classes,
        )
        if retfound_ckpt is not None:
            load_retfound_weights(model, retfound_ckpt)
        return model

    raise ValueError(f"unknown model name: {name}")


# ----------------------------------------------------------------------------
# grad-cam configuration per model: picks the right "last spatial feature map"
# layer and provides the reshape_transform required by transformer backbones
# (pytorch-grad-cam needs (B, C, H, W); transformers output (B, n_tokens, D)).
# returns (target_layers, reshape_transform). reshape_transform is None for
# convnets.
# ----------------------------------------------------------------------------
def _make_vit_reshape(grid_h, grid_w, drop_n_prefix_tokens=1):
    # drop CLS (and any register) tokens, then fold the remaining patch tokens
    # into a (B, D, H, W) spatial map so grad-cam can pool gradients normally.
    def _t(tensor):
        x = tensor[:, drop_n_prefix_tokens:, :]
        b, _, d = x.shape
        x = x.reshape(b, grid_h, grid_w, d).permute(0, 3, 1, 2).contiguous()
        return x
    return _t


def gradcam_setup(name, model):
    name = name.lower()
    if name == "cnn":
        # last conv block of VanillaCNN (output 14x14x256 at 224 input)
        return [model.features[-1]], None

    if name == "resnet50":
        # last residual block of ResNet50 (output 7x7x2048 at 224 input)
        return [model.layer4[-1]], None

    if name == "vit":
        # huggingface ViT-B/16 at 224 -> 14x14 patch grid (+1 CLS)
        last_block = model.vit.vit.encoder.layer[-1]
        target = last_block.layernorm_before
        return [target], _make_vit_reshape(14, 14, drop_n_prefix_tokens=1)

    if name == "dinov2":
        # timm vit_base_patch14_dinov2 at 224 -> 16x16 patch grid (+1 CLS).
        # Note: lvd142m variant has no register tokens, so drop only CLS.
        target = model.blocks[-1].norm1
        return [target], _make_vit_reshape(16, 16, drop_n_prefix_tokens=1)

    if name == "retfound":
        # timm vit_large_patch16_224 -> 14x14 patch grid (+1 CLS)
        target = model.blocks[-1].norm1
        return [target], _make_vit_reshape(14, 14, drop_n_prefix_tokens=1)

    raise ValueError(f"no grad-cam config for model name: {name}")
