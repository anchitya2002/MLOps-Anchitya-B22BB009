"""
model.py - ViT-S Model Setup with Optional LoRA via PEFT.

Provides functions to create a ViT-S model pretrained on ImageNet,
optionally apply LoRA adapters via PEFT, and count trainable parameters.
"""

import timm
import torch.nn as nn
from peft import get_peft_model, LoraConfig


def create_vit_model(num_classes=100, pretrained=True, model_name="vit_small_patch16_224"):
    """Create a ViT-S model with a classification head for CIFAR-100.

    Args:
        num_classes: Number of output classes (100 for CIFAR-100).
        pretrained: Whether to load ImageNet pretrained weights.
        model_name: timm model name.

    Returns:
        timm ViT model with updated classification head.
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


def apply_lora(model, rank=8, alpha=16, dropout=0.1, target_modules=None):
    """Apply LoRA adapters to a ViT model using PEFT.

    LoRA is applied to the attention QKV weights. The classification head
    is kept trainable via modules_to_save.

    Args:
        model: Base ViT model from timm.
        rank: LoRA rank (r parameter).
        alpha: LoRA alpha (scaling factor).
        dropout: LoRA dropout rate.
        target_modules: List of module name patterns to apply LoRA to.

    Returns:
        PEFT-wrapped model with LoRA adapters.
    """
    if target_modules is None:
        target_modules = ["qkv"]

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        modules_to_save=["head"],  # Keep classification head trainable
    )

    peft_model = get_peft_model(model, lora_config)
    return peft_model


def get_model(use_lora=False, rank=8, alpha=16, dropout=0.1,
              num_classes=100, pretrained=True, model_name="vit_small_patch16_224"):
    """Get a ViT-S model, optionally with LoRA.

    Args:
        use_lora: Whether to apply LoRA adapters.
        rank: LoRA rank.
        alpha: LoRA alpha.
        dropout: LoRA dropout.
        num_classes: Number of classes.
        pretrained: Use pretrained weights.
        model_name: timm model name.

    Returns:
        Tuple of (model, trainable_params, total_params).
    """
    model = create_vit_model(
        num_classes=num_classes,
        pretrained=pretrained,
        model_name=model_name
    )

    if use_lora:
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False

        # Apply LoRA (this will make LoRA params + head trainable)
        model = apply_lora(model, rank=rank, alpha=alpha, dropout=dropout)
    else:
        # Without LoRA: freeze backbone, train only classification head
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze classification head
        for param in model.head.parameters():
            param.requires_grad = True

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Model: {model_name}")
    print(f"LoRA: {'Yes (r={}, alpha={}, dropout={})'.format(rank, alpha, dropout) if use_lora else 'No'}")
    print(f"Trainable params: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    return model, trainable_params, total_params


def get_model_partial_frozen(rank=8, alpha=16, dropout=0.1,
                              num_classes=100, freeze_layers=6,
                              model_name="vit_small_patch16_224"):
    """[Optional Q7] Get a ViT-S model with partial freezing + LoRA.

    Freezes the first `freeze_layers` transformer blocks, keeps the rest
    trainable, and applies LoRA to the frozen part.

    Args:
        rank: LoRA rank.
        alpha: LoRA alpha.
        dropout: LoRA dropout.
        num_classes: Number of classes.
        freeze_layers: Number of initial transformer blocks to freeze.
        model_name: timm model name.

    Returns:
        Tuple of (model, trainable_params, total_params).
    """
    model = create_vit_model(
        num_classes=num_classes,
        pretrained=True,
        model_name=model_name
    )

    # First freeze all
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layers after freeze_layers
    total_blocks = len(model.blocks)
    for i in range(freeze_layers, total_blocks):
        for param in model.blocks[i].parameters():
            param.requires_grad = True

    # Unfreeze classification head
    for param in model.head.parameters():
        param.requires_grad = True

    # Unfreeze norm layer
    if hasattr(model, 'norm'):
        for param in model.norm.parameters():
            param.requires_grad = True

    # Apply LoRA only to the frozen blocks' qkv layers
    frozen_target_modules = []
    for i in range(freeze_layers):
        frozen_target_modules.append(f"blocks.{i}.attn.qkv")

    if frozen_target_modules:
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=frozen_target_modules,
            bias="none",
            modules_to_save=["head"],
        )
        model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Partial Frozen Model (freeze_layers={freeze_layers})")
    print(f"LoRA on frozen blocks: r={rank}, alpha={alpha}, dropout={dropout}")
    print(f"Trainable params: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    return model, trainable_params, total_params
