#!/usr/bin/env python3
"""
Convert CLIP model to ONNX format for faster CPU inference.

This script converts the Hugging Face CLIP model to ONNX format, which provides
2-3x faster inference on CPU compared to standard PyTorch.

Usage:
    python scripts/convert_to_onnx.py
    python scripts/convert_to_onnx.py --output-dir models/onnx
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import onnx
from transformers import CLIPModel, CLIPProcessor
import logging

# Fix encoding issues on Windows
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_clip_to_onnx(
    model_name: str,
    output_dir: Path,
    opset_version: int = 18  # Changed to 18 to avoid version warnings
):
    """
    Convert CLIP model to ONNX format.

    Args:
        model_name: Name of the CLIP model to convert
        output_dir: Directory to save ONNX models
        opset_version: ONNX opset version
    """
    logger.info(f"Loading CLIP model: {model_name}")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    model.eval()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert Vision Model (for images)
    logger.info("Converting vision model to ONNX...")
    vision_onnx_path = output_dir / "clip_vision.onnx"

    # Create dummy input for vision model
    dummy_pixel_values = torch.randn(1, 3, 224, 224)

    # Export vision model
    torch.onnx.export(
        model.vision_model,
        dummy_pixel_values,
        vision_onnx_path,
        input_names=['pixel_values'],
        output_names=['image_embeds'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'image_embeds': {0: 'batch_size'}
        },
        opset_version=opset_version,
        do_constant_folding=True
    )

    logger.info(f"Vision model saved to: {vision_onnx_path}")

    # Convert Text Model (for text)
    logger.info("Converting text model to ONNX...")
    text_onnx_path = output_dir / "clip_text.onnx"

    # Create dummy input for text model
    dummy_input_ids = torch.randint(0, 1000, (1, 77))
    dummy_attention_mask = torch.ones(1, 77, dtype=torch.long)

    # Export text model
    torch.onnx.export(
        model.text_model,
        (dummy_input_ids, dummy_attention_mask),
        text_onnx_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['text_embeds'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'text_embeds': {0: 'batch_size'}
        },
        opset_version=opset_version,
        do_constant_folding=True
    )

    logger.info(f"Text model saved to: {text_onnx_path}")

    # Verify ONNX models
    logger.info("Verifying ONNX models...")
    vision_model = onnx.load(vision_onnx_path)
    onnx.checker.check_model(vision_model)
    logger.info("✓ Vision model verified")

    text_model = onnx.load(text_onnx_path)
    onnx.checker.check_model(text_model)
    logger.info("✓ Text model verified")

    # Test inference
    logger.info("Testing ONNX inference...")
    import onnxruntime as ort

    # Test vision model
    vision_session = ort.InferenceSession(str(vision_onnx_path))
    vision_output = vision_session.run(
        None,
        {'pixel_values': dummy_pixel_values.numpy()}
    )
    logger.info(f"✓ Vision model output shape: {vision_output[0].shape}")

    # Test text model
    text_session = ort.InferenceSession(str(text_onnx_path))
    text_output = text_session.run(
        None,
        {
            'input_ids': dummy_input_ids.numpy(),
            'attention_mask': dummy_attention_mask.numpy()
        }
    )
    logger.info(f"✓ Text model output shape: {text_output[0].shape}")

    # Save model info
    info_path = output_dir / "model_info.txt"
    with open(info_path, 'w') as f:
        f.write(f"CLIP Model: {model_name}\n")
        f.write(f"ONNX Opset Version: {opset_version}\n")
        f.write(f"Vision Model: clip_vision.onnx\n")
        f.write(f"Text Model: clip_text.onnx\n")
        f.write(f"Vision Output Dimension: {vision_output[0].shape[-1]}\n")
        f.write(f"Text Output Dimension: {text_output[0].shape[-1]}\n")

    logger.info(f"Model info saved to: {info_path}")
    logger.info("="*50)
    logger.info("ONNX conversion complete!")
    logger.info("="*50)
    logger.info(f"Models saved in: {output_dir.absolute()}")
    logger.info("")
    logger.info("To use ONNX models, update your .env:")
    logger.info("USE_ONNX=true")
    logger.info(f"ONNX_VISION_MODEL_PATH={vision_onnx_path.absolute()}")
    logger.info(f"ONNX_TEXT_MODEL_PATH={text_onnx_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CLIP model to ONNX format"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="CLIP model name (default: from settings)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/onnx",
        help="Output directory for ONNX models"
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=18,  # Updated default to 18
        help="ONNX opset version"
    )

    args = parser.parse_args()

    # Get model name from settings if not provided
    if args.model_name is None:
        settings = get_settings()
        model_name = settings.CLIP_MODEL_NAME
    else:
        model_name = args.model_name

    output_dir = Path(args.output_dir)

    try:
        convert_clip_to_onnx(model_name, output_dir, args.opset_version)
    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
