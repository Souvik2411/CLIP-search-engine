import time
import uuid
from PIL import Image
from typing import Optional, Dict, Any
import logging
from datetime import datetime

from app.services.clip_service import CLIPService
from app.services.color_service import color_service
from app.services.texture_service import texture_service
from app.services.material_service import MaterialService
from app.services.style_service import StyleService
from app.services.faiss_service import FAISSService
from app.services.metadata_db_service import metadata_db_service
from app.services.llm_service import llm_service
from app.services.s3_service import s3_service

logger = logging.getLogger(__name__)


class IndexingService:
    """
    Offline indexing pipeline that extracts all features from an image
    and stores them in FAISS + SQLite.
    """

    def __init__(
        self,
        clip_service: CLIPService,
        faiss_service: FAISSService
    ):
        self.clip_service = clip_service
        self.faiss_service = faiss_service

        # Initialize services that depend on CLIP
        self.material_service = MaterialService(clip_service)
        self.style_service = StyleService(clip_service)

    def process_image(
        self,
        image: Image.Image,
        image_id: Optional[str] = None,
        s3_key: Optional[str] = None,
        generate_summary: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single image through the complete extraction pipeline.

        This runs offline (during upload) and can take 3-10 seconds.

        Args:
            image: PIL Image object
            image_id: Optional image ID (generated if not provided)
            s3_key: S3 key for the image (optional)
            generate_summary: Whether to generate LLM summary

        Returns:
            Dictionary with all extracted features and metadata
        """
        start_time = time.time()

        # Generate image ID if not provided
        if image_id is None:
            image_id = f"img_{uuid.uuid4().hex[:12]}"

        logger.info(f"Processing image {image_id}...")

        # ===== PHASE 1: CLIP Features (600-1000ms) =====
        logger.info("Extracting CLIP features...")
        clip_start = time.time()

        # Get embedding and labels in one call for efficiency
        embedding, arch_labels, object_labels = self.clip_service.get_image_embedding_and_dual_labels(
            image=image,
            top_k_arch=5,
            top_k_objects=10
        )

        # Combine labels
        all_labels = [
            {"label": label, "confidence": float(score), "type": "architecture"}
            for label, score in arch_labels
        ]
        all_labels.extend([
            {"label": label, "confidence": float(score), "type": "object"}
            for label, score in object_labels
        ])

        primary_label = arch_labels[0][0] if arch_labels else "unknown"

        clip_time = time.time() - clip_start
        logger.info(f"CLIP features extracted in {clip_time:.2f}s")

        # ===== PHASE 2: Color Features (50-100ms) =====
        logger.info("Extracting color features...")
        color_start = time.time()

        color_features = color_service.extract_all_color_features(image, n_colors=8)

        color_time = time.time() - color_start
        logger.info(f"Color features extracted in {color_time:.2f}s")

        # ===== PHASE 3: Texture Features (100-200ms) =====
        logger.info("Extracting texture features...")
        texture_start = time.time()

        texture_features = texture_service.extract_texture_features(image)

        texture_time = time.time() - texture_start
        logger.info(f"Texture features extracted in {texture_time:.2f}s")

        # ===== PHASE 4: Material Detection (200-400ms) =====
        logger.info("Detecting materials...")
        material_start = time.time()

        material_features = self.material_service.extract_material_features(
            image,
            top_k_materials=5,
            top_k_categories=3
        )

        material_time = time.time() - material_start
        logger.info(f"Materials detected in {material_time:.2f}s")

        # ===== PHASE 5: Style Classification (200-400ms) =====
        logger.info("Classifying style...")
        style_start = time.time()

        style_features = self.style_service.extract_style_features(
            image,
            top_k_styles=3,
            top_k_scenes=2,
            top_k_ambiance=4
        )

        style_time = time.time() - style_start
        logger.info(f"Style classified in {style_time:.2f}s")

        # ===== PHASE 6: Get S3 URL (if applicable) =====
        s3_url = None
        if s3_key:
            try:
                s3_url = s3_service.get_presigned_url(s3_key)
            except Exception as e:
                logger.warning(f"Could not generate S3 URL: {e}")
                s3_url = f"s3://{s3_service.settings.S3_BUCKET_NAME}/{s3_key}"
        else:
            s3_url = f"local://{image_id}"

        # ===== PHASE 7: LLM Summary (800-1500ms) - Optional =====
        summary = None
        llm_time = 0

        if generate_summary:
            logger.info("Generating LLM summary...")
            llm_start = time.time()

            try:
                # Prepare context for LLM
                label_names = [label["label"] for label in all_labels[:10]]
                material_names = [m["name"] for m in material_features["materials"][:3]]
                style_names = [s["name"] for s in style_features["styles"][:2]]

                # Create concise prompt
                prompt = f"""Analyze this interior/architectural image and provide a brief, natural description (2-3 sentences).

Detected elements:
- Objects: {', '.join(label_names)}
- Materials: {', '.join(material_names)}
- Style: {', '.join(style_names)}
- Colors: {color_features['color_temperature']} tones, {color_features['brightness_category']}
- Scene: {style_features['primary_scene']}

Provide a concise, flowing description that would help someone understand what this image shows."""

                summary = llm_service.generate_summary(
                    prompt=prompt,
                    max_tokens=150
                )

            except Exception as e:
                logger.error(f"Error generating summary: {e}")
                summary = f"A {style_features['primary_scene']} featuring {', '.join(label_names[:3])}."

            llm_time = time.time() - llm_start
            logger.info(f"Summary generated in {llm_time:.2f}s")

        # ===== PHASE 8: Add to FAISS Index =====
        logger.info("Adding to FAISS index...")

        # Get current index size to know the faiss_index position
        faiss_index = self.faiss_service.size

        # Add to FAISS - store s3_key (raw path) not s3_url (presigned URL)
        # Presigned URLs expire, so we store the key and generate URLs on-demand
        faiss_metadata = {
            "image_id": image_id,
            "s3_key": s3_key if s3_key else f"local/{image_id}"
        }

        self.faiss_service.add_embeddings(
            embeddings=embedding.reshape(1, -1),
            metadata_list=[faiss_metadata]
        )

        # ===== PHASE 9: Compile Metadata =====
        processing_time = time.time() - start_time

        metadata = {
            "image_id": image_id,
            "s3_url": s3_url,
            "faiss_index": faiss_index,
            "upload_date": datetime.now().isoformat(),
            "processing_time": round(processing_time, 2),

            # Features
            "clip_labels": all_labels,
            "primary_label": primary_label,
            "color": color_features,
            "texture": texture_features,
            "materials": material_features,
            "style": style_features,
            "summary": summary,

            # Timing breakdown (for optimization)
            "timing": {
                "clip": round(clip_time, 2),
                "color": round(color_time, 2),
                "texture": round(texture_time, 2),
                "material": round(material_time, 2),
                "style": round(style_time, 2),
                "llm": round(llm_time, 2),
                "total": round(processing_time, 2)
            }
        }

        # ===== PHASE 10: Save to SQLite =====
        logger.info("Saving metadata to database...")
        metadata_db_service.insert_image_metadata(metadata)

        # ===== PHASE 11: Save FAISS index =====
        logger.info("Saving FAISS index...")
        self.faiss_service.save_index()

        logger.info(f"Image {image_id} processed successfully in {processing_time:.2f}s")

        return metadata

    def process_batch(
        self,
        images: list[tuple[Image.Image, str]],
        generate_summaries: bool = True
    ) -> list[Dict[str, Any]]:
        """
        Process multiple images in batch.

        Args:
            images: List of (image, s3_key) tuples
            generate_summaries: Whether to generate LLM summaries

        Returns:
            List of metadata dictionaries
        """
        results = []

        for i, (image, s3_key) in enumerate(images):
            logger.info(f"Processing image {i+1}/{len(images)}")

            try:
                metadata = self.process_image(
                    image=image,
                    s3_key=s3_key,
                    generate_summary=generate_summaries
                )
                results.append(metadata)

            except Exception as e:
                logger.error(f"Error processing image {i+1}: {e}")
                # Continue with next image
                continue

        logger.info(f"Batch processing complete: {len(results)}/{len(images)} images processed")

        return results

    def reindex_image(self, image_id: str, image: Image.Image) -> bool:
        """
        Reindex an existing image with updated features.

        Args:
            image_id: Existing image ID
            image: PIL Image object

        Returns:
            True if successful
        """
        try:
            # Get existing metadata
            existing_metadata = metadata_db_service.get_image_metadata(image_id)

            if not existing_metadata:
                logger.error(f"Image {image_id} not found in database")
                return False

            s3_url = existing_metadata.get("s3_url")

            # Remove from FAISS (requires rebuilding index)
            # This is expensive, so we'll just add new embedding
            # and mark old one as deleted in metadata

            # Process image again
            metadata = self.process_image(
                image=image,
                image_id=image_id,
                s3_key=None,
                generate_summary=True
            )

            # Update S3 URL from existing metadata
            metadata["s3_url"] = s3_url

            logger.info(f"Image {image_id} reindexed successfully")
            return True

        except Exception as e:
            logger.error(f"Error reindexing image {image_id}: {e}")
            return False


# Note: This service will be initialized in routes with CLIP and FAISS services
# indexing_service = IndexingService(clip_service, faiss_service)
