import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime

from app.config import get_settings

logger = logging.getLogger(__name__)


class MetadataDBService:
    """Service for managing structured image metadata in SQLite."""

    def __init__(self):
        self.settings = get_settings()
        self.db_path = Path("data/index/metadata.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        return conn

    def _init_database(self):
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Main images table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                image_id TEXT PRIMARY KEY,
                s3_url TEXT NOT NULL,
                faiss_index INTEGER,
                upload_date TEXT NOT NULL,
                processing_time REAL,

                -- CLIP features
                clip_labels TEXT,  -- JSON array
                primary_label TEXT,

                -- Color features
                color_palette TEXT,  -- JSON array of hex codes
                dominant_color TEXT,
                color_temperature TEXT,  -- warm, cool, neutral
                brightness REAL,
                saturation REAL,
                brightness_category TEXT,  -- bright, medium, dark
                saturation_category TEXT,  -- vibrant, moderate, muted

                -- Material features
                materials TEXT,  -- JSON array
                material_categories TEXT,  -- JSON object
                primary_material TEXT,
                primary_material_category TEXT,

                -- Texture features
                texture_roughness TEXT,  -- smooth, moderate, rough
                texture_pattern TEXT,  -- uniform, regular, irregular
                texture_complexity TEXT,  -- simple, moderate, complex
                texture_detail_level TEXT,  -- low_detail, medium_detail, high_detail
                texture_edge_level TEXT,  -- low_edges, medium_edges, high_edges
                texture_contrast REAL,
                texture_homogeneity REAL,

                -- Style features
                styles TEXT,  -- JSON array
                primary_style TEXT,
                scenes TEXT,  -- JSON array (room types)
                primary_scene TEXT,
                ambiance TEXT,  -- JSON array

                -- LLM-generated summary
                summary TEXT,

                -- Metadata
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for fast filtering
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_color_temp
            ON images(color_temperature)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_brightness_cat
            ON images(brightness_category)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_primary_material
            ON images(primary_material_category)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_texture_roughness
            ON images(texture_roughness)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_primary_style
            ON images(primary_style)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_primary_scene
            ON images(primary_scene)
        """)

        # Create full-text search table for text fields
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS images_fts USING fts5(
                image_id,
                clip_labels,
                summary,
                styles,
                scenes,
                materials,
                content=images
            )
        """)

        conn.commit()
        conn.close()

        logger.info(f"Database initialized at {self.db_path}")

    def insert_image_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Insert or update image metadata.

        Args:
            metadata: Dictionary containing all image metadata

        Returns:
            True if successful
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Prepare data
            image_id = metadata.get("image_id")

            cursor.execute("""
                INSERT OR REPLACE INTO images (
                    image_id, s3_url, faiss_index, upload_date, processing_time,
                    clip_labels, primary_label,
                    color_palette, dominant_color, color_temperature,
                    brightness, saturation, brightness_category, saturation_category,
                    materials, material_categories, primary_material, primary_material_category,
                    texture_roughness, texture_pattern, texture_complexity,
                    texture_detail_level, texture_edge_level, texture_contrast, texture_homogeneity,
                    styles, primary_style, scenes, primary_scene, ambiance,
                    summary, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                image_id,
                metadata.get("s3_url"),
                metadata.get("faiss_index"),
                metadata.get("upload_date", datetime.now().isoformat()),
                metadata.get("processing_time"),

                # CLIP
                json.dumps(metadata.get("clip_labels", [])),
                metadata.get("primary_label"),

                # Color
                json.dumps(metadata.get("color", {}).get("palette", [])),
                metadata.get("color", {}).get("dominant_color"),
                metadata.get("color", {}).get("color_temperature"),
                metadata.get("color", {}).get("brightness"),
                metadata.get("color", {}).get("saturation"),
                metadata.get("color", {}).get("brightness_category"),
                metadata.get("color", {}).get("saturation_category"),

                # Materials
                json.dumps(metadata.get("materials", {}).get("materials", [])),
                json.dumps(metadata.get("materials", {}).get("categories", {})),
                metadata.get("materials", {}).get("primary_material"),
                metadata.get("materials", {}).get("primary_category"),

                # Texture
                metadata.get("texture", {}).get("classification", {}).get("roughness"),
                metadata.get("texture", {}).get("classification", {}).get("pattern"),
                metadata.get("texture", {}).get("classification", {}).get("complexity"),
                metadata.get("texture", {}).get("classification", {}).get("detail_level"),
                metadata.get("texture", {}).get("edges", {}).get("level"),
                metadata.get("texture", {}).get("glcm", {}).get("contrast"),
                metadata.get("texture", {}).get("glcm", {}).get("homogeneity"),

                # Style
                json.dumps(metadata.get("style", {}).get("style_tags", [])),
                metadata.get("style", {}).get("primary_style"),
                json.dumps(metadata.get("style", {}).get("scene_tags", [])),
                metadata.get("style", {}).get("primary_scene"),
                json.dumps(metadata.get("style", {}).get("ambiance_tags", [])),

                # Summary
                metadata.get("summary"),

                datetime.now().isoformat()
            ))

            # Update FTS table
            cursor.execute("""
                INSERT OR REPLACE INTO images_fts (
                    image_id, clip_labels, summary, styles, scenes, materials
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                image_id,
                json.dumps(metadata.get("clip_labels", [])),
                metadata.get("summary", ""),
                json.dumps(metadata.get("style", {}).get("style_tags", [])),
                json.dumps(metadata.get("style", {}).get("scene_tags", [])),
                json.dumps(metadata.get("materials", {}).get("materials", []))
            ))

            conn.commit()
            logger.info(f"Inserted metadata for image {image_id}")
            return True

        except Exception as e:
            logger.error(f"Error inserting metadata: {e}")
            conn.rollback()
            return False

        finally:
            conn.close()

    def get_image_metadata(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific image.

        Args:
            image_id: Image ID

        Returns:
            Dictionary with metadata or None
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM images WHERE image_id = ?", (image_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    def search_by_filters(
        self,
        color_temp: Optional[str] = None,
        brightness: Optional[str] = None,
        material_category: Optional[str] = None,
        texture_roughness: Optional[str] = None,
        style: Optional[str] = None,
        scene: Optional[str] = None,
        limit: int = 100
    ) -> List[str]:
        """
        Search for image IDs matching filters.

        Args:
            color_temp: Color temperature (warm, cool, neutral)
            brightness: Brightness category (bright, medium, dark)
            material_category: Material category (wood, metal, fabric, etc.)
            texture_roughness: Texture roughness (smooth, moderate, rough)
            style: Primary style
            scene: Primary scene/room type
            limit: Maximum number of results

        Returns:
            List of image IDs
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT image_id FROM images WHERE 1=1"
        params = []

        if color_temp:
            query += " AND color_temperature = ?"
            params.append(color_temp)

        if brightness:
            query += " AND brightness_category = ?"
            params.append(brightness)

        if material_category:
            query += " AND primary_material_category = ?"
            params.append(material_category)

        if texture_roughness:
            query += " AND texture_roughness = ?"
            params.append(texture_roughness)

        if style:
            query += " AND primary_style LIKE ?"
            params.append(f"%{style}%")

        if scene:
            query += " AND primary_scene LIKE ?"
            params.append(f"%{scene}%")

        query += f" LIMIT {limit}"

        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()

        return [row[0] for row in results]

    def full_text_search(self, query: str, limit: int = 100) -> List[str]:
        """
        Perform full-text search on text fields.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of image IDs
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT image_id FROM images_fts
            WHERE images_fts MATCH ?
            LIMIT ?
        """, (query, limit))

        results = cursor.fetchall()
        conn.close()

        return [row[0] for row in results]

    def delete_image(self, image_id: str) -> bool:
        """
        Delete image metadata.

        Args:
            image_id: Image ID

        Returns:
            True if successful
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM images WHERE image_id = ?", (image_id,))
            cursor.execute("DELETE FROM images_fts WHERE image_id = ?", (image_id,))
            conn.commit()
            logger.info(f"Deleted metadata for image {image_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting metadata: {e}")
            conn.rollback()
            return False

        finally:
            conn.close()

    def get_all_image_ids(self) -> List[str]:
        """Get all image IDs in database."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT image_id FROM images")
        results = cursor.fetchall()
        conn.close()

        return [row[0] for row in results]

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM images")
        total_images = cursor.fetchone()[0]

        # Get distribution of styles
        cursor.execute("""
            SELECT primary_style, COUNT(*) as count
            FROM images
            WHERE primary_style IS NOT NULL
            GROUP BY primary_style
            ORDER BY count DESC
            LIMIT 10
        """)
        top_styles = [{"style": row[0], "count": row[1]} for row in cursor.fetchall()]

        # Get distribution of scenes
        cursor.execute("""
            SELECT primary_scene, COUNT(*) as count
            FROM images
            WHERE primary_scene IS NOT NULL
            GROUP BY primary_scene
            ORDER BY count DESC
            LIMIT 10
        """)
        top_scenes = [{"scene": row[0], "count": row[1]} for row in cursor.fetchall()]

        conn.close()

        return {
            "total_images": total_images,
            "top_styles": top_styles,
            "top_scenes": top_scenes
        }


# Global instance
metadata_db_service = MetadataDBService()
