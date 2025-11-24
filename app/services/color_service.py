import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from typing import Optional
import logging
import colorsys

logger = logging.getLogger(__name__)


class ColorService:
    """Service for extracting color features from images."""

    def __init__(self):
        self.n_colors = 8  # Number of dominant colors to extract

    def rgb_to_hex(self, rgb: tuple) -> str:
        """Convert RGB tuple to hex color string."""
        return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

    def get_color_temperature(self, rgb: tuple) -> str:
        """
        Determine if a color is warm or cool.

        Args:
            rgb: RGB tuple (0-255 range)

        Returns:
            "warm", "cool", or "neutral"
        """
        r, g, b = rgb

        # Convert to HSV for better temperature detection
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        h = h * 360  # Convert to degrees

        # Low saturation = neutral
        if s < 0.15:
            return "neutral"

        # Warm colors: red, orange, yellow (0-60 degrees and 300-360 degrees)
        if (h >= 0 and h <= 60) or (h >= 300 and h <= 360):
            return "warm"

        # Cool colors: green, blue, purple (120-300 degrees)
        if h >= 120 and h <= 300:
            return "cool"

        # Neutral zone
        return "neutral"

    def extract_color_palette(self, image: Image.Image, n_colors: Optional[int] = None) -> list[str]:
        """
        Extract dominant color palette using K-Means clustering.

        Args:
            image: PIL Image object
            n_colors: Number of colors to extract (default: 8)

        Returns:
            List of hex color codes
        """
        if n_colors is None:
            n_colors = self.n_colors

        # Convert PIL to numpy array
        img_array = np.array(image)

        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            logger.warning("Image is not RGB, converting to RGB")
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

        # Reshape image to be a list of pixels
        pixels = img_bgr.reshape(-1, 3)

        # Remove very dark and very bright pixels (likely shadows/highlights)
        brightness = np.mean(pixels, axis=1)
        mask = (brightness > 20) & (brightness < 235)
        filtered_pixels = pixels[mask]

        if len(filtered_pixels) < n_colors:
            filtered_pixels = pixels  # Fallback to all pixels

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(filtered_pixels)

        # Get cluster centers (dominant colors)
        colors = kmeans.cluster_centers_

        # Count pixels in each cluster
        labels = kmeans.labels_
        counts = np.bincount(labels)

        # Sort colors by frequency
        indices = np.argsort(-counts)
        sorted_colors = colors[indices]

        # Convert BGR to RGB and then to hex
        hex_colors = []
        for color in sorted_colors:
            # BGR to RGB
            rgb = (color[2], color[1], color[0])
            hex_color = self.rgb_to_hex(rgb)
            hex_colors.append(hex_color)

        return hex_colors

    def get_color_statistics(self, image: Image.Image) -> dict:
        """
        Extract comprehensive color statistics from image.

        Args:
            image: PIL Image object

        Returns:
            Dictionary with color statistics
        """
        # Convert PIL to numpy array
        img_array = np.array(image)

        # Convert to HSV for better color analysis
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        else:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        # Extract HSV channels
        h, s, v = cv2.split(img_hsv)

        # Calculate statistics
        avg_brightness = float(np.mean(v)) / 255.0
        avg_saturation = float(np.mean(s)) / 255.0

        # Determine overall color temperature
        # Sample dominant pixels for temperature
        pixels_rgb = img_array.reshape(-1, 3)
        sample_size = min(1000, len(pixels_rgb))
        sample_indices = np.random.choice(len(pixels_rgb), sample_size, replace=False)
        sample_pixels = pixels_rgb[sample_indices]

        warm_count = 0
        cool_count = 0
        neutral_count = 0

        for pixel in sample_pixels:
            temp = self.get_color_temperature(tuple(pixel))
            if temp == "warm":
                warm_count += 1
            elif temp == "cool":
                cool_count += 1
            else:
                neutral_count += 1

        # Determine dominant temperature
        if warm_count > cool_count and warm_count > neutral_count:
            color_temp = "warm"
        elif cool_count > warm_count and cool_count > neutral_count:
            color_temp = "cool"
        else:
            color_temp = "neutral"

        # Determine brightness category
        if avg_brightness > 0.7:
            brightness_category = "bright"
        elif avg_brightness > 0.4:
            brightness_category = "medium"
        else:
            brightness_category = "dark"

        # Determine saturation category
        if avg_saturation > 0.6:
            saturation_category = "vibrant"
        elif avg_saturation > 0.3:
            saturation_category = "moderate"
        else:
            saturation_category = "muted"

        return {
            "brightness": round(avg_brightness, 3),
            "saturation": round(avg_saturation, 3),
            "color_temperature": color_temp,
            "brightness_category": brightness_category,
            "saturation_category": saturation_category
        }

    def extract_all_color_features(self, image: Image.Image, n_colors: int = 8) -> dict:
        """
        Extract all color features in one call for efficiency.

        Args:
            image: PIL Image object
            n_colors: Number of colors in palette

        Returns:
            Dictionary with all color features
        """
        palette = self.extract_color_palette(image, n_colors)
        stats = self.get_color_statistics(image)

        return {
            "palette": palette,
            "dominant_color": palette[0] if palette else "#000000",
            **stats
        }


# Global instance
color_service = ColorService()
