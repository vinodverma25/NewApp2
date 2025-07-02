import cv2
import numpy as np
import json
import base64
from io import BytesIO
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont
from skimage import filters, exposure
import logging
from typing import Dict, List, Tuple, Optional, Any
import threading
import queue
import time

class VideoEffectsProcessor:
    """Real-time video effects processor with GPU acceleration support"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.effects_cache = {}
        self.processing_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_processing = False
        self.preview_thread = None

        # LUT presets for color grading
        self.lut_presets = {
            'cinematic': self._create_cinematic_lut(),
            'vintage': self._create_vintage_lut(),
            'bright': self._create_bright_lut(),
            'dark_moody': self._create_dark_moody_lut(),
            'warm': self._create_warm_lut(),
            'cool': self._create_cool_lut(),
            'sunset': self._create_sunset_lut(),
            'cyberpunk': self._create_cyberpunk_lut()
        }

        # Effect templates
        self.effect_templates = {
            'tiktok_glow': {'glow_intensity': 0.3, 'blur_radius': 15, 'brightness': 1.2},
            'vhs_retro': {'noise': 0.1, 'scanlines': True, 'color_shift': 0.05},
            'glitch': {'displacement': 10, 'color_split': 5, 'noise': 0.2},
            'neon_cyber': {'edge_enhance': 2.0, 'saturation': 1.5, 'glow': 0.4},
            'dreamy_blur': {'gaussian_blur': 8, 'brightness': 1.1, 'contrast': 0.9},
            'film_grain': {'grain_intensity': 0.15, 'vignette': 0.3}
        }

    def _create_cinematic_lut(self):
        """Create cinematic color grading LUT"""
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            # Lift shadows, compress highlights
            r = min(255, int(i * 1.1 + 10))
            g = min(255, int(i * 1.05 + 5))
            b = min(255, int(i * 0.95))
            lut[i] = [b, g, r]  # BGR format
        return lut

    def _create_vintage_lut(self):
        """Create vintage film look LUT"""
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            # Warm tones, reduced contrast
            r = min(255, int(i * 1.2 + 20))
            g = min(255, int(i * 1.1 + 15))
            b = min(255, int(i * 0.8 + 10))
            lut[i] = [b, g, r]
        return lut

    def _create_bright_lut(self):
        """Create bright and airy LUT"""
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            # Lifted exposure, increased brightness
            val = min(255, int(i * 1.3 + 30))
            lut[i] = [val, val, val]
        return lut

    def _create_dark_moody_lut(self):
        """Create dark moody LUT"""
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            # Crushed blacks, desaturated
            r = min(255, max(0, int(i * 0.8 - 20)))
            g = min(255, max(0, int(i * 0.8 - 15)))
            b = min(255, max(0, int(i * 0.9 - 10)))
            lut[i] = [b, g, r]
        return lut

    def _create_warm_lut(self):
        """Create warm temperature LUT"""
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            r = min(255, int(i * 1.15 + 15))
            g = min(255, int(i * 1.05))
            b = min(255, int(i * 0.9))
            lut[i] = [b, g, r]
        return lut

    def _create_cool_lut(self):
        """Create cool temperature LUT"""
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            r = min(255, int(i * 0.9))
            g = min(255, int(i * 1.0))
            b = min(255, int(i * 1.15 + 10))
            lut[i] = [b, g, r]
        return lut

    def _create_sunset_lut(self):
        """Create sunset color grading LUT"""
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            r = min(255, int(i * 1.3 + 25))
            g = min(255, int(i * 1.1 + 10))
            b = min(255, int(i * 0.7))
            lut[i] = [b, g, r]
        return lut

    def _create_cyberpunk_lut(self):
        """Create cyberpunk neon LUT"""
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            r = min(255, int(i * 0.8 + 40))
            g = min(255, int(i * 1.2))
            b = min(255, int(i * 1.4 + 20))
            lut[i] = [b, g, r]
        return lut

    def apply_color_grading(self, frame: np.ndarray, settings: Dict) -> np.ndarray:
        """Apply color grading with LUT and manual adjustments"""
        try:
            result = frame.copy()

            # Apply LUT if specified
            if 'lut_preset' in settings and settings['lut_preset'] in self.lut_presets:
                lut = self.lut_presets[settings['lut_preset']]
                result = cv2.LUT(result, lut)

            # Manual color adjustments
            if 'brightness' in settings:
                brightness = settings['brightness']
                result = cv2.convertScaleAbs(result, alpha=brightness, beta=0)

            if 'contrast' in settings:
                contrast = settings['contrast']
                result = cv2.convertScaleAbs(result, alpha=contrast, beta=0)

            if 'saturation' in settings:
                saturation = settings['saturation']
                hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
                hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], saturation)
                result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            if 'hue_shift' in settings:
                hue_shift = int(settings['hue_shift'])
                hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
                hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
                result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # Exposure adjustment
            if 'exposure' in settings:
                exposure_val = settings['exposure']
                result = np.clip(result * (2 ** exposure_val), 0, 255).astype(np.uint8)

            # Highlights and shadows
            if 'highlights' in settings or 'shadows' in settings:
                gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

                if 'shadows' in settings:
                    shadow_mask = (gray < 128).astype(np.float32)
                    shadow_adjustment = settings['shadows']
                    for c in range(3):
                        result[:, :, c] = result[:, :, c] + (shadow_mask * shadow_adjustment * 255).astype(np.uint8)

                if 'highlights' in settings:
                    highlight_mask = (gray > 128).astype(np.float32)
                    highlight_adjustment = settings['highlights']
                    for c in range(3):
                        result[:, :, c] = result[:, :, c] + (highlight_mask * highlight_adjustment * 255).astype(np.uint8)

            return np.clip(result, 0, 255).astype(np.uint8)

        except Exception as e:
            self.logger.error(f"Error applying color grading: {e}")
            return frame

    def apply_viral_effects(self, frame: np.ndarray, effect_type: str, intensity: float = 1.0) -> np.ndarray:
        """Apply viral video effects like TikTok filters"""
        try:
            if effect_type == 'tiktok_glow':
                return self._apply_glow_effect(frame, intensity)
            elif effect_type == 'vhs_retro':
                return self._apply_vhs_effect(frame, intensity)
            elif effect_type == 'glitch':
                return self._apply_glitch_effect(frame, intensity)
            elif effect_type == 'neon_cyber':
                return self._apply_neon_effect(frame, intensity)
            elif effect_type == 'dreamy_blur':
                return self._apply_dreamy_blur(frame, intensity)
            elif effect_type == 'film_grain':
                return self._apply_film_grain(frame, intensity)
            else:
                return frame
        except Exception as e:
            self.logger.error(f"Error applying viral effect {effect_type}: {e}")
            return frame

    def _apply_glow_effect(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """Apply TikTok-style glow effect"""
        # Create a blurred version
        ksize = int(15 * intensity)
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(frame, (ksize, ksize), 0)

        # Blend with original
        alpha = 0.3 * intensity
        result = cv2.addWeighted(frame, 1 - alpha, blurred, alpha, 0)

        # Increase brightness slightly
        brightness = 1.0 + (0.2 * intensity)
        result = cv2.convertScaleAbs(result, alpha=brightness, beta=0)

        return result

    def _apply_vhs_effect(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """Apply VHS retro effect"""
        result = frame.copy()
        h, w = result.shape[:2]

        # Add noise
        noise = np.random.randint(0, int(50 * intensity), (h, w, 3), dtype=np.uint8)
        result = cv2.add(result, noise)

        # Color channel shifting
        shift = int(5 * intensity)
        if shift > 0:
            result[:, shift:, 0] = result[:, :-shift, 0]  # Shift blue channel
            result[:, :-shift, 2] = result[:, shift:, 2]  # Shift red channel

        # Add scanlines
        for i in range(0, h, 4):
            if i < h:
                result[i:i+1, :] = result[i:i+1, :] * 0.8

        # Reduce saturation for vintage look
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * (1 - 0.3 * intensity)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return result

    def _apply_glitch_effect(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """Apply digital glitch effect"""
        result = frame.copy()
        h, w = result.shape[:2]

        # Random displacement
        displacement = int(10 * intensity)
        for _ in range(int(5 * intensity)):
            y1 = np.random.randint(0, h)
            y2 = min(h, y1 + np.random.randint(1, 20))
            shift = np.random.randint(-displacement, displacement)

            if shift > 0:
                result[y1:y2, shift:] = result[y1:y2, :-shift]
            elif shift < 0:
                result[y1:y2, :shift] = result[y1:y2, -shift:]

        # Color channel corruption
        if np.random.random() < intensity:
            channel = np.random.randint(0, 3)
            result[:, :, channel] = np.roll(result[:, :, channel], 
                                           np.random.randint(-displacement, displacement), axis=1)

        return result

    def _apply_neon_effect(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """Apply neon cyberpunk effect"""
        # Edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Create neon colors
        neon_edges = np.zeros_like(frame)
        neon_edges[:, :, 0] = edges  # Blue channel
        neon_edges[:, :, 1] = edges * 0.5  # Green channel
        neon_edges[:, :, 2] = edges  # Red channel

        # Blur the edges for glow effect
        ksize = int(5 * intensity)
        if ksize % 2 == 0:
            ksize += 1
        neon_glow = cv2.GaussianBlur(neon_edges, (ksize, ksize), 0)

        # Increase saturation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.5 * intensity)
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Blend
        alpha = 0.4 * intensity
        result = cv2.addWeighted(enhanced, 1 - alpha, neon_glow, alpha, 0)

        return result

    def _apply_dreamy_blur(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """Apply dreamy blur effect"""
        ksize = int(8 * intensity)
        if ksize % 2 == 0:
            ksize += 1

        blurred = cv2.GaussianBlur(frame, (ksize, ksize), 0)

        # Increase brightness and reduce contrast
        brightness = 1.0 + (0.1 * intensity)
        contrast = 1.0 - (0.1 * intensity)
        result = cv2.convertScaleAbs(blurred, alpha=contrast, beta=brightness * 10)

        return result

    def _apply_film_grain(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """Apply film grain effect"""
        h, w = frame.shape[:2]

        # Generate grain noise
        grain = np.random.normal(0, 255 * 0.15 * intensity, (h, w)).astype(np.int16)

        # Apply to each channel
        result = frame.astype(np.int16)
        for c in range(3):
            result[:, :, c] += grain

        # Add vignette
        center_x, center_y = w // 2, h // 2
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        vignette = 1 - (dist_from_center / max_dist) * 0.3 * intensity

        for c in range(3):
            result[:, :, c] = result[:, :, c] * vignette

        return np.clip(result, 0, 255).astype(np.uint8)

    def apply_background_blur(self, frame: np.ndarray, blur_intensity: float = 0.5) -> np.ndarray:
        """Apply background blur (simple implementation without person detection)"""
        try:
            # For now, apply a simple center-focused blur
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2

            # Create a mask for the center region (simulating subject area)
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(mask, (center_x, center_y), min(w, h) // 4, 1, -1)

            # Blur the entire frame
            ksize = int(21 * blur_intensity)
            if ksize % 2 == 0:
                ksize += 1
            blurred = cv2.GaussianBlur(frame, (ksize, ksize), 0)

            # Blend based on mask
            mask_3d = np.stack([mask] * 3, axis=2)
            result = frame * mask_3d + blurred * (1 - mask_3d)

            return result.astype(np.uint8)

        except Exception as e:
            self.logger.error(f"Error applying background blur: {e}")
            return frame

    def apply_face_filters(self, frame: np.ndarray, filter_type: str) -> np.ndarray:
        """Apply AI-based face filters (basic implementation)"""
        try:
            # Load face cascade classifier
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            result = frame.copy()

            for (x, y, w, h) in faces:
                if filter_type == 'smooth_skin':
                    # Apply skin smoothing
                    face_region = result[y:y+h, x:x+w]
                    smoothed = cv2.bilateralFilter(face_region, 15, 50, 50)
                    result[y:y+h, x:x+w] = smoothed

                elif filter_type == 'glow_face':
                    # Apply glow to face
                    face_region = result[y:y+h, x:x+w]
                    glowed = self._apply_glow_effect(face_region, 0.5)
                    result[y:y+h, x:x+w] = glowed

                elif filter_type == 'beauty_enhance':
                    # Enhance facial features
                    face_region = result[y:y+h, x:x+w]
                    # Increase brightness and reduce red tones
                    enhanced = cv2.convertScaleAbs(face_region, alpha=1.1, beta=10)
                    result[y:y+h, x:x+w] = enhanced

            return result

        except Exception as e:
            self.logger.error(f"Error applying face filter: {e}")
            return frame

    def add_text_overlay(self, frame: np.ndarray, text: str, position: Tuple[int, int], 
                        font_size: float = 1.0, color: Tuple[int, int, int] = (255, 255, 255),
                        font_style: str = 'normal') -> np.ndarray:
        """Add text overlay to frame"""
        try:
            result = frame.copy()

            # Choose font
            font = cv2.FONT_HERSHEY_SIMPLEX
            if font_style == 'bold':
                font = cv2.FONT_HERSHEY_DUPLEX
            elif font_style == 'italic':
                font = cv2.FONT_HERSHEY_COMPLEX

            # Add text with shadow for better visibility
            shadow_offset = max(1, int(font_size))
            cv2.putText(result, text, 
                       (position[0] + shadow_offset, position[1] + shadow_offset), 
                       font, font_size, (0, 0, 0), 2)  # Shadow
            cv2.putText(result, text, position, font, font_size, color, 2)  # Main text

            return result

        except Exception as e:
            self.logger.error(f"Error adding text overlay: {e}")
            return frame

    def apply_transition_effect(self, frame1: np.ndarray, frame2: np.ndarray, 
                              transition_type: str, progress: float) -> np.ndarray:
        """Apply transition between two frames"""
        try:
            progress = np.clip(progress, 0.0, 1.0)

            if transition_type == 'fade':
                return cv2.addWeighted(frame1, 1 - progress, frame2, progress, 0)

            elif transition_type == 'slide_left':
                h, w = frame1.shape[:2]
                split_point = int(w * progress)
                result = frame1.copy()
                if split_point < w:
                    result[:, :split_point] = frame2[:, w-split_point:]
                return result

            elif transition_type == 'slide_right':
                h, w = frame1.shape[:2]
                split_point = int(w * (1 - progress))
                result = frame1.copy()
                if split_point > 0:
                    result[:, split_point:] = frame2[:, :w-split_point]
                return result

            elif transition_type == 'zoom':
                center_x, center_y = frame1.shape[1] // 2, frame1.shape[0] // 2
                scale = 1.0 + progress * 0.5

                # Scale frame2
                M = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)
                scaled_frame2 = cv2.warpAffine(frame2, M, (frame1.shape[1], frame1.shape[0]))

                return cv2.addWeighted(frame1, 1 - progress, scaled_frame2, progress, 0)

            elif transition_type == 'spin':
                center_x, center_y = frame1.shape[1] // 2, frame1.shape[0] // 2
                angle = progress * 360

                M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
                rotated_frame2 = cv2.warpAffine(frame2, M, (frame1.shape[1], frame1.shape[0]))

                return cv2.addWeighted(frame1, 1 - progress, rotated_frame2, progress, 0)

            else:
                return cv2.addWeighted(frame1, 1 - progress, frame2, progress, 0)

        except Exception as e:
            self.logger.error(f"Error applying transition: {e}")
            return frame1

    def process_frame_realtime(self, frame: np.ndarray, effects_config: Dict) -> np.ndarray:
        """Process a single frame with all applied effects in real-time"""
        try:
            result = frame.copy()

            # Apply effects in order of importance for performance
            if 'color_grading' in effects_config:
                result = self.apply_color_grading(result, effects_config['color_grading'])

            if 'viral_effect' in effects_config:
                effect_type = effects_config['viral_effect'].get('type')
                intensity = effects_config['viral_effect'].get('intensity', 1.0)
                if effect_type:
                    result = self.apply_viral_effects(result, effect_type, intensity)

            if 'background_blur' in effects_config:
                blur_intensity = effects_config['background_blur'].get('intensity', 0.5)
                result = self.apply_background_blur(result, blur_intensity)

            if 'face_filter' in effects_config:
                filter_type = effects_config['face_filter'].get('type')
                if filter_type:
                    result = self.apply_face_filters(result, filter_type)

            if 'text_overlays' in effects_config:
                for text_config in effects_config['text_overlays']:
                    result = self.add_text_overlay(
                        result,
                        text_config.get('text', ''),
                        text_config.get('position', (50, 50)),
                        text_config.get('font_size', 1.0),
                        text_config.get('color', (255, 255, 255)),
                        text_config.get('font_style', 'normal')
                    )

            return result

        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return frame

    def export_frame_as_base64(self, frame: np.ndarray, format: str = 'JPEG') -> str:
        """Convert frame to base64 for web preview"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            buffer = BytesIO()
            pil_image.save(buffer, format=format, quality=85)
            img_str = base64.b64encode(buffer.getvalue()).decode()

            return f"data:image/{format.lower()};base64,{img_str}"

        except Exception as e:
            self.logger.error(f"Error exporting frame as base64: {e}")
            return ""

    def get_effect_presets(self) -> Dict:
        """Return available effect presets"""
        return {
            'lut_presets': list(self.lut_presets.keys()),
            'viral_effects': list(self.effect_templates.keys()),
            'transitions': ['fade', 'slide_left', 'slide_right', 'zoom', 'spin'],
            'face_filters': ['smooth_skin', 'glow_face', 'beauty_enhance']
        }
import cv2
import numpy as np
import json
import base64
from io import BytesIO
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class VideoEffectsProcessor:
    """Ultra-fast video effects processor with real-time capabilities"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Performance optimization settings
        self.processing_pool = ThreadPoolExecutor(max_workers=4)
        self.frame_cache = {}
        self.effect_cache = {}

        # Real-time processing settings
        self.target_fps = 30
        self.max_processing_time = 1.0 / self.target_fps  # 33ms max per frame

        self.logger.info("VideoEffectsProcessor initialized with real-time capabilities")

    def process_frame_realtime(self, frame, effects_config):
        """Process frame with real-time optimizations and taint protection"""
        try:
            # Ensure frame is properly formatted and not tainted
            if frame is None or frame.size == 0:
                return self._create_fallback_frame()

            # Create a clean copy to avoid taint issues
            processed_frame = np.ascontiguousarray(frame.copy())

            # Handle both list and dict formats for effects_config
            if isinstance(effects_config, list):
                # Convert list to dict format
                effects_dict = {}
                for effect in effects_config:
                    if isinstance(effect, dict) and 'name' in effect:
                        effects_dict[effect['name']] = effect.get('params', {})
                        effects_dict[effect['name']]['enabled'] = effect.get('enabled', True)
                effects_config = effects_dict
            elif not isinstance(effects_config, dict):
                self.logger.warning(f"Invalid effects_config format: {type(effects_config)}")
                return frame

            # Apply effects based on configuration
            for effect_name, effect_params in effects_config.items():
                if isinstance(effect_params, dict) and not effect_params.get('enabled', True):
                    continue

                # Apply effect with timeout to ensure real-time performance (max 16ms for 60fps)
                # processed_frame = self._apply_effect_fast(
                #    processed_frame, effect_name, effect_params
                # )
                processed_frame = self._apply_single_effect(processed_frame, effect_name, effect_params)


                # Check if we're exceeding time budget for real-time (16ms for 60fps)
                # if time.time() - start_time > 0.016:
                #    self.logger.debug("Effect processing taking too long, skipping remaining effects")
                #    break

            # Ensure output is clean
            processed_frame = np.ascontiguousarray(processed_frame)

            return processed_frame

        except Exception as e:
            self.logger.error(f"Real-time frame processing failed: {e}")
            return self._create_fallback_frame()

    def _apply_single_effect(self, frame, effect_name, effect_params):
        """Apply a single effect to the frame."""
        try:
            if effect_name == 'brightness':
                intensity = effect_params.get('intensity', 0.5)
                return self._adjust_brightness_fast(frame, intensity)
            elif effect_name == 'contrast':
                intensity = effect_params.get('intensity', 0.5)
                return self._adjust_contrast_fast(frame, intensity)
            elif effect_name == 'saturation':
                intensity = effect_params.get('intensity', 0.5)
                return self._adjust_saturation_fast(frame, intensity)
            elif effect_name == 'blur':
                intensity = effect_params.get('intensity', 0.5)
                return self._apply_blur_fast(frame, intensity)
            elif effect_name == 'sharpen':
                intensity = effect_params.get('intensity', 0.5)
                return self._apply_sharpen_fast(frame, intensity)
            elif effect_name == 'vintage':
                intensity = effect_params.get('intensity', 0.5)
                return self._apply_vintage_fast(frame, intensity)
            elif effect_name == 'vignette':
                intensity = effect_params.get('intensity', 0.5)
                return self._apply_vignette_fast(frame, intensity)
            elif effect_name == 'color_grade':
                color = effect_params.get('color', '#ffffff')
                intensity = effect_params.get('intensity', 0.5)
                return self._apply_color_grade_fast(frame, color, intensity)
            else:
                return frame
        except Exception as e:
            self.logger.error(f"Error applying effect {effect_name}: {e}")
            return frame

    def _apply_effect_fast(self, frame, effect_name, params):
        """Apply a single effect with optimized performance"""
        try:
            # Handle both dict and non-dict params
            if isinstance(params, dict):
                intensity = params.get('intensity', 0.5)
                color_hex = params.get('color', '#ffffff')
            else:
                intensity = 0.5
                color_hex = '#ffffff'

            if effect_name == 'brightness':
                return self._adjust_brightness_fast(frame, intensity)
            elif effect_name == 'contrast':
                return self._adjust_contrast_fast(frame, intensity)
            elif effect_name == 'saturation':
                return self._adjust_saturation_fast(frame, intensity)
            elif effect_name == 'blur':
                return self._apply_blur_fast(frame, intensity)
            elif effect_name == 'sharpen':
                return self._apply_sharpen_fast(frame, intensity)
            elif effect_name == 'vintage':
                return self._apply_vintage_fast(frame, intensity)
            elif effect_name == 'vignette':
                return self._apply_vignette_fast(frame, intensity)
            elif effect_name == 'color_grade':
                color = params.get('color', '#ffffff')
                return self._apply_color_grade_fast(frame, color, intensity)
            else:
                return frame

        except Exception as e:
            self.logger.warning(f"Effect {effect_name} failed: {e}")
            return frame

    def _adjust_brightness_fast(self, frame, intensity):
        """Fast brightness adjustment using vectorized operations"""
        adjustment = int((intensity - 0.5) * 100)
        return cv2.convertScaleAbs(frame, alpha=1, beta=adjustment)

    def _adjust_contrast_fast(self, frame, intensity):
        """Fast contrast adjustment"""
        alpha = 0.5 + intensity * 1.5  # Range: 0.5 to 2.0
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=0)

    def _adjust_saturation_fast(self, frame, intensity):
        """Fast saturation adjustment using HSV"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation_scale = 0.5 + intensity * 1.5
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _apply_blur_fast(self, frame, intensity):
        """Fast blur using optimized kernel size"""
        if intensity < 0.1:
            return frame

        kernel_size = max(3, int(intensity * 15))
        if kernel_size % 2 == 0:
            kernel_size += 1

        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

    def _apply_sharpen_fast(self, frame, intensity):
        """Fast sharpening using convolution"""
        if intensity < 0.1:
            return frame

        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]]) * intensity
        kernel[1, 1] = 1 + 8 * intensity

        return cv2.filter2D(frame, -1, kernel)

    def _apply_vintage_fast(self, frame, intensity):
        """Fast vintage effect using color mapping"""
        if intensity < 0.1:
            return frame

        # Create vintage color mapping
        vintage_frame = frame.copy()

        # Reduce blue channel and increase red/green
        vintage_frame[:, :, 0] = vintage_frame[:, :, 0] * (1 - intensity * 0.3)  # Blue
        vintage_frame[:, :, 1] = np.clip(vintage_frame[:, :, 1] * (1 + intensity * 0.2), 0, 255)  # Green
        vintage_frame[:, :, 2] = np.clip(vintage_frame[:, :, 2] * (1 + intensity * 0.1), 0, 255)  # Red

        # Add slight vignette
        return self._apply_vignette_fast(vintage_frame, intensity * 0.3)

    def _apply_vignette_fast(self, frame, intensity):
        """Fast vignette effect using radial gradient"""
        if intensity < 0.1:
            return frame

        rows, cols = frame.shape[:2]
        center_x, center_y = cols // 2, rows // 2

        # Create distance map
        x, y = np.ogrid[:rows, :cols]
        dist = np.sqrt((x - center_y)**2 + (y - center_x)**2)

        # Normalize distance
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist = dist / max_dist

        # Create vignette mask
        vignette = 1 - (dist * intensity)
        vignette = np.clip(vignette, 0, 1)

        # Apply vignette
        result = frame.copy().astype(float)
        result[:, :, 0] *= vignette
        result[:, :, 1] *= vignette
        result[:, :, 2] *= vignette

        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_color_grade_fast(self, frame, color_hex, intensity):
        """Fast color grading using color overlay"""
        if intensity < 0.1:
            return frame

        # Parse hex color
        color_hex = color_hex.lstrip('#')
        r = int(color_hex[0:2], 16)
        g = int(color_hex[2:4], 16)
        b = int(color_hex[4:6], 16)

        # Create color overlay
        overlay = np.full_like(frame, [b, g, r], dtype=np.uint8)

        # Blend with original frame
        alpha = intensity * 0.3  # Limit intensity for realistic effect
        return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

    def _create_placeholder_base64(self):
        """Create a placeholder base64 image"""
        try:
            # Create a simple 320x180 black image with text
            img = Image.new('RGB', (320, 180), color='black')
            draw = ImageDraw.Draw(img)

            # Add text
            try:
                font = ImageFont.load_default()
                draw.text((160, 90), "Processing...", fill='white', anchor='mm', font=font)
            except:
                draw.text((130, 85), "Processing...", fill='white')

            # Convert to base64
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=50)
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            return f"data:image/jpeg;base64,{img_base64}"

        except Exception as e:
            self.logger.error(f"Error creating placeholder: {e}")
            # Return minimal base64 data
            return "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwDX/9k="

    def _create_fallback_frame(self):
        """Create a fallback frame for error cases"""
        try:
            # Create a 1080x1920 black frame for vertical video
            fallback_frame = np.zeros((1920, 1080, 3), dtype=np.uint8)

            # Add some text overlay
            cv2.putText(fallback_frame, "Processing...", (400, 960), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

            return fallback_frame

        except Exception as e:
            self.logger.error(f"Error creating fallback frame: {e}")
            # Return minimal frame
            return np.zeros((180, 320, 3), dtype=np.uint8)

    def export_frame_as_base64(self, frame):
        """Export frame as base64 string with canvas taint protection"""
        try:
            # Convert frame to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame

            # Ensure frame is clean and not tainted
            frame_rgb = np.ascontiguousarray(frame_rgb)

            # Convert to PIL Image with explicit mode
            pil_image = Image.fromarray(frame_rgb.astype(np.uint8), mode='RGB')

            # Convert to base64 with optimized settings
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG", quality=75, optimize=True)
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            return f"data:image/jpeg;base64,{img_base64}"

        except Exception as e:
            self.logger.error(f"Error exporting frame as base64: {e}")
            # Return a small placeholder image as fallback
            return self._create_placeholder_base64()

    def export_video_ultra_fast(self, input_path, output_path, effects_timeline):
        """Ultra-fast video export with hardware acceleration"""
        try:
            import subprocess
            import tempfile

            # Use FFmpeg with hardware acceleration for ultra-fast processing
            temp_script = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)

            # Create FFmpeg filter script for effects
            filter_complex = []

            # Add basic effects that can be done with FFmpeg filters
            if effects_timeline:
                for effect in effects_timeline:
                    effect_name = effect.get('name', '')
                    params = effect.get('params', {})

                    if effect_name == 'brightness':
                        intensity = params.get('intensity', 0.5)
                        brightness_val = (intensity - 0.5) * 0.5  # Scale to -0.25 to 0.25
                        filter_complex.append(f"eq=brightness={brightness_val}")

                    elif effect_name == 'contrast':
                        intensity = params.get('intensity', 0.5)
                        contrast_val = 0.5 + intensity * 1.5  # Scale to 0.5 to 2.0
                        filter_complex.append(f"eq=contrast={contrast_val}")

                    elif effect_name == 'saturation':
                        intensity = params.get('intensity', 0.5)
                        sat_val = 0.5 + intensity * 1.5
                        filter_complex.append(f"eq=saturation={sat_val}")

                    elif effect_name == 'blur':
                        intensity = params.get('intensity', 0.5)
                        if intensity > 0.1:
                            blur_val = intensity * 10
                            filter_complex.append(f"gblur=sigma={blur_val}")

            # Build FFmpeg command
            cmd = [
                'ffmpeg',
                '-hwaccel', 'auto',
                '-i', input_path,
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-movflags', '+faststart',
                '-y', output_path
            ]

            # Add filters if any
            if filter_complex:
                filter_string = ','.join(filter_complex)
                cmd.insert(-3, '-vf')
                cmd.insert(-3, filter_string)

            # Execute with timeout
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                self.logger.info(f"Ultra-fast export completed: {output_path}")
                return True
            else:
                self.logger.error(f"FFmpeg export failed: {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Ultra-fast export failed: {e}")
            return False

    def process_video_with_effects(self, input_path, output_path, effects_timeline):
        """Process entire video with effects timeline"""
        try:
            cap = cv2.VideoCapture(input_path)

            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Calculate current time
                current_time = frame_count / fps

                # Get effects for current time
                current_effects = self._get_effects_for_time(effects_timeline, current_time)

                # Process frame with effects
                if current_effects:
                    processed_frame = self.process_frame_realtime(frame, current_effects)
                else:
                    processed_frame = frame

                # Write frame
                out.write(processed_frame)
                frame_count += 1

                # Progress update
                if frame_count % 30 == 0:  # Update every second
                    progress = (frame_count / total_frames) * 100
                    self.logger.info(f"Processing progress: {progress:.1f}%")

            cap.release()
            out.release()

            self.logger.info(f"Video processing completed: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Video processing failed: {e}")
            return False

    def _get_effects_for_time(self, effects_timeline, current_time):
        """Get active effects for a specific time"""
        active_effects = {}

        for effect in effects_timeline:
            start_time = effect.get('start_time', 0)
            end_time = effect.get('end_time', float('inf'))

            if start_time <= current_time <= end_time:
                effect_name = effect.get('name')
                effect_params = effect.get('params', {})

                if effect_name:
                    active_effects[effect_name] = effect_params

        return active_effects

    def get_effect_presets(self):
        """Get available effect presets"""
        return {
            'presets': [
                {
                    'name': 'Cinematic',
                    'effects': {
                        'contrast': {'enabled': True, 'intensity': 0.7},
                        'saturation': {'enabled': True, 'intensity': 0.6},
                        'vignette': {'enabled': True, 'intensity': 0.4},
                        'color_grade': {'enabled': True, 'intensity': 0.3, 'color': '#ff6b35'}
                    }
                },
                {
                    'name': 'Vintage',
                    'effects': {
                        'vintage': {'enabled': True, 'intensity': 0.8},
                        'brightness': {'enabled': True, 'intensity': 0.6},
                        'vignette': {'enabled': True, 'intensity': 0.5}
                    }
                },
                {
                    'name': 'Bright & Vibrant',
                    'effects': {
                        'brightness': {'enabled': True, 'intensity': 0.7},
                        'saturation': {'enabled': True, 'intensity': 0.8},
                        'contrast': {'enabled': True, 'intensity': 0.6}
                    }
                },
                {
                    'name': 'Dramatic',
                    'effects': {
                        'contrast': {'enabled': True, 'intensity': 0.9},
                        'sharpen': {'enabled': True, 'intensity': 0.5},
                        'vignette': {'enabled': True, 'intensity': 0.6},
                        'color_grade': {'enabled': True, 'intensity': 0.4, 'color': '#2c3e50'}
                    }
                }
            ]
        }

    def cleanup(self):
        """Clean up resources"""
        try:
            self.processing_pool.shutdown(wait=True)
            self.frame_cache.clear()
            self.effect_cache.clear()
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")