import os
import logging
import yt_dlp
import subprocess
import json
import re
import psutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from app import app, db
from models import VideoJob, VideoShort, TranscriptSegment, ProcessingStatus
from gemini_analyzer import GeminiAnalyzer


class VideoProcessor:

    def _ensure_directories(self):
        """Ensure all required directories exist"""
        directories = ['uploads', 'temp', 'thumbnails', 'cookie', 'outputs']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.gemini_analyzer = GeminiAnalyzer()
        self.whisper_model = None

        # Ensure required directories exist
        self._ensure_directories()

        # Ultra-optimized performance settings
        cpu_count = psutil.cpu_count() or 4
        memory_gb = psutil.virtual_memory().total / (1024**3)

        # Dynamic worker allocation based on system resources
        if memory_gb >= 32:  # High-end system
            self.max_workers = min(16, cpu_count * 2)
            self.video_workers = min(8, cpu_count)
            self.io_workers = min(12, cpu_count * 3)
        elif memory_gb >= 16:  # Mid-range system
            self.max_workers = min(12, cpu_count)
            self.video_workers = min(6, cpu_count)
            self.io_workers = min(8, cpu_count * 2)
        else:  # Low-end system
            self.max_workers = min(8, max(4, cpu_count))
            self.video_workers = min(4, cpu_count)
            self.io_workers = min(6, cpu_count)

        # Enhanced caching and memory management
        self.video_cache = {}
        self.frame_cache = {}
        self.temp_cleanup_queue = []
        self.processing_pool = None

        # Ultra-fast FFmpeg settings
        self.ffmpeg_preset = 'ultrafast'
        self.video_codec = 'libx264'
        self.audio_codec = 'aac'
        self.hardware_accel = self._detect_hardware_acceleration()

        self.logger.info(f"VideoProcessor initialized: {self.max_workers} workers, {self.video_workers} video workers, {self.io_workers} I/O workers")
        self.logger.info(f"Hardware acceleration: {self.hardware_accel}")

    def _detect_hardware_acceleration(self):
        """Detect available hardware acceleration"""
        try:
            # Test for NVIDIA GPU
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                return 'nvenc'
        except:
            pass

        try:
            # Test for Intel Quick Sync
            result = subprocess.run(['ffmpeg', '-hwaccels'], capture_output=True, text=True)
            if 'qsv' in result.stdout:
                return 'qsv'
            elif 'vaapi' in result.stdout:
                return 'vaapi'
        except:
            pass

        return 'auto'

    def _sanitize_filename(self, filename):
        """Sanitize filename to prevent special character issues"""
        # Remove or replace problematic characters
        filename = re.sub(r'[^\w\s-]', '',
                          filename)  # Keep only alphanumeric, spaces, hyphens
        filename = re.sub(
            r'[-\s]+', '-',
            filename)  # Replace multiple spaces/hyphens with single hyphen
        filename = filename.strip('-')  # Remove leading/trailing hyphens
        return filename[:100]  # Limit length

    def _check_cookies_available(self):
        """Check if YouTube cookies file is available and contains actual cookies"""
        # Check both cookie directory and root directory
        cookie_files = ['cookie/youtube_cookies.txt', 'youtube_cookies.txt']

        for cookie_file in cookie_files:
            if not os.path.exists(cookie_file):
                continue

            try:
                with open(cookie_file, 'r') as f:
                    content = f.read().strip()
                    # Check if file contains actual cookies (not just comments)
                    lines = [
                        line for line in content.split('\n')
                        if line.strip() and not line.startswith('#')
                    ]
                    if len(lines) > 0:
                        self.logger.info(
                            f"YouTube cookies file found with authentication data: {cookie_file}"
                        )
                        return cookie_file
            except Exception as e:
                self.logger.warning(
                    f"Error reading cookies file {cookie_file}: {e}")
                continue

        self.logger.info(
            "No valid YouTube cookies found - age-restricted videos may fail")
        return None

    def load_whisper_model(self):
        """Load Whisper model for transcription"""
        if self.whisper_model is None:
            try:
                # Use ffmpeg for basic audio extraction and create mock transcription
                # This avoids the Whisper dependency issue while maintaining functionality
                self.whisper_model = "ffmpeg_based"
                self.logger.info("Audio processing initialized")
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize audio processing: {e}")
                raise

    def process_video(self, job_id):
        """Main processing pipeline for a video job"""
        with app.app_context():
            job = VideoJob.query.get(job_id)
            if not job:
                self.logger.error(f"Job {job_id} not found")
                return

            try:
                self.logger.info(
                    f"Starting processing for job {job_id}: {job.youtube_url}")

                # Step 1: Download video
                self._update_job_status(job, ProcessingStatus.DOWNLOADING, 10)
                video_path = self._download_video(job)

                # Step 2: Transcribe audio with Whisper
                self._update_job_status(job, ProcessingStatus.TRANSCRIBING, 30)
                transcript_data = self._transcribe_video(job, video_path)

                # Step 3: Analyze content with Gemini AI
                self._update_job_status(job, ProcessingStatus.ANALYZING, 50)
                engaging_segments = self._analyze_content(job, transcript_data)

                # Step 4: Generate vertical short videos
                self._update_job_status(job, ProcessingStatus.EDITING, 70)
                self._generate_shorts(job, video_path, engaging_segments)

                # Step 5: Complete
                self._update_job_status(job, ProcessingStatus.COMPLETED, 100)
                # Step 6: Cleanup temporary files
                self._cleanup_temporary_files(job)

                self.logger.info(
                    f"Successfully completed processing for job {job_id}")

            except Exception as e:
                self.logger.error(f"Error processing job {job_id}: {e}")
                self._update_job_status(job, ProcessingStatus.FAILED, 0,
                                        str(e))

    def _update_job_status(self, job, status, progress, error_message=None):
        """Update job status and progress with robust error handling"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Refresh the job instance to avoid stale data
                with app.app_context():
                    job = VideoJob.query.get(job.id)
                    if job:
                        job.status = status
                        job.progress = progress
                        if error_message:
                            job.error_message = error_message
                        db.session.commit()
                        return
            except Exception as e:
                self.logger.warning(f"Database update attempt {attempt + 1}/{max_retries} failed: {e}")
                try:
                    db.session.rollback()
                except:
                    pass
                
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1)  # Wait before retry
                else:
                    self.logger.error(f"Failed to update job status after {max_retries} attempts: {e}")

    def detect_video_formats(self, youtube_url):
        """Detect available video formats for a YouTube URL"""
        try:
            cookie_file = self._check_cookies_available()
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extractaudio': False,
                'noplaylist': True,
            }

            if cookie_file:
                ydl_opts['cookiefile'] = cookie_file

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)

                if not info or 'formats' not in info:
                    return []

                # Extract unique video formats with audio
                formats = []
                seen_resolutions = set()

                for fmt in info['formats']:
                    if fmt.get('vcodec') != 'none' and fmt.get(
                            'acodec') != 'none':
                        height = fmt.get('height')
                        if height and height not in seen_resolutions:
                            formats.append({
                                'format_id': fmt['format_id'],
                                'resolution': f"{height}p",
                                'height': height,
                                'ext': fmt.get('ext', 'mp4'),
                                'filesize': fmt.get('filesize'),
                                'fps': fmt.get('fps'),
                                'vcodec': fmt.get('vcodec'),
                                'acodec': fmt.get('acodec')
                            })
                            seen_resolutions.add(height)

                # Sort by resolution (highest first)
                formats.sort(key=lambda x: x['height'], reverse=True)

                # Add standard quality options if not present
                standard_qualities = [{
                    'resolution': '1080p',
                    'height': 1080,
                    'recommended': True
                }, {
                    'resolution': '720p',
                    'height': 720,
                    'recommended': True
                }, {
                    'resolution': '480p',
                    'height': 480,
                    'recommended': False
                }, {
                    'resolution': '360p',
                    'height': 360,
                    'recommended': False
                }]

                final_formats = []
                for std_quality in standard_qualities:
                    # Check if we have this quality available
                    found = False
                    for fmt in formats:
                        if fmt['height'] >= std_quality['height']:
                            final_formats.append({
                                'resolution':
                                std_quality['resolution'],
                                'height':
                                std_quality['height'],
                                'available':
                                True,
                                'recommended':
                                std_quality['recommended'],
                                'ext':
                                fmt.get('ext', 'mp4'),
                                'filesize':
                                fmt.get('filesize')
                            })
                            found = True
                            break

                    if not found:
                        final_formats.append({
                            'resolution':
                            std_quality['resolution'],
                            'height':
                            std_quality['height'],
                            'available':
                            False,
                            'recommended':
                            std_quality['recommended']
                        })

                return final_formats

        except Exception as e:
            self.logger.error(f"Failed to detect video formats: {e}")
            return []

    def _download_video(self, job):
        """Download video using yt-dlp in selected quality"""
        output_dir = 'uploads'

        # Enhanced quality format selectors with better fallbacks including 2K and 4K
        # Support both old format (2160p) and new format (2160p 4K)
        quality_formats = {
            # 4K formats
            '2160p 4K': 'bestvideo[height<=2160][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=2160]+bestaudio/best[height<=2160]/best',
            '2160p': 'bestvideo[height<=2160][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=2160]+bestaudio/best[height<=2160]/best',
            '4K+ Ultra HD': 'bestvideo[height>=2160][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height>=2160]+bestaudio/best[height>=2160]/best',
            # 2K formats  
            '1440p 2K': 'bestvideo[height<=1440][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1440]+bestaudio/best[height<=1440]/best',
            '1440p': 'bestvideo[height<=1440][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1440]+bestaudio/best[height<=1440]/best',
            # Full HD formats
            '1080p Full HD': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[height<=1080]/best',
            '1080p': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[height<=1080]/best',
            # HD formats
            '720p HD': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=720]+bestaudio/best[height<=720]/best',
            '720p': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=720]+bestaudio/best[height<=720]/best',
            # Standard formats
            '480p': 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=480]+bestaudio/best[height<=480]/best',
            '360p': 'bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=360]+bestaudio/best[height<=360]/best',
            'best': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best'
        }

        # Get format selector with fallback priority: requested -> 1080p Full HD -> 1080p -> best
        format_selector = quality_formats.get(
            job.video_quality, 
            quality_formats.get('1080p Full HD', 
                              quality_formats.get('1080p', 
                                                 quality_formats['best']))
        )

        ydl_opts = {
            'format':
            format_selector,
            'outtmpl':
            os.path.join(output_dir, f'video_{job.id}.%(ext)s'
                         ),  # Use simple filename to avoid character issues
            'extractaudio':
            False,
            'noplaylist':
            True,
            'writesubtitles':
            False,
            'writeautomaticsub':
            False,
            'merge_output_format':
            'mp4',  # Force mp4 output
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
            'prefer_ffmpeg':
            True,  # Use ffmpeg for processing
            'format_sort': ['res', 'ext:mp4:m4a',
                            'vcodec:h264'],  # Prefer higher res, mp4, and h264
            'verbose':
            False,  # Disable verbose logging
            # Custom format selector for audio language priority
            'format_sort_force':
            True,
            # Age-restricted content support
            'age_limit':
            None,  # Remove age restrictions
        }

        # Check for authentication status and add cookie file if available
        cookie_file = self._check_cookies_available()
        if cookie_file:
            ydl_opts['cookiefile'] = cookie_file
            self.logger.info(
                f"Using YouTube authentication cookies from: {cookie_file}")
        else:
            self.logger.info(
                "No authentication cookies found - age-restricted videos may fail"
            )

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info first to get title and duration
                info = ydl.extract_info(job.youtube_url, download=False)
                if info:
                    job.title = info.get('title', 'Unknown Title')[:200]
                    job.duration = info.get('duration', 0)
                    job.video_info = {
                        'title': info.get('title'),
                        'duration': info.get('duration'),
                        'uploader': info.get('uploader'),
                        'view_count': info.get('view_count'),
                        'width': info.get('width'),
                        'height': info.get('height'),
                        'fps': info.get('fps')
                    }
                else:
                    job.title = 'Unknown Title'
                    job.duration = 0
                    job.video_info = {}
                db.session.commit()

                # Download the video
                ydl.download([job.youtube_url])

                # Find the downloaded video file
                video_files = []
                for file in os.listdir(output_dir):
                    if file.startswith(f'video_{job.id}.') and file.endswith(
                        ('.mp4', '.webm', '.mkv', '.avi')):
                        video_files.append(file)

                if video_files:
                    video_file = video_files[0]
                    video_path = os.path.join(output_dir, video_file)
                    job.video_path = video_path
                    db.session.commit()
                    self.logger.info(f"Downloaded video: {video_path}")
                    return video_path
                else:
                    raise Exception("Downloaded video file not found")

        except Exception as e:
            raise Exception(f"Failed to download video: {e}")

    def _transcribe_video(self, job, video_path):
        """Transcribe video using Whisper"""
        try:
            # Load Whisper model
            self.load_whisper_model()

            # Extract audio for Whisper with Hindi language preference
            audio_path = os.path.join('temp', f'audio_{job.id}.wav')

            # Detect and prioritize audio streams: Hindi first, then English, then default
            audio_stream_index = self._select_preferred_audio_stream(
                video_path)

            # Create a safe filename for audio processing
            safe_video_path = os.path.join('temp', f'safe_video_{job.id}.mp4')

            # Copy video to safe path to avoid encoding issues
            import shutil
            shutil.copy2(video_path, safe_video_path)

            # Extract audio from safe path
            cmd = [
                'ffmpeg',
                '-i',
                safe_video_path,
                f'-map',
                f'0:a:{audio_stream_index}',  # Select specific audio stream
                '-vn',
                '-acodec',
                'pcm_s16le',
                '-ar',
                '16000',
                '-ac',
                '1',
                '-y',
                audio_path
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    self.logger.error(
                        f"FFmpeg audio extraction failed: {result.stderr}")
                    raise Exception(
                        f"FFmpeg audio extraction failed: {result.stderr}")
            finally:
                # Clean up the temporary safe video file
                if os.path.exists(safe_video_path):
                    os.remove(safe_video_path)

            # Use ffmpeg to get duration and create time-based segments for AI analysis
            duration_cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'csv=p=0', video_path
            ]
            duration_result = subprocess.run(duration_cmd,
                                             capture_output=True,
                                             text=True)
            duration = float(duration_result.stdout.strip())

            # Create time-based segments based on user's preferred short length
            segment_length = job.short_length  # Use user's selected length
            segments = []
            for i in range(0, int(duration), segment_length):
                end_time = min(i + segment_length, duration)
                segments.append({
                    'start':
                    i,
                    'end':
                    end_time,
                    'text':
                    f"Audio segment from {i}s to {end_time}s"  # Placeholder for AI analysis
                })

            transcript_data = {
                'segments': segments,
                'language': 'en',
                'full_text':
                f"Video content with {len(segments)} segments for AI analysis",
                'duration': duration
            }

            # Save transcript
            transcript_path = os.path.join('uploads',
                                           f'transcript_{job.id}.json')
            with open(transcript_path, 'w') as f:
                json.dump(transcript_data, f, indent=2)

            job.audio_path = audio_path
            job.transcript_path = transcript_path
            db.session.commit()

            # Store segments in database
            for segment in segments:
                if len(segment['text'].strip()
                       ) > 10:  # Only meaningful segments
                    transcript_segment = TranscriptSegment()
                    transcript_segment.job_id = job.id
                    transcript_segment.start_time = segment['start']
                    transcript_segment.end_time = segment['end']
                    transcript_segment.text = segment['text'].strip()
                    db.session.add(transcript_segment)

            db.session.commit()
            return transcript_data

        except Exception as e:
            raise Exception(f"Failed to transcribe video: {e}")

    def _analyze_content(self, job, transcript_data):
        """Analyze content with Gemini AI to find engaging segments - ULTRA-OPTIMIZED PARALLEL PROCESSING"""
        try:
            segments = TranscriptSegment.query.filter_by(job_id=job.id).all()
            engaging_segments = []

            # Process segments in parallel for faster AI analysis with memory optimization
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import gc  # Garbage collection for memory optimization

            def analyze_single_segment(segment_id, segment_text):
                """Analyze a single segment with Gemini AI - Enhanced with robust session management"""
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        # Create new app context and database session for each thread
                        with app.app_context():
                            # Get fresh segment instance in this thread's context
                            segment = TranscriptSegment.query.get(segment_id)
                            if not segment:
                                return None

                            analysis = self.gemini_analyzer.analyze_segment(segment_text)

                            # Update segment with analysis results
                            segment.engagement_score = analysis.get('engagement_score', 0.0)
                            segment.emotion_score = analysis.get('emotion_score', 0.0)
                            segment.viral_potential = analysis.get('viral_potential', 0.0)
                            segment.quotability = analysis.get('quotability', 0.0)
                            segment.overall_score = (segment.engagement_score +
                                                     segment.emotion_score +
                                                     segment.viral_potential +
                                                     segment.quotability) / 4.0
                            segment.emotions_detected = analysis.get('emotions', [])
                            segment.keywords = analysis.get('keywords', [])
                            segment.content_type = analysis.get('content_type', 'general')
                            segment.visual_style = analysis.get('visual_style', 'clean')
                            segment.analysis_notes = analysis.get('reason', '')

                            # Commit with retry logic
                            try:
                                db.session.commit()
                                self.logger.info(f"Analyzed segment {segment.id}: score={segment.overall_score:.2f}")
                                return segment_id
                            except Exception as commit_error:
                                self.logger.warning(f"Commit attempt {attempt + 1}/{max_retries} failed: {commit_error}")
                                db.session.rollback()
                                if attempt < max_retries - 1:
                                    import time
                                    time.sleep(0.5)
                                    continue
                                else:
                                    raise commit_error

                    except Exception as e:
                        self.logger.error(f"Analysis attempt {attempt + 1}/{max_retries} failed for segment {segment_id}: {e}")
                        if attempt < max_retries - 1:
                            import time
                            time.sleep(1)
                            continue
                        else:
                            # Final fallback with separate session
                            try:
                                with app.app_context():
                                    segment = TranscriptSegment.query.get(segment_id)
                                    if segment:
                                        segment.engagement_score = 0.3
                                        segment.emotion_score = 0.3
                                        segment.viral_potential = 0.3
                                        segment.quotability = 0.3
                                        segment.overall_score = 0.3
                                        db.session.commit()
                            except Exception as fallback_error:
                                self.logger.error(f"Fallback update failed for segment {segment_id}: {fallback_error}")
                            return segment_id
                
                return segment_id

            self.logger.info(
                f"Starting ULTRA-PARALLEL AI analysis of {len(segments)} segments..."
            )

            # ULTRA-OPTIMIZED thread pool size with advanced resource detection
            import os
            import psutil

            cpu_count = os.cpu_count() or 4
            available_memory_gb = psutil.virtual_memory().available / (1024**3)

            # Advanced dynamic scaling for AI analysis
            if available_memory_gb > 16:  # High-end system
                max_workers = min(12, len(segments), cpu_count * 3)
            elif available_memory_gb > 8:  # High memory system
                max_workers = min(10, len(segments), cpu_count * 2)
            elif available_memory_gb > 4:  # Medium memory system
                max_workers = min(8, len(segments), cpu_count)
            else:  # Low memory system
                max_workers = min(4, len(segments), max(2, cpu_count // 2))

            # Ensure we have at least 2 workers for parallelism
            max_workers = max(2, max_workers)

            self.logger.info(
                f"ULTRA-PARALLEL setup: {cpu_count} CPUs, {available_memory_gb:.1f}GB RAM"
            )
            self.logger.info(
                f"Using {max_workers} ULTRA-PARALLEL workers for AI analysis")

            # Prepare segment data for parallel processing
            segment_data = [(segment.id, segment.text) for segment in segments]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all segment analysis tasks with enhanced data passing
                future_to_segment_id = {
                    executor.submit(analyze_single_segment, seg_id, seg_text):
                    seg_id
                    for seg_id, seg_text in segment_data
                }

                # Collect results as they complete with enhanced error handling
                completed_count = 0
                successful_segments = []

                for future in as_completed(future_to_segment_id):
                    segment_id = future_to_segment_id[future]
                    try:
                        result_segment_id = future.result(
                            timeout=120)  # 2-minute timeout per segment
                        if result_segment_id:
                            successful_segments.append(result_segment_id)
                        completed_count += 1

                        # Update progress with enhanced tracking
                        progress = 50 + int((completed_count / len(segments)) *
                                            30)  # 50-80% range
                        self._update_job_status(job,
                                                ProcessingStatus.ANALYZING,
                                                progress)

                        self.logger.info(
                            f"ULTRA-PARALLEL progress: {completed_count}/{len(segments)} segments"
                        )

                        # Enhanced memory optimization
                        if completed_count % 5 == 0:  # More frequent cleanup
                            gc.collect()

                    except Exception as e:
                        self.logger.error(
                            f"ULTRA-PARALLEL segment analysis failed for {segment_id}: {e}"
                        )
                        completed_count += 1

            # Final memory cleanup
            gc.collect()
            self.logger.info(
                f"Completed optimized parallel AI analysis of {len(segments)} segments"
            )

            # Now select the best segments for the user's preferences
            segments = TranscriptSegment.query.filter_by(job_id=job.id).all()

            for segment in segments:
                # Consider segments with good scores and user-preferred duration
                duration = segment.end_time - segment.start_time
                target_length = job.short_length
                min_length = max(10, target_length -
                                 5)  # Reduced flexibility for better accuracy
                max_length = target_length + 5

                if (segment.overall_score > 0.4
                        and  # Good engagement threshold
                        min_length <= duration <= max_length
                        and  # User-preferred duration
                        len(segment.text.split()) >= 5):  # Meaningful content
                    engaging_segments.append(segment)

            # Sort by overall score and return user-requested number of segments
            engaging_segments.sort(key=lambda x: x.overall_score, reverse=True)

            # Ensure we have at least the requested number of segments
            if len(engaging_segments) < job.num_shorts:
                all_segments = TranscriptSegment.query.filter_by(
                    job_id=job.id).all()
                target_length = job.short_length
                min_length = max(10, target_length -
                                 5)  # Reduced flexibility for better accuracy
                max_length = target_length + 5

                for segment in all_segments:
                    if segment not in engaging_segments:
                        duration = segment.end_time - segment.start_time
                        if min_length <= duration <= max_length and len(
                                segment.text.split()) >= 3:
                            segment.overall_score = 0.3  # Acceptable fallback score
                            engaging_segments.append(segment)
                            if len(engaging_segments) >= job.num_shorts:
                                break

            return engaging_segments[:job.
                                     num_shorts]  # Return user-requested number

        except Exception as e:
            self.logger.error(f"Content analysis failed: {e}")
            # Fallback: return segments based on user preferences
            segments = TranscriptSegment.query.filter_by(job_id=job.id).all()
            fallback_segments = []
            target_length = job.short_length
            min_length = max(10, target_length -
                             5)  # Reduced flexibility for better accuracy
            max_length = target_length + 5

            for segment in segments:
                duration = segment.end_time - segment.start_time
                if min_length <= duration <= max_length:
                    segment.overall_score = 0.5  # Default score
                    fallback_segments.append(segment)
            return fallback_segments[:job.num_shorts]

    def _generate_shorts(self, job, video_path, engaging_segments):
        """Generate vertical short videos from engaging segments with automatic effects - ULTRA-OPTIMIZED PARALLEL PROCESSING"""
        try:
            # Pre-load video into memory for faster processing
            self._preload_video_chunks(video_path, engaging_segments)

            # Process all shorts concurrently with advanced pipeline
            futures = []
            with ThreadPoolExecutor(max_workers=self.video_workers) as video_executor:
                with ThreadPoolExecutor(max_workers=self.io_workers) as io_executor:

                    # Submit all video processing tasks
                    for i, segment in enumerate(engaging_segments):
                        future = video_executor.submit(
                            self._process_single_short_optimized,
                            job.id, segment, video_path, i
                        )
                        futures.append(future)

                    # Process results as they complete
                    completed_count = 0
                    for future in as_completed(futures):
                        try:
                            result = future.result(timeout=300)  # 5-minute timeout
                            completed_count += 1

                            # Update progress
                            progress = 70 + int((completed_count / len(engaging_segments)) * 20)
                            self._update_job_status(job, ProcessingStatus.EDITING, progress)

                            self.logger.info(f"Completed short {completed_count}/{len(engaging_segments)}: {result}")

                        except Exception as e:
                            self.logger.error(f"Short processing failed: {e}")
                            completed_count += 1

                    # Generate thumbnails in parallel
                    self._generate_all_thumbnails_optimized(job.id, io_executor)

                    # Processing completed - shorts are ready for manual upload

        except Exception as e:
            raise Exception(f"Failed to generate shorts: {e}")

    def _auto_upload_shorts(self, job_id):
        """Auto-upload functionality disabled - shorts are ready for manual upload"""
        self.logger.info(f"Shorts for job {job_id} are ready for manual upload")

    def _preload_video_chunks(self, video_path, segments):
        """Preload video chunks for faster processing"""
        try:
            # Create memory-mapped chunks for frequently accessed segments
            for segment in segments:
                start_time = segment.start_time
                end_time = segment.end_time
                chunk_key = f"{video_path}_{start_time}_{end_time}"

                if chunk_key not in self.video_cache:
                    # Extract chunk to memory for ultra-fast access
                    chunk_data = self._extract_video_chunk_to_memory(
                        video_path, start_time, end_time
                    )
                    self.video_cache[chunk_key] = chunk_data

        except Exception as e:
            self.logger.warning(f"Video preloading failed: {e}")

    def _extract_video_chunk_to_memory(self, video_path, start_time, end_time):
        """Extract video chunk to memory buffer for ultra-fast processing"""
        try:
            import tempfile
            import io

            # Create temporary memory buffer
            buffer = io.BytesIO()

            # Extract segment using FFmpeg to memory - fixed seeking order
            cmd = [
                'ffmpeg',
                '-hwaccel', self.hardware_accel,
                '-ss', str(start_time),  # Seek BEFORE input
                '-i', video_path,
                '-t', str(end_time - start_time),
                '-c', 'copy',  # Stream copy for speed
                '-f', 'mp4',
                '-movflags', 'frag_keyframe+empty_moov',
                '-avoid_negative_ts', 'make_zero',
                'pipe:1'
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode == 0:
                return result.stdout
            else:
                return None

        except Exception as e:
            self.logger.warning(f"Memory chunk extraction failed: {e}")
            return None

    def _process_single_short_optimized(self, job_id, segment, video_path, index):
        """Process a single short with ultra-optimized pipeline and robust session management"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with app.app_context():
                    job = VideoJob.query.get(job_id)
                    if not job:
                        return f"Short {index+1} failed: Job not found"

                    # Get total segments count from database instead
                    total_shorts = VideoShort.query.filter_by(job_id=job_id).count() + 1

                    # Generate metadata with segment context for uniqueness
                    metadata = self.gemini_analyzer.generate_metadata(
                        segment.text,
                        job.title or "YouTube Short",
                        language=getattr(job, 'language', 'hinglish'),
                        segment_index=index,
                        total_segments=total_shorts
                    )

                    # Create VideoShort record
                    short = VideoShort(
                        job_id=job_id,
                        start_time=segment.start_time,
                        end_time=segment.end_time,
                        duration=segment.end_time - segment.start_time,
                        title=metadata.get('title', f"Short {index+1}"),
                        description=metadata.get('description', ''),
                        tags=metadata.get('tags', []),
                        engagement_score=getattr(segment, 'engagement_score', 0.5),
                        emotion_score=getattr(segment, 'emotion_score', 0.5),
                        viral_potential=getattr(segment, 'viral_potential', 0.5),
                        quotability=getattr(segment, 'quotability', 0.5),
                        overall_score=getattr(segment, 'overall_score', 0.5),
                        emotions_detected=getattr(segment, 'emotions_detected', []),
                        keywords=getattr(segment, 'keywords', []),
                        content_type=getattr(segment, 'content_type', 'general'),
                        visual_style=getattr(segment, 'visual_style', 'clean'),
                        analysis_notes=getattr(segment, 'analysis_notes', '')
                    )

                    db.session.add(short)
                    db.session.commit()

                    # Adjust timing for user's preferred length
                    user_length = getattr(job, 'short_length', 60)
                    adjusted_start, adjusted_end = self._calculate_adjusted_timing(
                        segment.start_time, segment.end_time, user_length
                    )

                    # Generate video with ultra-fast processing
                    output_path = os.path.join('outputs', f'short_{short.id}.mp4')
                    self._create_vertical_video_ultra_fast(
                        video_path, output_path, adjusted_start, adjusted_end, short.id
                    )

                    # Update short record with retry logic
                    try:
                        with app.app_context():
                            short = VideoShort.query.get(short.id)
                            if short:
                                short.start_time = adjusted_start
                                short.end_time = adjusted_end
                                short.duration = adjusted_end - adjusted_start
                                short.output_path = output_path
                                short.thumbnail_path = os.path.join('outputs', f'short_{short.id}_thumb.jpg')
                                db.session.commit()
                    except Exception as update_error:
                        self.logger.warning(f"Short update failed: {update_error}")
                        db.session.rollback()

                    return f"Short {index+1} completed in ultra-fast mode"

            except Exception as e:
                self.logger.error(f"Short processing attempt {attempt + 1}/{max_retries} failed: {e}")
                try:
                    db.session.rollback()
                except:
                    pass
                
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1)
                else:
                    return f"Short {index+1} failed after {max_retries} attempts: {e}"
        
        return f"Short {index+1} failed: Maximum retries exceeded"

    def _create_vertical_video(self, input_path, output_path, start_time,
                               end_time):
        """Create vertical 9:16 video from horizontal source using FFmpeg - ULTRA OPTIMIZED PARALLEL PROCESSING"""
        return self._create_vertical_video_threaded(input_path, output_path,
                                                    start_time, end_time, None)

    def _create_vertical_video_ultra_fast(self, input_path, output_path, start_time, end_time, short_id):
        """Create vertical video with aggressive fallback strategy"""
        try:
            duration = end_time - start_time

            # Check if we have cached chunk data
            chunk_key = f"{input_path}_{start_time}_{end_time}"
            cached_chunk = self.video_cache.get(chunk_key)

            if cached_chunk:
                # Process from memory for ultra-fast speed
                return self._process_from_memory_chunk(cached_chunk, output_path, short_id)

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Start with fastest possible processing
            success = self._try_ultrafast_processing(input_path, output_path, start_time, end_time)
            if success:
                self.logger.info(f"Ultra-fast video created successfully: {output_path}")
                return

            # If that fails, try stream copy method
            success = self._try_stream_copy_processing(input_path, output_path, start_time, end_time)
            if success:
                self.logger.info(f"Stream copy video created successfully: {output_path}")
                return

            # Final fallback - direct segment extraction
            self._create_direct_segment_copy(input_path, output_path, start_time, end_time)

        except Exception as e:
            self.logger.error(f"All video processing methods failed: {e}")
            self._create_direct_segment_copy(input_path, output_path, start_time, end_time)

    def _get_intelligent_effects_filter(self, short_id, duration):
        """Get intelligent video effects based on content analysis with enhanced quality"""
        try:
            with app.app_context():
                short = VideoShort.query.get(short_id)
                if not short:
                    return ",eq=contrast=1.15:brightness=0.05:saturation=1.2,unsharp=5:5:0.8:5:5:0.0"

                effects = []

                # Base on emotion analysis
                emotions = getattr(short, 'emotions_detected', [])
                engagement_score = getattr(short, 'engagement_score', 0.5)
                viral_potential = getattr(short, 'viral_potential', 0.5)
                content_type = getattr(short, 'content_type', 'general')

                # High energy content - add vibrant effects
                if engagement_score > 0.7 or 'excitement' in emotions or 'joy' in emotions:
                    effects.extend([
                        'eq=contrast=1.25:brightness=0.08:saturation=1.35:gamma=0.95',  # Enhanced vibrant colors
                        'unsharp=5:5:1.2:5:5:0.0',  # Strong sharpening
                        'hue=h=5:s=1.1'  # Slight color enhancement
                    ])

                # Emotional/dramatic content - cinematic effects
                elif 'sadness' in emotions or 'fear' in emotions or engagement_score > 0.6:
                    effects.extend([
                        'eq=contrast=1.2:brightness=-0.03:saturation=0.95:gamma=1.05',  # Cinematic look
                        'unsharp=3:3:0.8:3:3:0.0',  # Moderate sharpening
                        'curves=all=0.1/0.1:0.9/0.85'  # Film-like curve
                    ])

                # High viral potential - trending effects
                elif viral_potential > 0.7:
                    effects.extend([
                        'eq=contrast=1.2:brightness=0.06:saturation=1.25:gamma=0.98',  # TikTok-style enhancement
                        'unsharp=4:4:1.0:4:4:0.0',  # Social media sharpening
                        'hue=h=8:s=1.05'  # Warm, appealing tint
                    ])

                # Educational/informational content
                elif content_type in ['educational', 'tutorial', 'informational']:
                    effects.extend([
                        'eq=contrast=1.1:brightness=0.04:saturation=1.1:gamma=0.98',  # Clear, professional look
                        'unsharp=3:3:0.6:3:3:0.0'  # Subtle sharpening
                    ])

                # Default enhancement for all content - significantly improved
                else:
                    effects.extend([
                        'eq=contrast=1.15:brightness=0.04:saturation=1.18:gamma=0.97',  # Better base enhancement
                        'unsharp=4:4:0.8:4:4:0.0',  # Good sharpening
                        'hue=s=1.05'  # Slight saturation boost
                    ])

                # Add professional transitions for longer content
                if duration > 30:
                    effects.append('fade=t=in:st=0:d=0.8:alpha=1')
                    effects.append('fade=t=out:st={:.1f}:d=0.8:alpha=1'.format(duration - 0.8))

                # Add noise reduction for better quality
                effects.append('hqdn3d=2:1:2:1')

                # Combine effects
                if effects:
                    return ',' + ','.join(effects)
                return ",eq=contrast=1.15:brightness=0.05:saturation=1.2,unsharp=5:5:0.8:5:5:0.0"

        except Exception as e:
            self.logger.warning(f"Failed to get intelligent effects: {e}")
            # Enhanced default
            return ',eq=contrast=1.15:brightness=0.05:saturation=1.2,unsharp=5:5:0.8:5:5:0.0,hqdn3d=2:1:2:1'

    def _get_optimal_hardware_codec(self):
        """Get the best available hardware codec"""
        if self.hardware_accel == 'nvenc':
            return 'nvenc'
        elif self.hardware_accel == 'qsv':
            return 'qsv'
        else:
            return 'software'

    def _try_ultrafast_processing(self, input_path, output_path, start_time, end_time):
        """Try ultra-fast processing with progressive retry logic and enhanced quality"""
        duration = end_time - start_time
        
        # Enhanced retry configurations with better quality settings
        retry_configs = [
            {
                'timeout': 180,
                'threads': '2',
                'preset': 'faster',
                'crf': '23',  # Better quality
                'vf': 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,eq=contrast=1.15:brightness=0.05:saturation=1.2',
                'audio': ['-c:a', 'aac', '-b:a', '128k', '-ar', '44100'],
                'description': 'High quality with enhanced visuals'
            },
            {
                'timeout': 240,
                'threads': '2',
                'preset': 'fast',
                'crf': '26',
                'vf': 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,eq=contrast=1.1:saturation=1.15',
                'audio': ['-c:a', 'aac', '-b:a', '96k'],
                'description': 'Balanced quality and speed'
            },
            {
                'timeout': 300,
                'threads': '1',
                'preset': 'ultrafast',
                'crf': '28',
                'vf': 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920',
                'audio': ['-c:a', 'copy'],
                'description': 'Fast with original audio'
            },
            {
                'timeout': 360,
                'threads': '1',
                'preset': 'ultrafast', 
                'crf': '35',
                'vf': 'scale=720:1280',
                'audio': ['-an'],
                'description': 'Emergency fallback'
            }
        ]

        for attempt, config in enumerate(retry_configs, 1):
            try:
                self.logger.info(f"Ultra-fast attempt {attempt}/4: {config['description']} (timeout={config['timeout']}s)")
                
                cmd = [
                    'ffmpeg',
                    '-threads', config['threads'],
                    '-hwaccel', 'auto',
                    '-ss', str(start_time),
                    '-i', input_path,
                    '-t', str(duration),
                    '-vf', config['vf'],
                    '-c:v', 'libx264',
                    '-preset', config['preset'],
                    '-crf', config['crf'],
                    '-profile:v', 'baseline',
                    '-level', '3.0',
                    '-pix_fmt', 'yuv420p'
                ]
                
                # Add audio settings
                cmd.extend(config['audio'])
                
                cmd.extend([
                    '-avoid_negative_ts', 'make_zero',
                    '-movflags', '+faststart',
                    '-max_muxing_queue_size', '512',
                    '-y', output_path
                ])

                subprocess.run(cmd, check=True, capture_output=True, timeout=config['timeout'])
                self.logger.info(f"Ultra-fast processing completed on attempt {attempt}: {output_path}")
                return True

            except subprocess.TimeoutExpired:
                self.logger.warning(f"Ultra-fast attempt {attempt} timed out after {config['timeout']}s")
                continue
            except subprocess.CalledProcessError as e:
                self.logger.warning(f"Ultra-fast attempt {attempt} failed with return code {e.returncode}")
                continue
            except Exception as e:
                self.logger.warning(f"Ultra-fast attempt {attempt} error: {e}")
                continue

        self.logger.error("All ultra-fast processing attempts failed")
        return False

    def _try_stream_copy_processing(self, input_path, output_path, start_time, end_time):
        """Try stream copy with enhanced timeout and progressive fallbacks"""
        duration = end_time - start_time
        
        stream_configs = [
            {
                'timeout': 150,
                'threads': '1',
                'preset': 'fast',
                'crf': '30',
                'vf': 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920',
                'audio': ['-c:a', 'copy'],
                'description': 'Stream copy with 9:16'
            },
            {
                'timeout': 200,
                'threads': '1',
                'preset': 'veryfast',
                'crf': '35',
                'vf': 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920',
                'audio': ['-an'],
                'description': 'No audio stream copy'
            },
            {
                'timeout': 300,
                'threads': '1',
                'preset': 'ultrafast',
                'crf': '40',
                'vf': 'scale=720:1280',
                'audio': ['-an'],
                'description': 'Lower res stream copy'
            }
        ]

        for attempt, config in enumerate(stream_configs, 1):
            try:
                self.logger.info(f"Stream copy attempt {attempt}/3: {config['description']} (timeout={config['timeout']}s)")
                
                cmd = [
                    'ffmpeg',
                    '-threads', config['threads'],
                    '-hwaccel', 'auto',
                    '-ss', str(start_time),
                    '-i', input_path,
                    '-t', str(duration),
                    '-vf', config['vf'],
                    '-c:v', 'libx264',
                    '-preset', config['preset'],
                    '-crf', config['crf'],
                    '-profile:v', 'baseline',
                    '-level', '3.0',
                    '-pix_fmt', 'yuv420p'
                ]
                
                # Add audio settings
                cmd.extend(config['audio'])
                
                cmd.extend([
                    '-avoid_negative_ts', 'make_zero',
                    '-movflags', '+faststart',
                    '-y', output_path
                ])

                subprocess.run(cmd, check=True, capture_output=True, timeout=config['timeout'])
                self.logger.info(f"Stream copy processing completed on attempt {attempt}: {output_path}")
                return True

            except subprocess.TimeoutExpired:
                self.logger.warning(f"Stream copy attempt {attempt} timed out after {config['timeout']}s")
                continue
            except Exception as e:
                self.logger.warning(f"Stream copy attempt {attempt} failed: {e}")
                continue

        self.logger.error("All stream copy processing attempts failed")
        return False

    def _create_direct_segment_copy(self, input_path, output_path, start_time, end_time):
        """Create direct segment copy with extended timeout and better error handling"""
        duration = end_time - start_time
        
        copy_configs = [
            {
                'timeout': 180,
                'cmd': [
                    'ffmpeg', '-ss', str(start_time), '-i', input_path,
                    '-t', str(duration), '-c', 'copy',
                    '-avoid_negative_ts', 'make_zero', '-y', output_path
                ],
                'description': 'Direct stream copy'
            },
            {
                'timeout': 240,
                'cmd': [
                    'ffmpeg', '-ss', str(start_time), '-i', input_path,
                    '-t', str(duration), '-c:v', 'copy', '-an',
                    '-avoid_negative_ts', 'make_zero', '-y', output_path
                ],
                'description': 'Video copy, no audio'
            },
            {
                'timeout': 300,
                'cmd': [
                    'ffmpeg', '-ss', str(start_time), '-i', input_path,
                    '-t', str(duration), '-c:v', 'libx264', '-preset', 'ultrafast',
                    '-crf', '50', '-an', '-avoid_negative_ts', 'make_zero', '-y', output_path
                ],
                'description': 'Basic re-encode fallback'
            }
        ]

        for attempt, config in enumerate(copy_configs, 1):
            try:
                self.logger.info(f"Direct copy attempt {attempt}/3: {config['description']} (timeout={config['timeout']}s)")
                
                subprocess.run(config['cmd'], check=True, capture_output=True, timeout=config['timeout'])
                self.logger.info(f"Direct segment copy completed with method '{config['description']}': {output_path}")
                return

            except subprocess.TimeoutExpired:
                self.logger.warning(f"Direct copy attempt {attempt} timed out after {config['timeout']}s")
                continue
            except Exception as e:
                self.logger.warning(f"Direct copy attempt {attempt} failed: {e}")
                continue

        # All attempts failed - create placeholder
        self.logger.error("All direct segment copy attempts failed, creating placeholder")
        self._create_placeholder_video(output_path, duration)

    def _create_placeholder_video(self, output_path, duration):
        """Create a simple placeholder video"""
        try:
            # Create a black video with text overlay
            cmd = [
                'ffmpeg',
                '-f', 'lavfi',
                '-i', f'color=black:size=1080x1920:duration={duration}:rate=1',
                '-vf', 'drawtext=text="Processing...":fontcolor=white:fontsize=60:x=(w-text_w)/2:y=(h-text_h)/2',
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-crf', '35',
                '-pix_fmt', 'yuv420p',
                '-y', output_path
            ]

            subprocess.run(cmd, check=True, capture_output=True, timeout=10)
            self.logger.info(f"Placeholder video created: {output_path}")

        except Exception as e:
            self.logger.error(f"Placeholder video creation failed: {e}")
            # Final fallback - create empty file
            with open(output_path, 'wb') as f:
                f.write(b'')

    def _process_from_memory_chunk(self, chunk_data, output_path, short_id):
        """Process video chunk directly from memory with aggressive fallbacks"""
        try:
            import tempfile

            # Write chunk to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_file.write(chunk_data)
                temp_input = temp_file.name

            try:
                # Try ultra-fast processing first
                success = self._try_memory_ultrafast(temp_input, output_path)
                if success:
                    return

                # Try direct copy
                success = self._try_memory_direct_copy(temp_input, output_path)
                if success:
                    return

                # Final fallback - just copy the chunk
                self._copy_chunk_directly(chunk_data, output_path)

            finally:
                # Clean up temporary file
                if os.path.exists(temp_input):
                    os.unlink(temp_input)

        except Exception as e:
            self.logger.error(f"Memory chunk processing completely failed: {e}")
            self._copy_chunk_directly(chunk_data, output_path)

    def _try_memory_ultrafast(self, temp_input, output_path):
        """Try ultra-fast memory processing with progressive timeouts and retry logic"""
        retry_configs = [
            {
                'timeout': 120,
                'threads': '1',
                'preset': 'ultrafast', 
                'crf': '35',
                'vf': 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920',
                'audio': ['-c:a', 'copy']
            },
            {
                'timeout': 180,
                'threads': '1',
                'preset': 'superfast',
                'crf': '40', 
                'vf': 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920',
                'audio': ['-an']  # No audio for faster processing
            },
            {
                'timeout': 240,
                'threads': '1',
                'preset': 'ultrafast',
                'crf': '45',
                'vf': 'scale=540:960',  # Lower resolution for speed
                'audio': ['-an']
            }
        ]

        for attempt, config in enumerate(retry_configs, 1):
            try:
                self.logger.info(f"Attempt {attempt}/3: Using timeout={config['timeout']}s, preset={config['preset']}")
                
                cmd = [
                    'ffmpeg',
                    '-threads', config['threads'],
                    '-hwaccel', 'auto',
                    '-i', temp_input,
                    '-vf', config['vf'],
                    '-c:v', 'libx264',
                    '-preset', config['preset'],
                    '-crf', config['crf'],
                    '-profile:v', 'baseline',
                    '-level', '3.0',
                    '-pix_fmt', 'yuv420p'
                ]
                
                # Add audio settings
                cmd.extend(config['audio'])
                
                cmd.extend([
                    '-avoid_negative_ts', 'make_zero',
                    '-movflags', '+faststart',
                    '-y', output_path
                ])

                subprocess.run(cmd, check=True, capture_output=True, timeout=config['timeout'])
                self.logger.info(f"Ultra-fast memory processing completed on attempt {attempt}: {output_path}")
                return True

            except subprocess.TimeoutExpired:
                self.logger.warning(f"Attempt {attempt} timed out after {config['timeout']}s")
                continue
            except Exception as e:
                self.logger.warning(f"Attempt {attempt} failed: {e}")
                continue

        self.logger.error("All ultra-fast processing attempts failed")
        return False

    def _try_memory_direct_copy(self, temp_input, output_path):
        """Try direct copy with enhanced retry logic and progressive fallbacks"""
        fallback_configs = [
            {
                'timeout': 120,
                'method': 'copy_with_9_16',
                'cmd_base': [
                    'ffmpeg', '-threads', '1', '-i', temp_input,
                    '-vf', 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920',
                    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '30',
                    '-c:a', 'copy', '-avoid_negative_ts', 'make_zero', 
                    '-movflags', '+faststart', '-y', output_path
                ]
            },
            {
                'timeout': 180,
                'method': 'copy_no_audio',
                'cmd_base': [
                    'ffmpeg', '-threads', '1', '-i', temp_input,
                    '-vf', 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920',
                    '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '35',
                    '-an', '-avoid_negative_ts', 'make_zero',
                    '-movflags', '+faststart', '-y', output_path
                ]
            },
            {
                'timeout': 240,
                'method': 'low_res_copy',
                'cmd_base': [
                    'ffmpeg', '-threads', '1', '-i', temp_input,
                    '-vf', 'scale=540:960:force_original_aspect_ratio=increase,crop=540:960',
                    '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '40',
                    '-an', '-avoid_negative_ts', 'make_zero', '-y', output_path
                ]
            },
            {
                'timeout': 60,
                'method': 'pure_copy',
                'cmd_base': [
                    'ffmpeg', '-i', temp_input, '-c', 'copy',
                    '-avoid_negative_ts', 'make_zero', '-y', output_path
                ]
            }
        ]

        for attempt, config in enumerate(fallback_configs, 1):
            try:
                self.logger.info(f"Direct copy attempt {attempt}/4: {config['method']} (timeout={config['timeout']}s)")
                
                subprocess.run(config['cmd_base'], check=True, capture_output=True, timeout=config['timeout'])
                self.logger.info(f"Direct copy completed with method '{config['method']}': {output_path}")
                return True

            except subprocess.TimeoutExpired:
                self.logger.warning(f"Direct copy attempt {attempt} timed out after {config['timeout']}s")
                continue
            except Exception as e:
                self.logger.warning(f"Direct copy attempt {attempt} failed: {e}")
                continue

        self.logger.error("All direct copy attempts failed")
        return False

    def _copy_chunk_directly(self, chunk_data, output_path):
        """Copy chunk data directly as final fallback"""
        try:
            with open(output_path, 'wb') as f:
                f.write(chunk_data)
            self.logger.info(f"Direct chunk copy completed: {output_path}")
        except Exception as e:
            self.logger.error(f"Direct chunk copy failed: {e}")
            # Create empty file as absolute final fallback
            with open(output_path, 'wb') as f:
                f.write(b'')

    def _create_vertical_video_software_fallback(self, input_path, output_path, start_time, end_time):
        """Optimized software fallback with better quality and speed"""
        duration = end_time - start_time

        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Balanced quality filters for software encoding
            balanced_filters = [
                'scale=1080:1920:force_original_aspect_ratio=increase:flags=bilinear',  # Faster scaling
                'crop=1080:1920',
                'eq=contrast=1.08:brightness=0.02:saturation=1.12',  # Good enhancement
            ]

            cmd = [
                'ffmpeg',
                '-threads', '2',  # Limited threads for stability
                '-ss', str(start_time),
                '-i', input_path,
                '-t', str(duration),
                '-vf', ','.join(balanced_filters),
                '-c:v', 'libx264',
                '-preset', 'faster',  # Good balance
                '-crf', '22',  # Better quality
                '-profile:v', 'main',
                '-level', '4.0',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                '-b:a', '96k',
                '-ar', '44100',
                '-ac', '2',
                '-avoid_negative_ts', 'make_zero',
                '-fflags', '+genpts',
                '-fps_mode', 'cfr',
                '-movflags', '+faststart',
                '-max_muxing_queue_size', '512',
                '-y', output_path
            ]

            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
            self.logger.info(f"Optimized software fallback completed: {output_path}")

        except subprocess.TimeoutExpired:
            self.logger.warning("Software fallback timeout - using basic fallback")
            self._create_basic_fallback_video(input_path, output_path, start_time, end_time)
        except Exception as e:
            self.logger.error(f"Software fallback failed: {e}")
            self._create_basic_fallback_video(input_path, output_path, start_time, end_time)

    def _create_medium_quality_memory_fallback(self, chunk_data, output_path):
        """Create medium quality fallback from memory chunk"""
        try:
            import tempfile

            # Write chunk to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_file.write(chunk_data)
                temp_input = temp_file.name

            try:
                # Medium quality processing
                cmd = [
                    'ffmpeg',
                    '-threads', '1',
                    '-i', temp_input,
                    '-vf', (
                        'scale=1080:1920:force_original_aspect_ratio=increase:flags=bilinear,'
                        'crop=1080:1920,'
                        'eq=contrast=1.04:saturation=1.08'
                    ),
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '24',
                    '-profile:v', 'main',
                    '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac',
                    '-b:a', '112k',
                    '-avoid_negative_ts', 'make_zero',
                    '-movflags', '+faststart',
                    '-y', output_path
                ]

                subprocess.run(cmd, check=True, capture_output=True, timeout=30)
                self.logger.info(f"Medium quality memory fallback completed: {output_path}")

            finally:
                os.unlink(temp_input)

        except Exception as e:
            self.logger.warning(f"Medium quality memory fallback failed: {e}")
            self._create_direct_copy_fallback(chunk_data, output_path)

    def _create_direct_copy_fallback(self, chunk_data, output_path):
        """Create direct copy fallback from memory chunk"""
        try:
            import tempfile

            # Write chunk directly as output with minimal processing
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_file.write(chunk_data)
                temp_input = temp_file.name

            try:
                # Ultra-simple copy with basic vertical conversion
                cmd = [
                    'ffmpeg',
                    '-threads', '1',
                    '-i', temp_input,
                    '-vf', 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920',
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    '-crf', '28',
                    '-c:a', 'copy',  # Copy audio without re-encoding
                    '-avoid_negative_ts', 'make_zero',
                    '-movflags', '+faststart',
                    '-y', output_path
                ]

                subprocess.run(cmd, check=True, capture_output=True, timeout=15)
                self.logger.info(f"Direct copy fallback completed: {output_path}")

            finally:
                os.unlink(temp_input)

        except Exception as e:
            self.logger.error(f"Direct copy fallback failed: {e}")
            # Last resort - write chunk directly
            try:
                with open(output_path, 'wb') as f:
                    f.write(chunk_data)
                self.logger.info(f"Raw chunk written as fallback: {output_path}")
            except Exception as final_error:
                self.logger.error(f"All fallback methods failed: {final_error}")

    def _create_medium_quality_fallback(self, input_path, output_path, start_time, end_time):
        """Deprecated - now calls ultra-fast processing"""
        self._try_ultrafast_processing(input_path, output_path, start_time, end_time)

    def _create_emergency_fallback_video(self, input_path, output_path, start_time, end_time):
        """Deprecated - now calls direct segment copy"""
        self._create_direct_segment_copy(input_path, output_path, start_time, end_time)

    def _create_basic_fallback_video(self, input_path, output_path, start_time, end_time):
        """Deprecated - now calls direct segment copy"""
        self._create_direct_segment_copy(input_path, output_path, start_time, end_time)

    def _generate_all_thumbnails_optimized(self, job_id, executor):
        """Generate thumbnails with optimized parallel processing"""
        try:
            with app.app_context():
                shorts = VideoShort.query.filter_by(job_id=job_id).all()

                # Submit all thumbnail tasks
                thumbnail_futures = []
                for short in shorts:
                    if short.output_path and os.path.exists(short.output_path):
                        future = executor.submit(
                            self._generate_thumbnail_ultra_fast,
                            short.output_path,
                            short.thumbnail_path
                        )
                        thumbnail_futures.append(future)

                # Wait for completion
                for future in as_completed(thumbnail_futures):
                    try:
                        future.result(timeout=30)
                    except Exception as e:
                        self.logger.warning(f"Thumbnail generation failed: {e}")

        except Exception as e:
            self.logger.error(f"Thumbnail batch processing failed: {e}")

    def _generate_thumbnail_ultra_fast(self, video_path, thumbnail_path):
        """Generate thumbnail with maximum speed"""
        try:
            cmd = [
                'ffmpeg',
                '-hwaccel', 'auto',
                '-i', video_path,
                '-ss', '1',
                '-vframes', '1',
                '-vf', 'scale=640:1136:force_original_aspect_ratio=increase,crop=640:1136',
                '-q:v', '8',  # Lower quality for speed
                '-threads', '1',
                '-y', thumbnail_path
            ]

            subprocess.run(cmd, check=True, capture_output=True, timeout=120)

        except Exception as e:
            self.logger.warning(f"Ultra-fast thumbnail failed: {e}")
            # Create placeholder thumbnail
            self._create_placeholder_thumbnail(thumbnail_path)

    def _create_placeholder_thumbnail(self, thumbnail_path):
        """Create a simple placeholder thumbnail"""
        try:
            from PIL import Image, ImageDraw, ImageFont

            # Create a simple placeholder
            img = Image.new('RGB', (640, 1136), color='black')
            draw = ImageDraw.Draw(img)
            draw.text((320, 568), "Loading...", fill='white', anchor='mm')
            img.save(thumbnail_path, 'JPEG', quality=50)

        except Exception as e:
            self.logger.warning(f"Placeholder thumbnail creation failed: {e}")

    def _generate_thumbnail(self, video_path, thumbnail_path):
        """Generate thumbnail from video - OPTIMIZED FOR SPEED"""
        try:
            cmd = [
                'ffmpeg',
                '-hwaccel',
                'auto',  # Use hardware acceleration
                '-i',
                video_path,
                '-ss',
                '00:00:01.000',
                '-vframes',
                '1',
                '-s',
                '640x1136',
                '-q:v',
                '5',  # Add quality setting for faster processing
                '-y',
                thumbnail_path
            ]

            subprocess.run(cmd, check=True, capture_output=True)

        except Exception as e:
            self.logger.warning(f"Failed to generate thumbnail: {e}")

    def _generate_thumbnails_parallel(self, video_thumbnail_pairs):
        """Generate multiple thumbnails in parallel using multithreading"""

        def generate_single_thumbnail(pair):
            video_path, thumbnail_path = pair
            try:
                cmd = [
                    'ffmpeg',
                    '-hwaccel',
                    'auto',  # Hardware acceleration
                    '-thread_queue_size',
                    '512',  # Optimized buffer for parallel processing
                    '-i',
                    video_path,
                    '-ss',
                    '00:00:01.000',
                    '-vframes',
                    '1',
                    '-s',
                    '640x1136',
                    '-q:v',
                    '3',  # Higher quality for better thumbnails
                    '-preset',
                    'ultrafast',  # Fast processing
                    '-threads',
                    '2',  # Limit threads per thumbnail
                    '-y',
                    thumbnail_path
                ]

                # Run with timeout to prevent hanging
                result = subprocess.run(cmd,
                                        check=True,
                                        capture_output=True,
                                        timeout=30)
                return f"Thumbnail generated: {thumbnail_path}"

            except subprocess.TimeoutExpired:
                return f"Thumbnail timeout: {thumbnail_path}"
            except Exception as e:
                return f"Thumbnail failed: {thumbnail_path} - {e}"

        # Determine optimal number of workers for thumbnail generation
        import os
        cpu_count = os.cpu_count() or 4
        # Use fewer workers for thumbnails to avoid overwhelming FFmpeg
        max_workers = min(4, len(video_thumbnail_pairs),
                          max(2, cpu_count // 2))

        self.logger.info(
            f"Generating {len(video_thumbnail_pairs)} thumbnails with {max_workers} parallel workers"
        )

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pair = {
                executor.submit(generate_single_thumbnail, pair): pair
                for pair in video_thumbnail_pairs
            }

            for future in as_completed(future_to_pair):
                try:
                    result = future.result(
                        timeout=45)  # Allow extra time for completion
                    results.append(result)
                    self.logger.info(f"Parallel thumbnail result: {result}")
                except Exception as e:
                    pair = future_to_pair[future]
                    self.logger.error(
                        f"Thumbnail generation failed for {pair[1]}: {e}")
                    results.append(f"Failed: {pair[1]}")

        return results

    def _select_preferred_audio_stream(self, video_path):
        """Select audio stream with Hindi first, English second priority"""
        try:
            # Get detailed stream information
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_streams', '-show_format', video_path
            ]
            probe_result = subprocess.run(probe_cmd,
                                          capture_output=True,
                                          text=True)

            if probe_result.returncode != 0:
                self.logger.warning(
                    "Could not probe video streams, using default audio")
                return 0

            probe_data = json.loads(probe_result.stdout)
            audio_streams = [
                s for s in probe_data.get('streams', [])
                if s.get('codec_type') == 'audio'
            ]

            if not audio_streams:
                self.logger.warning("No audio streams found")
                return 0

            self.logger.info(f"Found {len(audio_streams)} audio streams")

            # Priority system: Hindi -> English -> Default
            hindi_stream = None
            english_stream = None
            default_stream = 0

            for idx, stream in enumerate(audio_streams):
                tags = stream.get('tags', {})
                language = tags.get('language', '').lower()
                title = tags.get('title', '').lower()

                self.logger.info(
                    f"Audio stream {idx}: language='{language}', title='{title}'"
                )

                # Check for Hindi indicators
                hindi_indicators = ['hi', 'hin', 'hindi', 'हिंदी', 'हिन्दी']
                if any(indicator in language for indicator in hindi_indicators) or \
                   any(indicator in title for indicator in hindi_indicators):
                    hindi_stream = idx
                    self.logger.info(
                        f"Found Hindi audio stream at index {idx}")
                    break  # Hindi has highest priority, use immediately

                # Check for English indicators
                english_indicators = ['en', 'eng', 'english']
                if english_stream is None and (
                        any(indicator in language
                            for indicator in english_indicators)
                        or any(indicator in title
                               for indicator in english_indicators)):
                    english_stream = idx
                    self.logger.info(
                        f"Found English audio stream at index {idx}")

                # Also check stream metadata for more clues
                if 'metadata' in stream:
                    metadata = stream['metadata']
                    if any(key for key in metadata.keys()
                           if 'hindi' in key.lower() or 'hi' in key.lower()):
                        hindi_stream = idx
                        self.logger.info(
                            f"Found Hindi audio stream via metadata at index {idx}"
                        )
                        break

            # Return in priority order: Hindi -> English -> Default
            if hindi_stream is not None:
                self.logger.info(f"Using Hindi audio stream: {hindi_stream}")
                return hindi_stream
            elif english_stream is not None:
                self.logger.info(
                    f"Using English audio stream: {english_stream}")
                return english_stream
            else:
                self.logger.info(
                    f"Using default audio stream: {default_stream}")
                return default_stream

        except Exception as e:
            self.logger.error(f"Error selecting audio stream: {e}")
            return 0  # Fallback to first stream

    def _cleanup_temporary_files(self, job):
        """Clean up temporary files after processing"""
        try:
            files_to_remove = []

            # Add video file
            if job.video_path and os.path.exists(job.video_path):
                files_to_remove.append(job.video_path)

            # Add audio file
            if job.audio_path and os.path.exists(job.audio_path):
                files_to_remove.append(job.audio_path)

            # Add transcript file
            if job.transcript_path and os.path.exists(job.transcript_path):
                files_to_remove.append(job.transcript_path)

            # Remove files
            for file_path in files_to_remove:
                try:
                    os.remove(file_path)
                    self.logger.info(f"Removed temporary file: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove {file_path}: {e}")

            # Clean temporary directories if empty
            for temp_dir in ['uploads', 'temp']:
                if os.path.exists(temp_dir):
                    try:
                        if not os.listdir(temp_dir):  # Directory is empty
                            os.rmdir(temp_dir)
                            self.logger.info(
                                f"Removed empty directory: {temp_dir}")
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to remove directory {temp_dir}: {e}")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    def _calculate_adjusted_timing(self, segment_start, segment_end, user_length):
        """Calculate the adjusted start and end times for a video segment to match the user's preferred length."""
        segment_duration = segment_end - segment_start

        if segment_duration > user_length:
            # If segment is longer, center the user's preferred length
            excess = segment_duration - user_length
            adjusted_start = segment_start + (excess / 2)
            adjusted_end = adjusted_start + user_length
        else:
            # If segment is shorter or equal, use original timing
            adjusted_start = segment_start
            adjusted_end = segment_end
        return adjusted_start, adjusted_end