from app import db
from datetime import datetime
from enum import Enum
from sqlalchemy import JSON


class ProcessingStatus(Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    ANALYZING = "analyzing"
    EDITING = "editing"
    COMPLETED = "completed"
    FAILED = "failed"


class UploadStatus(Enum):
    PENDING = "pending"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoJob(db.Model):
    """Video processing job model"""
    id = db.Column(db.Integer, primary_key=True)
    youtube_url = db.Column(db.String(500), nullable=False)
    title = db.Column(db.String(200))
    duration = db.Column(db.Integer)  # Duration in seconds
    status = db.Column(db.Enum(ProcessingStatus), default=ProcessingStatus.PENDING)
    progress = db.Column(db.Integer, default=0)  # Progress percentage
    error_message = db.Column(db.Text)

    # File paths
    video_path = db.Column(db.String(500))
    audio_path = db.Column(db.String(500))
    transcript_path = db.Column(db.String(500))

    # Processing settings
    video_quality = db.Column(db.String(20), default='720p')
    language = db.Column(db.String(50), default='hinglish')
    short_length = db.Column(db.Integer, default=60)  # Duration in seconds
    num_shorts = db.Column(db.Integer, default=3)     # Number of shorts to generate

    # Video metadata
    video_info = db.Column(JSON)

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    shorts = db.relationship('VideoShort', backref='job', cascade='all, delete-orphan')
    transcript_segments = db.relationship('TranscriptSegment', backref='job', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<VideoJob {self.id}: {self.title}>'


class VideoShort(db.Model):
    """Generated short video model"""
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey('video_job.id'), nullable=False)

    # Video details
    title = db.Column(db.String(200))
    description = db.Column(db.Text)
    tags = db.Column(JSON)  # List of tags

    # Timing
    start_time = db.Column(db.Float, nullable=False)  # Start time in seconds
    end_time = db.Column(db.Float, nullable=False)    # End time in seconds
    duration = db.Column(db.Float)  # Duration in seconds

    # File paths
    output_path = db.Column(db.String(500))
    thumbnail_path = db.Column(db.String(500))

    # AI Analysis Scores
    engagement_score = db.Column(db.Float, default=0.0)
    emotion_score = db.Column(db.Float, default=0.0)
    viral_potential = db.Column(db.Float, default=0.0)
    quotability = db.Column(db.Float, default=0.0)
    overall_score = db.Column(db.Float, default=0.0)

    # AI Analysis Details
    emotions_detected = db.Column(JSON)  # List of detected emotions
    keywords = db.Column(JSON)  # List of keywords
    analysis_notes = db.Column(db.Text)  # AI analysis notes
    content_type = db.Column(db.String(100), default='general')
    visual_style = db.Column(db.String(100), default='clean')


    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    uploads = db.relationship('VideoUpload', backref='short', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<VideoShort {self.id}: {self.title}>'


class TranscriptSegment(db.Model):
    """Transcript segment model"""
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey('video_job.id'), nullable=False)

    # Segment details
    text = db.Column(db.Text, nullable=False)
    start_time = db.Column(db.Float, nullable=False)
    end_time = db.Column(db.Float, nullable=False)

    # Language detection
    language = db.Column(db.String(50))
    confidence = db.Column(db.Float)

    # AI Analysis scores
    engagement_score = db.Column(db.Float, default=0.0)
    emotion_score = db.Column(db.Float, default=0.0)
    viral_potential = db.Column(db.Float, default=0.0)
    quotability = db.Column(db.Float, default=0.0)
    overall_score = db.Column(db.Float, default=0.0)

    # AI Analysis Details
    emotions_detected = db.Column(JSON)  # List of detected emotions
    keywords = db.Column(JSON)  # List of keywords
    analysis_notes = db.Column(db.Text)  # AI analysis notes
    content_type = db.Column(db.String(100), default='general')
    visual_style = db.Column(db.String(100), default='clean')


    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<TranscriptSegment {self.id}: {self.text[:50]}>'


class YouTubeCredentials(db.Model):
    """YouTube OAuth credentials model"""
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(120), nullable=False)

    # OAuth tokens
    access_token = db.Column(db.Text, nullable=False)
    refresh_token = db.Column(db.Text)
    token_expires = db.Column(db.DateTime)

    # Channel information
    channel_id = db.Column(db.String(100), nullable=False)
    channel_title = db.Column(db.String(200))

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<YouTubeCredentials {self.user_email}: {self.channel_title}>'


class VideoUpload(db.Model):
    """Video upload tracking model"""
    id = db.Column(db.Integer, primary_key=True)
    short_id = db.Column(db.Integer, db.ForeignKey('video_short.id'), nullable=False)

    # Upload details
    channel_id = db.Column(db.String(100), nullable=False)
    channel_title = db.Column(db.String(200))
    youtube_video_id = db.Column(db.String(100))

    # Upload status
    upload_status = db.Column(db.Enum(UploadStatus), default=UploadStatus.PENDING)
    upload_progress = db.Column(db.Integer, default=0)
    upload_error = db.Column(db.Text)

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<VideoUpload {self.id}: {self.channel_title}>'


# Keep the simple test models for compatibility
class User(db.Model):
    """Simple user model for testing"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<User {self.username}>'


class TestLog(db.Model):
    """Log model for tracking test activities"""
    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.Text, nullable=False)
    endpoint = db.Column(db.String(100))
    ip_address = db.Column(db.String(45))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<TestLog {self.id}: {self.message[:50]}>'