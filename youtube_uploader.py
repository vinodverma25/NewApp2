import os
import shutil
import logging
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from app import app, db
from models import VideoShort, YouTubeCredentials, UploadStatus, VideoUpload
from oauth_handler import OAuthHandler

logger = logging.getLogger(__name__)

class YouTubeUploader:
    def __init__(self):
        self.oauth_handler = OAuthHandler()

    def upload_short(self, short_id, user_email, channel_id=None):
        """Upload a short video to YouTube"""
        with app.app_context():
            short = VideoShort.query.get(short_id)
            if not short:
                logger.error(f"Short {short_id} not found")
                return

            db_creds = None
            try:
                logger.info(f"Starting YouTube upload for short {short_id} to channel {channel_id}")

                # Get valid credentials and channel info
                credentials_result = self._get_valid_credentials(user_email, channel_id)
                if not credentials_result or not credentials_result[0] or not credentials_result[1]:
                    raise Exception("No valid YouTube credentials found")
                
                oauth_creds, db_creds = credentials_result
                channel_title = getattr(db_creds, 'channel_title', 'Unknown Channel') or 'Unknown Channel'

                # Create or get upload record for tracking
                upload_record = self._create_upload_record(short.id, channel_id, channel_title)
                upload_record.upload_status = UploadStatus.UPLOADING
                db.session.commit()

                # Build YouTube service
                youtube = build('youtube', 'v3', credentials=oauth_creds)

                # Upload video
                video_id = self._upload_video(youtube, short, channel_title)

                # Update upload record on success
                upload_record.upload_status = UploadStatus.COMPLETED
                upload_record.youtube_video_id = video_id
                upload_record.upload_error = None
                upload_record.upload_progress = 100
                db.session.commit()

                # Keep local video files after upload (no longer auto-deleting)
                logger.info(f"Successfully uploaded short {short_id} to YouTube channel {channel_title}: {video_id}")
                logger.info(f"Local files preserved for short {short_id} after upload")

            except Exception as e:
                logger.error(f"Failed to upload short {short_id} to channel {channel_id}: {e}")
                # Update upload record on failure
                try:
                    channel_title = getattr(db_creds, 'channel_title', 'Unknown Channel') if db_creds else 'Unknown Channel'
                    upload_record = self._get_or_create_upload_record(short.id, channel_id, channel_title)
                    upload_record.upload_status = UploadStatus.FAILED
                    upload_record.upload_error = str(e)
                    db.session.commit()
                except Exception as record_error:
                    logger.error(f"Failed to update upload record: {record_error}")

    def _create_upload_record(self, short_id, channel_id, channel_title=None):
        """Create or get upload record for tracking uploads to different channels"""
        upload_record = VideoUpload.query.filter_by(short_id=short_id, channel_id=channel_id).first()
        
        if not upload_record:
            upload_record = VideoUpload()
            upload_record.short_id = short_id
            upload_record.channel_id = channel_id
            upload_record.channel_title = channel_title
            upload_record.upload_status = UploadStatus.PENDING
            db.session.add(upload_record)
            db.session.commit()
        
        return upload_record
    
    def _get_or_create_upload_record(self, short_id, channel_id, channel_title=None):
        """Get or create upload record for tracking"""
        return self._create_upload_record(short_id, channel_id, channel_title)

    def _get_valid_credentials(self, user_email, channel_id=None):
        """Get valid YouTube credentials, refreshing if necessary"""
        try:
            if channel_id:
                db_creds = YouTubeCredentials.query.filter_by(user_email=user_email, channel_id=channel_id).first()
            else:
                db_creds = YouTubeCredentials.query.filter_by(user_email=user_email).first()

            if not db_creds:
                return None, None

            # Create OAuth2 credentials object
            oauth_creds = Credentials(
                token=db_creds.access_token,
                refresh_token=db_creds.refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=self.oauth_handler.client_id,
                client_secret=self.oauth_handler.client_secret
            )

            # Refresh if expired
            if oauth_creds.expired and oauth_creds.refresh_token:
                oauth_creds.refresh(Request())

                # Update database with new token
                db_creds.access_token = oauth_creds.token
                if oauth_creds.expiry:
                    db_creds.token_expires = oauth_creds.expiry
                db.session.commit()

                logger.info(f"Refreshed credentials for {user_email}")

            return oauth_creds, db_creds

        except Exception as e:
            logger.error(f"Failed to get valid credentials: {e}")
            return None, None

    def _upload_video(self, youtube, short, channel_title):
        """Upload video to YouTube and return video ID"""
        try:
            # Prepare video metadata
            title = short.title or f"YouTube Short #{short.id}"
            if channel_title:
                # Add channel info to description for tracking
                description = f"{short.description or 'Generated YouTube Short'}\n\nUploaded to: {channel_title}"
            else:
                description = short.description or "Generated YouTube Short"
                
            body = {
                'snippet': {
                    'title': title,
                    'description': description,
                    'tags': short.tags or ['shorts', 'viral'],
                    'categoryId': '22',  # People & Blogs
                    'defaultLanguage': 'en',
                    'defaultAudioLanguage': 'en'
                },
                'status': {
                    'privacyStatus': 'public',  # Can be 'private', 'unlisted', or 'public'
                    'madeForKids': False,
                    'selfDeclaredMadeForKids': False
                }
            }

            # Create media upload object
            media = MediaFileUpload(
                short.output_path,
                chunksize=-1,
                resumable=True,
                mimetype='video/mp4'
            )

            # Insert video
            insert_request = youtube.videos().insert(
                part=','.join(body.keys()),
                body=body,
                media_body=media
            )

            # Execute the upload
            response = insert_request.execute()
            
            video_id = response['id']
            logger.info(f"Video uploaded successfully. Video ID: {video_id}")
            
            return video_id

        except Exception as e:
            logger.error(f"Failed to upload video: {e}")
            raise e

    def bulk_upload(self, short_id, user_email):
        """Upload a short to all connected YouTube accounts"""
        with app.app_context():
            short = VideoShort.query.get(short_id)
            if not short:
                logger.error(f"Short {short_id} not found")
                return

            # Get all accounts for user
            accounts = YouTubeCredentials.query.filter_by(user_email=user_email).all()
            
            if not accounts:
                logger.error(f"No YouTube accounts found for user {user_email}")
                return

            # Start uploads to all accounts
            upload_count = 0
            for account in accounts:
                try:
                    logger.info(f"Starting upload to channel {account.channel_title or account.channel_id}")
                    self.upload_short(short_id, user_email, account.channel_id)
                    upload_count += 1
                except Exception as e:
                    logger.error(f"Failed to upload to {account.channel_title}: {e}")

            logger.info(f"Bulk upload completed: {upload_count} uploads initiated")

    def _cleanup_local_files(self, short):
        """Delete local video files after successful upload to save storage space"""
        import os
        try:
            # Delete the main video file
            if short.output_path and os.path.exists(short.output_path):
                os.remove(short.output_path)
                logger.info(f"Deleted video file: {short.output_path}")
            
            # Delete the thumbnail file  
            if short.thumbnail_path and os.path.exists(short.thumbnail_path):
                os.remove(short.thumbnail_path)
                logger.info(f"Deleted thumbnail file: {short.thumbnail_path}")
                
            # Update database to clear file paths since files are deleted
            short.output_path = None
            short.thumbnail_path = None
            db.session.commit()
            
        except Exception as e:
            logger.error(f"Failed to cleanup local files for short {short.id}: {e}")

    def _should_cleanup_files(self, short_id):
        """Check if all uploads for this short are completed and we can safely delete local files"""
        uploads = VideoUpload.query.filter_by(short_id=short_id).all()
        if not uploads:
            return False
            
        # Only cleanup if at least one upload succeeded
        completed_uploads = [u for u in uploads if u.upload_status == UploadStatus.COMPLETED]
        return len(completed_uploads) > 0