from flask import render_template, request, redirect, url_for, flash, jsonify, send_file, session
from app import app, db
from models import VideoJob, VideoShort, TranscriptSegment, YouTubeCredentials, ProcessingStatus, UploadStatus, VideoUpload, User, TestLog
# Import YouTube components with fallback
try:
    from oauth_handler import OAuthHandler
    OAUTH_AVAILABLE = True
except ImportError as e:
    print(f"OAuth handler not available: {e}")
    OAUTH_AVAILABLE = False
    OAuthHandler = None

try:
    from youtube_uploader import YouTubeUploader
    UPLOADER_AVAILABLE = True
except ImportError as e:
    print(f"YouTube uploader not available: {e}")
    UPLOADER_AVAILABLE = False
    YouTubeUploader = None
# Import video effects processor with fallback
try:
    from video_effects_processor import VideoEffectsProcessor
    EFFECTS_AVAILABLE = True
except ImportError as e:
    print(f"Video effects processor not available: {e}")
    EFFECTS_AVAILABLE = False
    VideoEffectsProcessor = None

# Import video processor with fallback
try:
    from video_processor import VideoProcessor
    VIDEO_PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"Video processor not available: {e}")
    VIDEO_PROCESSOR_AVAILABLE = False
    VideoProcessor = None
import threading
import os
import re
import shutil
import subprocess
import time
import cv2
import numpy as np
from datetime import datetime
from keepalive import get_keepalive_status


def is_valid_youtube_url(url):
    """Validate if the URL is a valid YouTube URL"""
    youtube_regex = re.compile(
        r'(https?://)?(www\.)?(youtube\.com/(watch\?v=|embed/|v/)|youtu\.be/|m\.youtube\.com/watch\?v=)[\w-]+'
    )
    return bool(youtube_regex.match(url))


def log_activity(message, endpoint=None):
    """Helper function to log activity to database"""
    try:
        log_entry = TestLog(message=message,
                            endpoint=endpoint or request.endpoint,
                            ip_address=request.remote_addr)
        db.session.add(log_entry)
        db.session.commit()
    except Exception as e:
        app.logger.error(f'Failed to log activity: {e}')


@app.route('/')
def index():
    """Main landing page for YouTube Shorts Generator"""
    log_activity('Main page accessed')

    # Check if user has connected YouTube account
    user_email = session.get('user_email')
    youtube_connected = False
    if user_email:
        youtube_creds = YouTubeCredentials.query.filter_by(
            user_email=user_email).first()
        youtube_connected = youtube_creds is not None

    # Get recent jobs count for display
    try:
        recent_jobs = VideoJob.query.count()
        total_shorts = VideoShort.query.count()
    except Exception as e:
        app.logger.error(f'Database query error: {e}')
        recent_jobs = 0
        total_shorts = 0

    return render_template('index.html',
                           youtube_connected=youtube_connected,
                           recent_jobs=recent_jobs,
                           total_shorts=total_shorts)


@app.route('/submit', methods=['POST'])
def submit_video():
    """Submit a YouTube video for processing"""
    youtube_url = request.form.get('youtube_url', '').strip()
    language = request.form.get('language', 'hinglish')
    video_quality = request.form.get('video_quality', '720p')
    try:
        short_length = int(request.form.get('short_length', 60))
        num_shorts = int(request.form.get('num_shorts', 3))
    except (ValueError, TypeError):
        short_length = 60
        num_shorts = 3

    if not youtube_url:
        flash('Please enter a YouTube URL', 'error')
        return redirect(url_for('index'))

    if not is_valid_youtube_url(youtube_url):
        flash('Please enter a valid YouTube URL', 'error')
        return redirect(url_for('index'))

    # Validate short length and count
    if short_length < 15 or short_length > 120:
        flash('Short length must be between 15 and 120 seconds', 'error')
        return redirect(url_for('index'))

    if num_shorts < 1 or num_shorts > 10:
        flash('Number of shorts must be between 1 and 10', 'error')
        return redirect(url_for('index'))

    # Create new video job
    job = VideoJob(youtube_url=youtube_url,
                   language=language,
                   video_quality=video_quality,
                   short_length=short_length,
                   num_shorts=num_shorts,
                   status=ProcessingStatus.PENDING,
                   progress=0)
    db.session.add(job)
    db.session.commit()

    log_activity(f'Video processing job created: {youtube_url}', '/submit')

    flash('Video processing started! Check the Jobs page for progress.',
          'success')

    # Start processing in background thread
    user_email = session.get('user_email')
    if user_email:
        session['user_email'] = user_email

    thread = threading.Thread(target=process_video_async, args=(job.id, ))
    thread.daemon = True
    thread.start()

    return redirect(url_for('list_jobs'))


def process_video_async(job_id):
    """Process video in background thread"""
    try:
        if not VIDEO_PROCESSOR_AVAILABLE:
            with app.app_context():
                job = VideoJob.query.get(job_id)
                if job:
                    job.status = ProcessingStatus.FAILED
                    job.error_message = "Video processor not available"
                    db.session.commit()
            return

        processor = VideoProcessor()
        processor.process_video(job_id)
    except Exception as e:
        app.logger.error(f'Video processing failed: {e}')
        # Update job status to failed
        try:
            with app.app_context():
                job = VideoJob.query.get(job_id)
                if job:
                    job.status = ProcessingStatus.FAILED
                    job.error_message = str(e)
                    db.session.commit()
        except Exception as db_error:
            app.logger.error(f'Failed to update job status: {db_error}')


@app.route('/jobs')
def list_jobs():
    """List all processing jobs"""
    log_activity('Jobs page accessed')
    try:
        jobs = VideoJob.query.order_by(VideoJob.created_at.desc()).all()

        # Get processing status counts
        status_counts = {
            ProcessingStatus.PENDING:
            VideoJob.query.filter_by(status=ProcessingStatus.PENDING).count(),
            ProcessingStatus.DOWNLOADING:
            VideoJob.query.filter_by(
                status=ProcessingStatus.DOWNLOADING).count(),
            ProcessingStatus.TRANSCRIBING:
            VideoJob.query.filter_by(
                status=ProcessingStatus.TRANSCRIBING).count(),
            ProcessingStatus.ANALYZING:
            VideoJob.query.filter_by(
                status=ProcessingStatus.ANALYZING).count(),
            ProcessingStatus.EDITING:
            VideoJob.query.filter_by(status=ProcessingStatus.EDITING).count(),
            ProcessingStatus.COMPLETED:
            VideoJob.query.filter_by(
                status=ProcessingStatus.COMPLETED).count(),
            ProcessingStatus.FAILED:
            VideoJob.query.filter_by(status=ProcessingStatus.FAILED).count()
        }
    except Exception as e:
        app.logger.error(f'Database query error in jobs listing: {e}')
        jobs = []
        status_counts = {status: 0 for status in ProcessingStatus}

    return render_template('jobs.html', jobs=jobs, status_counts=status_counts)


@app.route('/job/<int:job_id>')
def job_details(job_id):
    """Show detailed information about a specific job"""
    job = VideoJob.query.get_or_404(job_id)
    log_activity(f'Job details accessed: {job_id}')
    return jsonify({
        'id': job.id,
        'title': job.title,
        'youtube_url': job.youtube_url,
        'status': job.status.value,
        'progress': job.progress,
        'duration': job.duration,
        'language': job.language,
        'video_quality': job.video_quality,
        'short_length': job.short_length,
        'num_shorts': job.num_shorts,
        'created_at': job.created_at.isoformat(),
        'shorts_count': len(job.shorts),
        'error_message': job.error_message
    })


@app.route('/results/<int:job_id>')
def view_results(job_id):
    """View results of a processed video"""
    job = VideoJob.query.get_or_404(job_id)
    log_activity(f'Results viewed for job: {job_id}')

    if job.status != ProcessingStatus.COMPLETED:
        flash('Video processing is not yet complete', 'warning')
        return redirect(url_for('list_jobs'))

    shorts = VideoShort.query.filter_by(job_id=job_id).all()

    # Get user's YouTube accounts
    user_email = session.get('user_email')
    youtube_accounts = []
    youtube_connected = False
    if user_email:
        youtube_accounts = YouTubeCredentials.query.filter_by(
            user_email=user_email).all()
        youtube_connected = len(youtube_accounts) > 0

    # Get upload status for each short
    upload_status = {}
    for short in shorts:
        upload_status[short.id] = {}
        uploads = VideoUpload.query.filter_by(short_id=short.id).all()
        for upload in uploads:
            upload_status[short.id][upload.channel_id] = {
                'status': upload.upload_status,
                'progress': upload.upload_progress,
                'error': upload.upload_error,
                'youtube_video_id': upload.youtube_video_id,
                'channel_title': upload.channel_title
            }

    return render_template('results.html',
                           job=job,
                           shorts=shorts,
                           youtube_accounts=youtube_accounts,
                           youtube_connected=youtube_connected,
                           upload_status=upload_status)


@app.route('/download/<int:short_id>')
def download_short(short_id):
    """Download a generated short video"""
    short = VideoShort.query.get_or_404(short_id)
    log_activity(f'Short downloaded: {short_id}')

    if not short.output_path or not os.path.exists(short.output_path):
        flash('Video file not found', 'error')
        return redirect(url_for('list_jobs'))

    return send_file(short.output_path, as_attachment=True)


@app.route('/upload/<int:short_id>/<channel_id>', methods=['POST'])
def upload_to_youtube(short_id, channel_id):
    """Upload a short to specific YouTube channel"""
    short = VideoShort.query.get_or_404(short_id)
    user_email = session.get('user_email')

    if not user_email:
        flash('Please sign in to upload to YouTube', 'error')
        return redirect(url_for('youtube_auth'))

    # Verify user has credentials for this channel
    creds = YouTubeCredentials.query.filter_by(user_email=user_email,
                                               channel_id=channel_id).first()

    if not creds:
        flash('YouTube channel not found or not authorized', 'error')
        return redirect(url_for('youtube_accounts'))

    # Start upload in background
    uploader = YouTubeUploader()
    thread = threading.Thread(target=uploader.upload_short,
                              args=(short_id, user_email, channel_id))
    thread.daemon = True
    thread.start()

    log_activity(
        f'YouTube upload started: short {short_id} to channel {channel_id}')
    flash(f'Upload started to {creds.channel_title}!', 'success')

    return redirect(url_for('view_results', job_id=short.job_id))


@app.route('/upload_all/<int:short_id>', methods=['POST'])
def upload_to_all_channels(short_id):
    """Upload a short to all connected YouTube channels"""
    short = VideoShort.query.get_or_404(short_id)
    user_email = session.get('user_email')

    if not user_email:
        flash('Please sign in to upload to YouTube', 'error')
        return redirect(url_for('youtube_auth'))

    # Get all user's YouTube accounts
    accounts = YouTubeCredentials.query.filter_by(user_email=user_email).all()

    if not accounts:
        flash('No YouTube accounts connected', 'error')
        return redirect(url_for('youtube_accounts'))

    # Create upload records and start uploads
    uploader = YouTubeUploader()
    upload_count = 0

    for account in accounts:
        # Check if already uploaded to this channel
        existing_upload = VideoUpload.query.filter_by(
            short_id=short_id, channel_id=account.channel_id).first()

        if existing_upload and existing_upload.upload_status == UploadStatus.COMPLETED:
            continue  # Skip if already uploaded successfully

        # Start upload in background thread
        thread = threading.Thread(target=uploader.upload_short,
                                  args=(short_id, user_email,
                                        account.channel_id))
        thread.daemon = True
        thread.start()
        upload_count += 1

    log_activity(
        f'Bulk upload started: short {short_id} to {upload_count} channels')

    if upload_count > 0:
        flash(f'Upload started to {upload_count} YouTube channels!', 'success')
    else:
        flash('No new uploads needed - already uploaded to all channels',
              'info')

    return redirect(url_for('view_results', job_id=short.job_id))


@app.route('/youtube/auth')
def youtube_auth():
    """Initiate YouTube OAuth flow"""
    log_activity('YouTube auth initiated')

    if not OAUTH_AVAILABLE:
        flash(
            'YouTube authentication is not available. OAuth handler module not found.',
            'error')
        return redirect(url_for('index'))

    try:
        oauth_handler = OAuthHandler()
        auth_url = oauth_handler.get_authorization_url()
        return redirect(auth_url)
    except Exception as e:
        flash(
            f'YouTube authentication setup failed: {str(e)}. Please configure YouTube API credentials.',
            'error')
        return redirect(url_for('index'))


@app.route('/youtube/callback')
def youtube_callback():
    """Handle YouTube OAuth callback"""
    code = request.args.get('code')
    error = request.args.get('error')

    if error:
        flash(f'Authentication failed: {error}', 'error')
        return redirect(url_for('index'))

    if not code:
        flash('No authorization code received', 'error')
        return redirect(url_for('index'))

    try:
        oauth_handler = OAuthHandler()
        state = request.args.get('state')
        result = oauth_handler.exchange_code_for_tokens(code, state)

        if not result:
            flash('Failed to get YouTube credentials', 'error')
            return redirect(url_for('index'))

        user_email = result['email']
        access_token = result['access_token']
        channel_info = result.get('channel_info')

        # Store user email in session
        session['user_email'] = user_email

        # Get channel ID from channel info or create default
        channel_id = channel_info.get(
            'id'
        ) if channel_info else f"default_{user_email.replace('@', '_').replace('.', '_')}"
        channel_title = channel_info.get('snippet', {}).get(
            'title') if channel_info else f"YouTube Account ({user_email})"

        # Check if credentials already exist for this channel
        existing_creds = YouTubeCredentials.query.filter_by(
            user_email=user_email, channel_id=channel_id).first()

        if existing_creds:
            # Update existing credentials
            existing_creds.access_token = access_token
            existing_creds.channel_title = channel_title
        else:
            # Create new credentials
            new_creds = YouTubeCredentials(user_email=user_email,
                                           access_token=access_token,
                                           refresh_token=result.get(
                                               'refresh_token', ''),
                                           channel_id=channel_id,
                                           channel_title=channel_title)
            db.session.add(new_creds)

        db.session.commit()
        log_activity(f'YouTube account connected: {channel_title}')
        flash(f'Successfully connected YouTube channel: {channel_title}',
              'success')

    except Exception as e:
        flash(f'Failed to save YouTube credentials: {str(e)}', 'error')
        return redirect(url_for('index'))

    return redirect(url_for('youtube_accounts'))


@app.route('/youtube/accounts')
def youtube_accounts():
    """Show connected YouTube accounts"""
    user_email = session.get('user_email')

    if not user_email:
        flash('Please connect a YouTube account first', 'error')
        return redirect(url_for('youtube_auth'))

    accounts = YouTubeCredentials.query.filter_by(user_email=user_email).all()
    log_activity('YouTube accounts page accessed')

    return render_template('youtube_accounts.html', accounts=accounts)


@app.route('/youtube/set-primary/<int:account_id>', methods=['POST'])
def set_primary_account(account_id):
    """Set a YouTube account as primary"""
    user_email = session.get('user_email')

    if not user_email:
        flash('Please sign in first', 'error')
        return redirect(url_for('youtube_auth'))

    # Get the account to set as primary
    account = YouTubeCredentials.query.filter_by(
        id=account_id, user_email=user_email).first()

    if not account:
        flash('Account not found', 'error')
        return redirect(url_for('youtube_accounts'))

    # Remove primary status from all accounts for this user
    YouTubeCredentials.query.filter_by(user_email=user_email).update(
        {'is_primary': False})

    # Set this account as primary
    account.is_primary = True
    db.session.commit()

    log_activity(f'Set primary YouTube account: {account.channel_title}')
    flash(f'Set {account.channel_title} as primary account', 'success')
    return redirect(url_for('youtube_accounts'))


@app.route('/youtube/disconnect-account/<int:account_id>', methods=['POST'])
def youtube_disconnect_account(account_id):
    """Disconnect a specific YouTube account"""
    user_email = session.get('user_email')

    if not user_email:
        flash('Please sign in first', 'error')
        return redirect(url_for('youtube_auth'))

    # Get the account to disconnect
    account = YouTubeCredentials.query.filter_by(
        id=account_id, user_email=user_email).first()

    if not account:
        flash('Account not found', 'error')
        return redirect(url_for('youtube_accounts'))

    # If removing primary account, set another as primary
    if account.is_primary:
        other_account = YouTubeCredentials.query.filter_by(
            user_email=user_email).filter(
                YouTubeCredentials.id != account_id).first()
        if other_account:
            other_account.is_primary = True

    # Revoke token with Google if we have OAuth handler
    if OAUTH_AVAILABLE:
        try:
            oauth_handler = OAuthHandler()
            oauth_handler.revoke_token(user_email, account.channel_id)
        except Exception as e:
            app.logger.warning(f'Failed to revoke token with Google: {e}')

    # Delete from database
    db.session.delete(account)
    db.session.commit()

    log_activity(f'Disconnected YouTube account: {account.channel_title}')
    flash(f'Disconnected {account.channel_title}', 'success')
    return redirect(url_for('youtube_accounts'))


@app.route('/youtube/disconnect', methods=['POST'])
def youtube_disconnect():
    """Disconnect all YouTube accounts"""
    user_email = session.get('user_email')

    if not user_email:
        flash('No user session found', 'error')
        return redirect(url_for('index'))

    # Delete all credentials for this user
    YouTubeCredentials.query.filter_by(user_email=user_email).delete()
    db.session.commit()

    # Clear session
    session.pop('user_email', None)

    log_activity('All YouTube accounts disconnected')
    flash('All YouTube accounts disconnected', 'success')
    return redirect(url_for('index'))


@app.route('/delete_job/<int:job_id>', methods=['POST'])
def delete_job(job_id):
    """Delete a processing job and all associated files"""
    job = VideoJob.query.get_or_404(job_id)

    try:
        # Delete associated files
        if job.video_path and os.path.exists(job.video_path):
            os.remove(job.video_path)
        if job.audio_path and os.path.exists(job.audio_path):
            os.remove(job.audio_path)
        if job.transcript_path and os.path.exists(job.transcript_path):
            os.remove(job.transcript_path)

        # Delete shorts and their files
        for short in job.shorts:
            if short.output_path and os.path.exists(short.output_path):
                os.remove(short.output_path)
            if short.thumbnail_path and os.path.exists(short.thumbnail_path):
                os.remove(short.thumbnail_path)

        # Delete from database (cascade will handle shorts and segments)
        db.session.delete(job)
        db.session.commit()

        log_activity(f'Job deleted: {job_id}')
        flash('Job and associated files deleted successfully', 'success')

    except Exception as e:
        flash(f'Error deleting job: {str(e)}', 'error')

    return redirect(url_for('list_jobs'))


@app.route('/jobs/delete-completed', methods=['POST'])
def delete_completed_jobs():
    """Delete all completed jobs and their associated files"""
    try:
        completed_jobs = VideoJob.query.filter(VideoJob.status == ProcessingStatus.COMPLETED).all()
        
        if not completed_jobs:
            return jsonify({'success': False, 'message': 'No completed jobs to delete'}), 400
        
        deleted_count = 0
        for job in completed_jobs:
            # Delete associated files
            if job.video_path and os.path.exists(job.video_path):
                os.remove(job.video_path)
            if job.audio_path and os.path.exists(job.audio_path):
                os.remove(job.audio_path)
            if job.transcript_path and os.path.exists(job.transcript_path):
                os.remove(job.transcript_path)

            # Delete shorts and their files
            for short in job.shorts:
                if short.output_path and os.path.exists(short.output_path):
                    os.remove(short.output_path)
                if short.thumbnail_path and os.path.exists(short.thumbnail_path):
                    os.remove(short.thumbnail_path)

            # Delete from database (cascade will handle shorts and segments)
            db.session.delete(job)
            deleted_count += 1

        db.session.commit()
        log_activity(f'Deleted {deleted_count} completed jobs')
        
        return jsonify({'success': True, 'message': f'Deleted {deleted_count} completed jobs'})

    except Exception as e:
        db.session.rollback()
        app.logger.error(f'Error deleting completed jobs: {str(e)}')
        return jsonify({'success': False, 'message': f'Error deleting completed jobs: {str(e)}'}), 500


@app.route('/cleanup', methods=['POST'])
def cleanup_storage():
    """Clean up storage by removing old files"""
    try:
        from cleanup import cleanup_old_jobs, cleanup_orphaned_files

        # Run cleanup operations
        freed_space_jobs = cleanup_old_jobs()
        freed_space_orphaned = cleanup_orphaned_files()

        total_freed = freed_space_jobs + freed_space_orphaned

        log_activity(f'Storage cleanup completed: {total_freed:.2f} MB freed')
        flash(f'Storage cleanup completed! Freed {total_freed:.2f} MB',
              'success')

    except Exception as e:
        flash(f'Cleanup failed: {str(e)}', 'error')

    return redirect(url_for('list_jobs'))


# Test endpoints (keeping for compatibility)
@app.route('/test')
def test():
    """Test endpoint for basic functionality"""
    log_activity('Test API endpoint accessed', '/test')

    return jsonify({
        'status': 'success',
        'message': 'YouTube Shorts Generator is working!',
        'method': request.method,
        'endpoint': '/test',
        'database_connected': True
    })


@app.route('/health')
def health_check():
    """Health check endpoint"""
    log_activity('Health check accessed', '/health')

    # Check database connection
    try:
        db.session.execute(db.text('SELECT 1'))
        db_status = 'connected'
    except Exception as e:
        db_status = f'error: {str(e)}'

    return jsonify({
        'status': 'healthy',
        'application': 'YouTube Shorts Generator',
        'version': '1.0.0',
        'database': db_status
    })


@app.route('/debug')
def debug_info():
    """Debug information endpoint"""
    log_activity('Debug info accessed', '/debug')

    return render_template('index.html',
                           debug_mode=True,
                           request_info={
                               'method': request.method,
                               'path': request.path,
                               'user_agent': request.headers.get('User-Agent'),
                               'remote_addr': request.remote_addr
                           })


# Simple test endpoints (keeping for compatibility)
@app.route('/users')
def list_users():
    """List all users in the database"""
    log_activity('Users list accessed', '/users')
    users = User.query.all()

    return jsonify({
        'users': [{
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'created_at': user.created_at.isoformat()
        } for user in users]
    })


@app.route('/logs')
def list_logs():
    """List recent activity logs"""
    log_activity('Logs accessed', '/logs')
    logs = TestLog.query.order_by(TestLog.timestamp.desc()).limit(50).all()

    return jsonify({
        'logs': [{
            'id': log.id,
            'message': log.message,
            'endpoint': log.endpoint,
            'ip_address': log.ip_address,
            'timestamp': log.timestamp.isoformat()
        } for log in logs]
    })


@app.route('/database/stats')
def database_stats():
    """Database statistics endpoint"""
    log_activity('Database stats accessed', '/database/stats')

    try:
        job_count = VideoJob.query.count()
        shorts_count = VideoShort.query.count()
        user_count = User.query.count()
        log_count = TestLog.query.count()

        latest_job = VideoJob.query.order_by(
            VideoJob.created_at.desc()).first()
        latest_log = TestLog.query.order_by(TestLog.timestamp.desc()).first()

        return jsonify({
            'status': 'success',
            'stats': {
                'total_jobs': job_count,
                'total_shorts': shorts_count,
                'total_users': user_count,
                'total_logs': log_count,
                'latest_job': {
                    'title': latest_job.title,
                    'status': latest_job.status.value,
                    'created_at': latest_job.created_at.isoformat()
                } if latest_job else None,
                'latest_activity': {
                    'message': latest_log.message,
                    'timestamp': latest_log.timestamp.isoformat()
                } if latest_log else None
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Database error: {str(e)}'
        }), 500


# Video Editor Routes
@app.route('/shorts/<int:short_id>/edit')
def video_editor(short_id):
    """Redirect to results page since effects are now applied automatically"""
    short = VideoShort.query.get_or_404(short_id)
    flash('Video effects are now applied automatically during processing!', 'info')
    return redirect(url_for('view_results', job_id=short.job_id))


@app.route('/api/process-frame', methods=['POST', 'OPTIONS'])
def process_frame_api():
    """API endpoint for real-time frame processing with effects and CORS support"""
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response

    if not EFFECTS_AVAILABLE:
        try:
            data = request.get_json()
            frame_data = data.get('frame_data')
            response = jsonify({
                'status':
                'success',
                'processed_frame':
                frame_data,
                'message':
                'Effects processor not available, returning original frame'
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        except Exception as e:
            response = jsonify({
                'status':
                'error',
                'message':
                'Video effects processor not available'
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 503

    try:
        start_time = time.time()
        data = request.get_json()
        frame_data = data.get('frame_data')
        effects_config = data.get('effects', {})
        timestamp = data.get('timestamp', 0)

        if not frame_data:
            return jsonify({
                'status': 'error',
                'message': 'No frame data provided'
            }), 400

        # Use cached processor for better performance
        if not hasattr(process_frame_api, 'processor'):
            process_frame_api.processor = VideoEffectsProcessor()

        processor = process_frame_api.processor

        # Fast frame decoding with canvas taint protection
        import base64
        import numpy as np
        from PIL import Image
        from io import BytesIO

        # Remove data URL prefix if present
        if frame_data.startswith('data:image'):
            frame_data = frame_data.split(',')[1]

        # Decode and convert with proper error handling
        try:
            # Clean base64 data
            frame_data = frame_data.replace(' ',
                                            '+')  # Fix any URL encoding issues

            image_data = base64.b64decode(frame_data)
            image = Image.open(BytesIO(image_data))

            # Ensure image is in RGB mode to avoid taint issues
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Convert to OpenCV format efficiently with clean array
            frame_array = np.ascontiguousarray(np.array(image))
            if len(frame_array.shape) == 3 and frame_array.shape[2] == 3:
                frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)

        except Exception as decode_error:
            response = jsonify({
                'status':
                'error',
                'message':
                f'Frame decoding failed: {str(decode_error)}'
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 400

        # Process frame with real-time optimizations
        processed_frame = processor.process_frame_realtime(
            frame_array, effects_config)

        # Convert back to base64 efficiently
        processed_base64 = processor.export_frame_as_base64(processed_frame)

        processing_time = time.time() - start_time

        response = jsonify({
            'status':
            'success',
            'processed_frame':
            processed_base64,
            'timestamp':
            timestamp,
            'processing_time_ms':
            round(processing_time * 1000, 2)
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    except Exception as e:
        app.logger.error(f'Error processing frame: {str(e)}')
        response = jsonify({
            'status': 'error',
            'message': f'Frame processing failed: {str(e)}'
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500


@app.route('/api/export-video', methods=['POST'])
def export_video_api():
    """API endpoint for exporting edited video with effects"""
    try:
        data = request.get_json()
        short_id = data.get('short_id')
        export_settings = data.get('settings', {})
        effects_timeline = data.get('effects_timeline', [])

        if not short_id:
            return jsonify({
                'status': 'error',
                'message': 'Short ID required'
            }), 400

        short = VideoShort.query.get_or_404(short_id)

        # Check if effects are applied
        if effects_timeline and EFFECTS_AVAILABLE:
            # Start background export with effects
            def export_with_effects():
                try:
                    with app.app_context():
                        processor = VideoEffectsProcessor()

                        # Create export filename with timestamp
                        import time
                        timestamp = int(time.time())
                        export_filename = f'short_{short_id}_edited_{timestamp}.mp4'
                        export_path = os.path.join('outputs', export_filename)

                        # Process video with effects using ultra-fast method
                        success = processor.export_video_ultra_fast(
                            short.output_path, export_path, effects_timeline)

                        if success:
                            # Update short record with new export path
                            updated_short = VideoShort.query.get(short_id)
                            if updated_short:
                                # Store both original and edited paths
                                if not hasattr(updated_short, 'edited_path'):
                                    updated_short.edited_path = export_path
                                db.session.commit()

                            app.logger.info(
                                f'Video export completed: {export_path}')
                        else:
                            app.logger.error(
                                f'Video export failed for short {short_id}')

                except Exception as e:
                    app.logger.error(f'Background export failed: {e}')

            # Start export in background
            export_thread = threading.Thread(target=export_with_effects)
            export_thread.daemon = True
            export_thread.start()

            return jsonify({
                'status': 'processing',
                'message':
                'Video export started. This may take a few minutes.',
                'short_id': short_id
            })
        else:
            # No effects, return original video
            return jsonify({
                'status':
                'success',
                'export_path':
                short.output_path,
                'download_url':
                url_for('download_short', short_id=short_id),
                'message':
                'Video exported successfully (no effects applied)'
            })

    except Exception as e:
        app.logger.error(f'Error exporting video: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': f'Export failed: {str(e)}'
        }), 500


@app.route('/api/upload-edited-video', methods=['POST'])
def upload_edited_video_api():
    """API endpoint for uploading edited video to YouTube"""
    try:
        data = request.get_json()
        short_id = data.get('short_id')
        upload_settings = data.get('settings', {})

        if not short_id:
            return jsonify({
                'status': 'error',
                'message': 'Short ID required'
            }), 400

        short = VideoShort.query.get_or_404(short_id)

        # Update metadata if provided
        if 'title' in upload_settings:
            short.title = upload_settings['title']
        if 'description' in upload_settings:
            short.description = upload_settings['description']
        if 'tags' in upload_settings:
            short.tags = upload_settings['tags'].split(',') if isinstance(
                upload_settings['tags'], str) else upload_settings['tags']

        db.session.commit()

        # Get connected YouTube accounts
        youtube_accounts = YouTubeCredentials.query.all()

        if not youtube_accounts:
            return jsonify({
                'status': 'error',
                'message': 'No YouTube accounts connected'
            }), 400

        # Start upload process
        uploader = YouTubeUploader()

        # Use first available account for now
        # In production, let user choose which account
        account = youtube_accounts[0]

        # Start upload in background thread
        def upload_task():
            try:
                uploader.upload_short(short_id, account.user_email,
                                      account.channel_id)
            except Exception as e:
                app.logger.error(f'Upload failed: {str(e)}')

        upload_thread = threading.Thread(target=upload_task)
        upload_thread.daemon = True
        upload_thread.start()

        return jsonify({
            'status':
            'success',
            'message':
            'Upload started. Check the jobs page for progress.',
            'upload_url':
            url_for('view_results', job_id=short.job_id)
        })

    except Exception as e:
        app.logger.error(f'Error starting upload: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': f'Upload failed: {str(e)}'
        }), 500


@app.route('/api/status/<int:job_id>')
def get_job_status(job_id):
    """Get current status of a processing job"""
    try:
        job = VideoJob.query.get_or_404(job_id)

        # Get current status text
        status_texts = {
            ProcessingStatus.PENDING:
            'Preparing to process video...',
            ProcessingStatus.DOWNLOADING:
            'Downloading video from YouTube...',
            ProcessingStatus.TRANSCRIBING:
            'Extracting and processing audio...',
            ProcessingStatus.ANALYZING:
            'Analyzing content with AI...',
            ProcessingStatus.EDITING:
            'Generating short videos...',
            ProcessingStatus.COMPLETED:
            'Processing completed successfully!',
            ProcessingStatus.FAILED:
            f'Processing failed: {job.error_message or "Unknown error"}'
        }

        shorts_count = VideoShort.query.filter_by(job_id=job_id).count()

        return jsonify({
            'status':
            job.status.value,
            'progress':
            job.progress,
            'current_status_text':
            status_texts.get(job.status, 'Processing...'),
            'title':
            job.title,
            'shorts_count':
            shorts_count,
            'error_message':
            job.error_message
        })

    except Exception as e:
        app.logger.error(f'Error getting job status: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': f'Failed to get status: {str(e)}'
        }), 500


@app.route('/api/effects-presets')
def get_effects_presets():
    """Get available effect presets and templates"""
    if not EFFECTS_AVAILABLE:
        return jsonify({
            'status': 'error',
            'message': 'Effects not available'
        }), 503

    try:
        processor = VideoEffectsProcessor()
        presets = processor.get_effect_presets()

        return jsonify({'status': 'success', 'presets': presets})

    except Exception as e:
        app.logger.error(f'Error getting presets: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': f'Failed to get presets: {str(e)}'
        }), 500


@app.route('/api/youtube/status')
def youtube_status():
    """Check YouTube connection status"""
    try:
        credentials = YouTubeCredentials.query.first()
        connected = credentials is not None

        return jsonify({
            'connected':
            connected,
            'channel_title':
            credentials.channel_title if credentials else None,
            'channel_id':
            credentials.channel_id if credentials else None
        })

    except Exception as e:
        app.logger.error(f'Error checking YouTube status: {str(e)}')
        return jsonify({'connected': False, 'error': str(e)})


@app.route('/api/upload/<int:short_id>/youtube', methods=['POST'])
def upload_short_to_youtube_api(short_id):
    """API endpoint for uploading short to YouTube"""
    try:
        data = request.get_json()
        short = VideoShort.query.get_or_404(short_id)

        # Check if YouTube credentials exist
        credentials = YouTubeCredentials.query.first()
        if not credentials:
            return jsonify({
                'status': 'error',
                'message': 'YouTube account not connected'
            }), 400

        # Start background upload
        def upload_task():
            try:
                with app.app_context():
                    from youtube_uploader import YouTubeUploader

                    uploader = YouTubeUploader()

                    # Update short metadata if provided
                    if data.get('title'):
                        short.title = data['title']
                    if data.get('description'):
                        short.description = data['description']
                    if data.get('tags'):
                        tags = data['tags'].split(',') if isinstance(
                            data['tags'], str) else data['tags']
                        short.tags = [
                            tag.strip() for tag in tags if tag.strip()
                        ]

                    db.session.commit()

                    # Upload using existing method - this handles the upload record creation internally
                    uploader.upload_short(short_id, credentials.user_email,
                                          credentials.channel_id)

            except Exception as e:
                app.logger.error(f'Background upload failed: {e}')

        # Start upload in background
        upload_thread = threading.Thread(target=upload_task)
        upload_thread.daemon = True
        upload_thread.start()

        return jsonify({
            'status': 'processing',
            'message': 'Upload started in background'
        })

    except Exception as e:
        app.logger.error(f'Error starting YouTube upload: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': f'Upload failed: {str(e)}'
        }), 500


@app.route('/api/upload/status/<int:short_id>')
def upload_status(short_id):
    """Check upload status for a short"""
    try:
        upload = VideoUpload.query.filter_by(short_id=short_id).order_by(
            VideoUpload.created_at.desc()).first()

        if not upload:
            return jsonify({
                'status': 'not_found',
                'message': 'No upload found for this short'
            })

        status_map = {
            UploadStatus.PENDING: 'pending',
            UploadStatus.UPLOADING: 'processing',
            UploadStatus.COMPLETED: 'completed',
            UploadStatus.FAILED: 'failed'
        }

        return jsonify({
            'status':
            status_map.get(upload.upload_status, 'unknown'),
            'progress':
            upload.upload_progress,
            'error':
            upload.upload_error,
            'youtube_video_id':
            upload.youtube_video_id
        })

    except Exception as e:
        app.logger.error(f'Error checking upload status: {str(e)}')
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/check-video-qualities', methods=['POST'])
def check_video_qualities():
    """Check available video qualities for a YouTube URL"""
    try:
        data = request.get_json()
        youtube_url = data.get('youtube_url')
        
        if not youtube_url:
            return jsonify({'error': 'YouTube URL is required'}), 400
        
        # Import yt-dlp for quality checking
        try:
            import yt_dlp
        except ImportError:
            return jsonify({
                'qualities': ['480p', '720p HD', '1080p Full HD', '1440p 2K', '2160p 4K'],
                'message': 'Quality detection unavailable, showing all possible options'
            })
        
        # Configure yt-dlp options with cookie support for high-quality videos
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'format': 'best',
            'cookiefile': 'cookie/youtube_cookies.txt',  # Use cookies to avoid bot detection
            'age_limit': 99,
            'skip_download': True,
            'ignoreerrors': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract video info
            info = ydl.extract_info(youtube_url, download=False)
            
            # Get available formats
            formats = info.get('formats', [])
            available_qualities = set()
            
            for fmt in formats:
                height = fmt.get('height')
                if height:
                    if height <= 480:
                        available_qualities.add('480p')
                    elif height <= 720:
                        available_qualities.add('720p HD')
                    elif height <= 1080:
                        available_qualities.add('1080p Full HD')
                    elif height <= 1440:
                        available_qualities.add('1440p 2K')
                    elif height <= 2160:
                        available_qualities.add('2160p 4K')
                    elif height > 2160:
                        available_qualities.add('4K+ Ultra HD')
            
            # Always include basic qualities if no formats found
            if not available_qualities:
                available_qualities.update(['480p', '720p HD', '1080p Full HD'])
            
            # Convert to list and sort by resolution
            quality_list = list(available_qualities)
            quality_order = {
                '480p': 1,
                '720p HD': 2,
                '1080p Full HD': 3,
                '1440p 2K': 4,
                '2160p 4K': 5,
                '4K+ Ultra HD': 6
            }
            
            quality_list.sort(key=lambda x: quality_order.get(x, 0))
            
            return jsonify({
                'qualities': quality_list,
                'message': 'Successfully checked available qualities'
            })
            
    except Exception as e:
        app.logger.error(f'Error checking video qualities: {str(e)}')
        return jsonify({
            'qualities': ['480p', '720p HD', '1080p Full HD', '1440p 2K', '2160p 4K'],
            'message': 'Error checking qualities, showing all possible options',
            'error': str(e)
        }), 200  # Return 200 to avoid breaking the UI


@app.route('/api/keepalive/status')
def keepalive_status():
    """Get the status of the keepalive service"""
    try:
        status = get_keepalive_status()
        return jsonify({
            'success':
            True,
            'status':
            status,
            'message':
            'KeepAlive service is running'
            if status['running'] else 'KeepAlive service is stopped'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to get keepalive status'
        }), 500


@app.route('/clear_all_data', methods=['POST'])
def clear_all_data():
    """Clear all app data including videos, jobs, and files"""
    try:
        # Delete all database records
        VideoUpload.query.delete()
        VideoShort.query.delete()
        TranscriptSegment.query.delete()
        VideoJob.query.delete()
        YouTubeCredentials.query.delete()
        TestLog.query.delete()
        db.session.commit()
        
        # Delete all physical files and directories
        directories_to_clear = ['uploads', 'temp', 'outputs', 'thumbnails', 'instance']
        for directory in directories_to_clear:
            if os.path.exists(directory):
                try:
                    shutil.rmtree(directory)
                    app.logger.info(f"Cleared directory: {directory}")
                except Exception as e:
                    app.logger.warning(f"Could not clear directory {directory}: {e}")
        
        # Recreate necessary directories
        for directory in ['uploads', 'temp', 'outputs', 'thumbnails']:
            os.makedirs(directory, exist_ok=True)
        
        app.logger.info("All data cleared successfully")
        flash('All data has been cleared successfully', 'success')
        return redirect(url_for('index'))
        
    except Exception as e:
        app.logger.error(f"Error clearing data: {e}")
        flash(f'Error clearing data: {str(e)}', 'error')
        return redirect(url_for('index'))
