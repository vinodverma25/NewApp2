import os
import json
import logging
import secrets
from datetime import datetime, timezone, timedelta
from urllib.parse import urlencode
import requests
from flask import session
from app import db
from models import YouTubeCredentials

logger = logging.getLogger(__name__)

class OAuthHandler:
    def __init__(self):
        self.client_id = os.environ.get("YOUTUBE_CLIENT_ID")
        self.client_secret = os.environ.get("YOUTUBE_CLIENT_SECRET")
        # Use Replit domain or fallback to localhost
        replit_domain = os.environ.get("REPLIT_DEV_DOMAIN")
        if replit_domain:
            self.redirect_uri = f"https://{replit_domain}/youtube/callback"
        else:
            # Try alternative environment variables for Replit
            app_url = os.environ.get("REPL_URL")
            if app_url:
                self.redirect_uri = f"{app_url}/youtube/callback"
            else:
                self.redirect_uri = os.environ.get("YOUTUBE_REDIRECT_URI", "http://localhost:5000/youtube/callback")
        
        if not self.client_id or not self.client_secret:
            # Set development defaults if not configured
            self.client_id = "your-client-id.apps.googleusercontent.com"
            self.client_secret = "your-client-secret"
            print("WARNING: Using default YouTube OAuth credentials. Please configure YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET environment variables for production use.")
        
        self.auth_uri = "https://accounts.google.com/o/oauth2/auth"
        self.token_uri = "https://oauth2.googleapis.com/token"
        self.userinfo_uri = "https://www.googleapis.com/oauth2/v2/userinfo"
        
        # Required scopes for YouTube upload
        self.scopes = [
            "https://www.googleapis.com/auth/youtube.upload",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile"
        ]
    
    def get_authorization_url(self):
        """Generate OAuth authorization URL"""
        # Generate and store state parameter for security
        state = secrets.token_urlsafe(32)
        session['oauth_state'] = state
        
        auth_params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'scope': ' '.join(self.scopes),
            'response_type': 'code',
            'access_type': 'offline',  # To get refresh token
            'prompt': 'consent select_account',  # Force consent and account selection
            'include_granted_scopes': 'true',  # Include previously granted scopes
            'state': state
        }
        
        auth_url = f"{self.auth_uri}?{urlencode(auth_params)}"
        logger.info(f"Generated OAuth URL for YouTube authentication")
        return auth_url
    
    def exchange_code_for_tokens(self, code, state=None):
        """Exchange authorization code for access tokens"""
        
        # Verify state parameter
        if state and session.get('oauth_state') != state:
            raise Exception("Invalid state parameter")
        
        # Clear state from session
        session.pop('oauth_state', None)
        
        # Exchange code for tokens
        token_data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': self.redirect_uri
        }
        
        try:
            # Get tokens
            response = requests.post(self.token_uri, data=token_data)
            response.raise_for_status()
            token_response = response.json()
            
            if 'error' in token_response:
                raise Exception(f"Token exchange failed: {token_response['error']}")
            
            access_token = token_response['access_token']
            refresh_token = token_response.get('refresh_token')
            expires_in = token_response.get('expires_in', 3600)
            
            # Get user info
            user_info_response = requests.get(
                self.userinfo_uri,
                headers={'Authorization': f'Bearer {access_token}'}
            )
            user_info_response.raise_for_status()
            user_info = user_info_response.json()
            
            user_email = user_info.get('email')
            if not user_email:
                raise Exception("Could not retrieve user email")
            
            # Get YouTube channel info
            channel_info = self._get_channel_info(access_token)
            
            # Calculate token expiry
            token_expires = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
            
            # Store or update credentials based on channel_id
            channel_id = channel_info.get('id') if channel_info else None
            if not channel_id:
                # If we can't get channel info, create unique identifier per auth attempt
                import time
                logger.warning("Could not retrieve YouTube channel information, creating unique identifier")
                channel_id = f"email_{user_email.replace('@', '_at_').replace('.', '_dot_')}_{int(time.time())}"
            
            existing_creds = YouTubeCredentials.query.filter_by(
                user_email=user_email, 
                channel_id=channel_id
            ).first()
            
            if existing_creds:
                # Update existing credentials
                existing_creds.access_token = access_token
                if refresh_token:  # Only update if we got a new refresh token
                    existing_creds.refresh_token = refresh_token
                existing_creds.token_expires = token_expires
                existing_creds.scope = ' '.join(self.scopes)
                existing_creds.updated_at = datetime.now(timezone.utc)
                
                if channel_info:
                    existing_creds.channel_title = channel_info.get('snippet', {}).get('title')
                    existing_creds.channel_thumbnail = channel_info.get('snippet', {}).get('thumbnails', {}).get('default', {}).get('url')
                    existing_creds.account_name = existing_creds.channel_title
                else:
                    existing_creds.channel_title = f"YouTube Account ({user_email})"
                    existing_creds.account_name = existing_creds.channel_title
                
                logger.info(f"Updated YouTube credentials for {user_email} - {existing_creds.channel_title}")
            else:
                # Create new credentials - always create new account when no existing found
                if not refresh_token:
                    logger.warning("No refresh token received - creating account anyway")
                
                # Check if this is the first account for this user
                existing_accounts = YouTubeCredentials.query.filter_by(user_email=user_email).count()
                is_primary = existing_accounts == 0
                
                new_creds = YouTubeCredentials()
                new_creds.user_email = user_email
                new_creds.access_token = access_token
                new_creds.refresh_token = refresh_token or ""  # Allow empty refresh token
                new_creds.token_expires = token_expires
                new_creds.scope = ' '.join(self.scopes)
                new_creds.is_primary = is_primary
                
                new_creds.channel_id = channel_id
                if channel_info:
                    new_creds.channel_title = channel_info.get('snippet', {}).get('title')
                    new_creds.channel_thumbnail = channel_info.get('snippet', {}).get('thumbnails', {}).get('default', {}).get('url')
                    new_creds.account_name = new_creds.channel_title
                else:
                    # Create a more unique account name when channel info is unavailable
                    account_number = existing_accounts + 1
                    new_creds.channel_title = f"YouTube Account #{account_number} ({user_email})"
                    new_creds.account_name = new_creds.channel_title
                
                try:
                    db.session.add(new_creds)
                    db.session.flush()  # Flush to catch constraint violations early
                    logger.info(f"Created new YouTube credentials for {user_email} - {new_creds.channel_title}")
                except Exception as e:
                    logger.error(f"Failed to create new credentials: {e}")
                    db.session.rollback()
                    # Try with a more unique identifier
                    import time
                    new_creds.channel_id = f"{channel_id}_{int(time.time())}"
                    db.session.add(new_creds)
                    db.session.flush()
                    logger.info(f"Created new YouTube credentials with unique ID for {user_email} - {new_creds.channel_title}")
            
            db.session.commit()
            
            return {
                'email': user_email,
                'access_token': access_token,
                'channel_info': channel_info
            }
            
        except requests.RequestException as e:
            logger.error(f"HTTP error during token exchange: {e}")
            raise Exception(f"Failed to exchange authorization code: {str(e)}")
        except Exception as e:
            logger.error(f"Error during token exchange: {e}")
            raise
    
    def _get_channel_info(self, access_token):
        """Get YouTube channel information"""
        try:
            # Use YouTube Data API to get channel info
            channel_url = "https://www.googleapis.com/youtube/v3/channels"
            params = {
                'part': 'snippet,statistics',
                'mine': 'true'
            }
            headers = {'Authorization': f'Bearer {access_token}'}
            
            response = requests.get(channel_url, params=params, headers=headers)
            
            if response.status_code == 403:
                logger.warning("YouTube Data API not enabled or insufficient permissions. Please enable YouTube Data API v3 in Google Cloud Console.")
                return None
            
            response.raise_for_status()
            
            data = response.json()
            if data.get('items'):
                return data['items'][0]
            
            logger.warning("No YouTube channel found for this account")
            return None
            
        except requests.RequestException as e:
            logger.warning(f"HTTP error getting YouTube channel info: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to get YouTube channel info: {e}")
            return None
    
    def refresh_token(self, user_email: str, channel_id: str = None):
        """Refresh access token using refresh token"""
        try:
            if channel_id:
                creds = YouTubeCredentials.query.filter_by(user_email=user_email, channel_id=channel_id).first()
            else:
                # Get primary account if no channel_id specified
                creds = YouTubeCredentials.query.filter_by(user_email=user_email, is_primary=True).first()
                if not creds:
                    # Fallback to first account
                    creds = YouTubeCredentials.query.filter_by(user_email=user_email).first()
            
            if not creds or not creds.refresh_token:
                return None
            
            token_data = {
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'refresh_token': creds.refresh_token,
                'grant_type': 'refresh_token'
            }
            
            response = requests.post(self.token_uri, data=token_data)
            response.raise_for_status()
            token_response = response.json()
            
            if 'error' in token_response:
                logger.error(f"Token refresh failed: {token_response['error']}")
                return None
            
            # Update credentials
            creds.access_token = token_response['access_token']
            expires_in = token_response.get('expires_in', 3600)
            creds.token_expires = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
            creds.updated_at = datetime.now(timezone.utc)
            
            db.session.commit()
            logger.info(f"Refreshed token for {user_email}")
            
            return creds
            
        except Exception as e:
            logger.error(f"Failed to refresh token: {e}")
            return None
    
    def revoke_token(self, user_email, channel_id=None):
        """Revoke stored tokens"""
        try:
            if channel_id:
                creds = YouTubeCredentials.query.filter_by(user_email=user_email, channel_id=channel_id).first()
            else:
                # Revoke all accounts for user
                creds_list = YouTubeCredentials.query.filter_by(user_email=user_email).all()
                for creds in creds_list:
                    try:
                        revoke_url = f"https://oauth2.googleapis.com/revoke?token={creds.access_token}"
                        requests.post(revoke_url)
                        db.session.delete(creds)
                    except Exception as e:
                        logger.warning(f"Failed to revoke token for {creds.channel_title}: {e}")
                
                db.session.commit()
                logger.info(f"Revoked all YouTube tokens for {user_email}")
                return True
            
            if not creds:
                return False
            
            # Revoke token with Google
            revoke_url = f"https://oauth2.googleapis.com/revoke?token={creds.access_token}"
            requests.post(revoke_url)
            
            # If removing primary account, set another as primary
            if creds.is_primary:
                other_creds = YouTubeCredentials.query.filter_by(user_email=user_email).filter(
                    YouTubeCredentials.id != creds.id
                ).first()
                if other_creds:
                    other_creds.is_primary = True
            
            # Delete from database
            db.session.delete(creds)
            db.session.commit()
            
            logger.info(f"Revoked YouTube tokens for {user_email} - {creds.channel_title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to revoke tokens: {e}")
            return False
