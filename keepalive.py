
import requests
import time
import threading
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KeepAlive:
    def __init__(self):
        # Get the app URL - try multiple environment variables
        self.app_url = self.get_app_url()
        self.ping_interval = 300  # 5 minutes in seconds
        self.running = False
        self.thread = None
        
    def get_app_url(self):
        """Get the app URL from environment variables"""
        # Try different environment variables that Replit might use
        replit_domain = os.environ.get("REPLIT_DEV_DOMAIN")
        if replit_domain:
            return f"https://{replit_domain}"
        
        repl_url = os.environ.get("REPL_URL")
        if repl_url:
            return repl_url
        
        # Fallback to localhost for development
        return "http://localhost:5000"
    
    def ping_app(self):
        """Send a ping request to keep the app alive"""
        try:
            response = requests.get(self.app_url, timeout=30)
            if response.status_code == 200:
                logger.info(f"✅ Successfully pinged app at {self.app_url} - Status: {response.status_code}")
            else:
                logger.warning(f"⚠️ App responded with status {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to ping app: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error pinging app: {e}")
    
    def keep_alive_loop(self):
        """Main loop that pings the app at regular intervals"""
        logger.info(f"🚀 Starting keepalive service for {self.app_url}")
        logger.info(f"⏰ Pinging every {self.ping_interval} seconds ({self.ping_interval//60} minutes)")
        
        while self.running:
            self.ping_app()
            time.sleep(self.ping_interval)
    
    def start(self):
        """Start the keepalive service in a separate thread"""
        if self.running:
            logger.warning("KeepAlive service is already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self.keep_alive_loop, daemon=True)
        self.thread.start()
        logger.info("🔄 KeepAlive service started successfully")
    
    def stop(self):
        """Stop the keepalive service"""
        if not self.running:
            logger.warning("KeepAlive service is not running")
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("🛑 KeepAlive service stopped")
    
    def status(self):
        """Get the current status of the keepalive service"""
        return {
            'running': self.running,
            'app_url': self.app_url,
            'ping_interval': self.ping_interval,
            'thread_alive': self.thread.is_alive() if self.thread else False
        }

# Global keepalive instance
keepalive_service = KeepAlive()

def start_keepalive():
    """Start the keepalive service"""
    keepalive_service.start()

def stop_keepalive():
    """Stop the keepalive service"""
    keepalive_service.stop()

def get_keepalive_status():
    """Get keepalive service status"""
    return keepalive_service.status()

# If this file is run directly, start the keepalive service
if __name__ == "__main__":
    try:
        keepalive_service.start()
        
        # Keep the main thread alive
        while True:
            time.sleep(60)  # Check every minute
            if not keepalive_service.running:
                break
                
    except KeyboardInterrupt:
        logger.info("🔚 Received interrupt signal, stopping keepalive service...")
        keepalive_service.stop()
    except Exception as e:
        logger.error(f"❌ Fatal error in keepalive service: {e}")
        keepalive_service.stop()
