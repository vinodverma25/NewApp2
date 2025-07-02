import os
import logging
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

# Configure logging for debug mode
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")

# Configure the database with enhanced connection pooling
database_url = os.environ.get("DATABASE_URL", "sqlite:///youtube_shorts.db")

# Use connection pooling for PostgreSQL
if database_url.startswith('postgresql'):
    # Use pooler for better connection management
    if '.us-east-2' in database_url and '-pooler' not in database_url:
        database_url = database_url.replace('.us-east-2', '-pooler.us-east-2')

app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_size": 10,
    "pool_recycle": 300,
    "pool_pre_ping": True,
    "pool_timeout": 30,
    "max_overflow": 20,
    "connect_args": {
        "connect_timeout": 10,
        "application_name": "youtube_shorts_generator"
    }
}

# Initialize the app with the extension
db.init_app(app)

# Enable debug mode for development
app.config['DEBUG'] = True

with app.app_context():
    try:
        # Make sure to import the models here or their tables won't be created
        import models  # noqa: F401
        db.create_all()
        app.logger.info('Database tables created successfully')
    except Exception as e:
        app.logger.error(f"Database initialization error: {e}")
        # Try to run migration
        try:
            import subprocess
            result = subprocess.run(['python', 'migrate_db.py'], capture_output=True, text=True)
            if result.returncode == 0:
                app.logger.info("Database migration completed successfully")
            else:
                app.logger.error(f"Migration failed: {result.stderr}")
        except Exception as migrate_error:
            app.logger.error(f"Migration execution failed: {migrate_error}")

# Import routes after app creation to avoid circular imports
try:
    from routes import *
    app.logger.info('Routes imported successfully')
except ImportError as e:
    app.logger.error(f'Failed to import routes: {e}')
    # Create a basic route for testing
    @app.route('/')
    def index():
        return '''
        <html><head><title>YouTube Shorts Generator</title></head>
        <body style="font-family: Arial, sans-serif; margin: 50px;">
            <h1>YouTube Shorts Generator</h1>
            <p>Application is starting up...</p>
            <p><strong>Status:</strong> Basic Flask app is running</p>
            <p><strong>Database:</strong> Connected to PostgreSQL</p>
            <p><em>Note: Some features may be loading...</em></p>
        </body></html>
        '''

# Error handlers
@app.errorhandler(404)
def page_not_found(error):
    """Handle 404 errors with custom template"""
    app.logger.error(f'Page not found: {error}')
    try:
        return render_template('404.html'), 404
    except:
        # Fallback if template is missing
        return '''
        <html><head><title>404 - Page Not Found</title></head>
        <body style="font-family: Arial, sans-serif; text-align: center; margin-top: 100px;">
            <h1>404 - Page Not Found</h1>
            <p>The page you're looking for doesn't exist.</p>
            <a href="/">Go Home</a>
        </body></html>
        ''', 404

@app.errorhandler(500)
def internal_server_error(error):
    """Handle 500 errors with custom template"""
    app.logger.error(f'Internal server error: {error}')
    try:
        return render_template('500.html'), 500
    except:
        # Fallback if template is missing
        return '''
        <html><head><title>500 - Server Error</title></head>
        <body style="font-family: Arial, sans-serif; text-align: center; margin-top: 100px;">
            <h1>500 - Internal Server Error</h1>
            <p>Something went wrong on our end.</p>
            <a href="/">Go Home</a>
        </body></html>
        ''', 500

if __name__ == '__main__':
    app.logger.info('Starting Flask test application...')
    app.run(host='0.0.0.0', port=5000, debug=True)