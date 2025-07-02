from app import app
from routes import *
from keepalive import start_keepalive

if __name__ == '__main__':
    # Start the keepalive service to keep the app active 24/7
    start_keepalive()
    app.run(host='0.0.0.0', port=5000, debug=True)