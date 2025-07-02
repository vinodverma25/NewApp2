
#!/usr/bin/env python3
"""
Database reset script - drops all tables and recreates them
"""
import os
from app import app, db
from models import *

def reset_database():
    """Drop all tables and recreate them"""
    with app.app_context():
        try:
            print("Starting database reset...")
            
            # Drop all tables
            print("Dropping all tables...")
            db.drop_all()
            print("All tables dropped successfully")
            
            # Recreate all tables
            print("Creating new tables...")
            db.create_all()
            print("All tables created successfully")
            
            print("Database reset completed successfully!")
            
        except Exception as e:
            print(f"Error during database reset: {e}")
            raise

if __name__ == "__main__":
    reset_database()
