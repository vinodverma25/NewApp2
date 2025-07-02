
#!/usr/bin/env python3
"""
Database migration script
"""
import os
import psycopg2
from app import app

def migrate_database():
    """Add missing columns to existing tables"""
    try:
        # Get database URL from environment
        database_url = os.environ.get('DATABASE_URL')
        if not database_url:
            print("ERROR: DATABASE_URL environment variable not set")
            return False

        print("Starting database migration...")
        
        # Connect to database
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        print("Connected to database successfully")

        # Check if video_short table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'video_short'
            );
        """)
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            print("video_short table does not exist. Creating all tables...")
            conn.close()
            
            # Use Flask-SQLAlchemy to create all tables
            with app.app_context():
                from app import db
                db.create_all()
                print("All tables created successfully")
            return True

        # Add missing columns to video_short table
        columns_to_add = [
            ("emotion_score", "FLOAT DEFAULT 0.0"),
            ("viral_potential", "FLOAT DEFAULT 0.0"), 
            ("quotability", "FLOAT DEFAULT 0.0"),
            ("overall_score", "FLOAT DEFAULT 0.0"),
            ("emotions_detected", "JSON"),
            ("keywords", "JSON"),
            ("analysis_notes", "TEXT")
        ]

        for column_name, column_def in columns_to_add:
            try:
                # Check if column exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.columns 
                        WHERE table_name = 'video_short' 
                        AND column_name = %s
                    );
                """, (column_name,))
                
                column_exists = cur.fetchone()[0]
                
                if not column_exists:
                    print(f"Adding {column_name} column...")
                    cur.execute(f"ALTER TABLE video_short ADD COLUMN {column_name} {column_def}")
                    print(f"Added {column_name} column successfully")
                else:
                    print(f"Column {column_name} already exists, skipping...")
                    
            except Exception as e:
                print(f"Error adding {column_name} column: {e}")
                continue

        # Update engagement_score to have default value if it's NULL
        print("Updating default values...")
        cur.execute("""
            UPDATE video_short 
            SET engagement_score = 0.5 
            WHERE engagement_score IS NULL
        """)
        
        cur.execute("""
            UPDATE video_short 
            SET emotion_score = 0.5 
            WHERE emotion_score IS NULL
        """)
        
        cur.execute("""
            UPDATE video_short 
            SET viral_potential = 0.5 
            WHERE viral_potential IS NULL
        """)
        
        cur.execute("""
            UPDATE video_short 
            SET quotability = 0.5 
            WHERE quotability IS NULL
        """)
        
        cur.execute("""
            UPDATE video_short 
            SET overall_score = 0.5 
            WHERE overall_score IS NULL
        """)

        # Commit changes
        conn.commit()
        cur.close()
        conn.close()

        print("Database migration completed successfully!")
        return True

    except Exception as e:
        print(f"ERROR: Database migration failed: {e}")
        return False

if __name__ == "__main__":
    success = migrate_database()
    if success:
        print("\nMigration successful!")
    else:
        print("\nMigration failed!")
