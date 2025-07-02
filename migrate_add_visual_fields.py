
from app import app, db
from models import VideoShort, TranscriptSegment

def migrate_add_visual_fields():
    """Add content_type and visual_style fields to existing records"""
    with app.app_context():
        try:
            # Create tables if they don't exist
            db.create_all()
            
            # Update existing VideoShort records
            shorts = VideoShort.query.filter(
                (VideoShort.content_type == None) | 
                (VideoShort.visual_style == None)
            ).all()
            
            for short in shorts:
                if not short.content_type:
                    short.content_type = 'general'
                if not short.visual_style:
                    short.visual_style = 'clean'
            
            # Update existing TranscriptSegment records
            segments = TranscriptSegment.query.filter(
                (TranscriptSegment.content_type == None) | 
                (TranscriptSegment.visual_style == None)
            ).all()
            
            for segment in segments:
                if not segment.content_type:
                    segment.content_type = 'general'
                if not segment.visual_style:
                    segment.visual_style = 'clean'
            
            db.session.commit()
            print(f"Successfully migrated {len(shorts)} shorts and {len(segments)} segments")
            
        except Exception as e:
            print(f"Migration failed: {e}")
            db.session.rollback()

if __name__ == '__main__':
    migrate_add_visual_fields()
