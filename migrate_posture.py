"""
Database migration script for adding the new posture analysis columns.

This script should be run when the model changes to add new columns to the PostureStatus table.
"""

import os
import logging
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_db_connection():
    """Create a database connection using environment variables"""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL environment variable not set")
        return None
    
    try:
        conn = psycopg2.connect(db_url)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        logger.info("Database connection established")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        return None

def check_column_exists(cursor, table, column):
    """Check if a column exists in a table"""
    cursor.execute("""
        SELECT column_name FROM information_schema.columns 
        WHERE table_name = %s AND column_name = %s
    """, (table, column))
    return cursor.fetchone() is not None

def migrate_posture_table():
    """Add new columns to the posture_status table"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Define the columns to add
        new_columns = [
            ("posture_quality", "VARCHAR(50) DEFAULT 'unknown'"),
            ("neck_angle", "FLOAT DEFAULT 0.0"),
            ("shoulder_alignment", "FLOAT DEFAULT 0.0"),
            ("head_forward_position", "FLOAT DEFAULT 0.0"),
            ("spine_curvature", "FLOAT DEFAULT 0.0"),
            ("symmetry_score", "FLOAT DEFAULT 0.0"),
            ("feedback", "VARCHAR(255)")
        ]
        
        # Check each column and add if it doesn't exist
        for column_name, column_def in new_columns:
            if not check_column_exists(cursor, "posture_status", column_name):
                logger.info(f"Adding column {column_name} to posture_status table")
                cursor.execute(f"ALTER TABLE posture_status ADD COLUMN {column_name} {column_def}")
                logger.info(f"Column {column_name} added successfully")
            else:
                logger.info(f"Column {column_name} already exists")
        
        cursor.close()
        conn.close()
        logger.info("Migration completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error during migration: {str(e)}")
        if conn:
            conn.close()
        return False

if __name__ == "__main__":
    logger.info("Starting posture table migration")
    success = migrate_posture_table()
    if success:
        logger.info("Migration completed successfully")
    else:
        logger.error("Migration failed")