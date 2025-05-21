"""
Test Database Connection Script

This script checks if the DATABASE_URL environment variable is properly set
and attempts to connect to the database to verify connectivity.
"""

import os
import sys
from sqlalchemy import create_engine, text

# Print debugging information
print("Testing database connection...")
print(f"Python version: {sys.version}")
print(f"Environment variables: {list(os.environ.keys())}")

# Try to get the DATABASE_URL
DATABASE_URL = os.environ.get('DATABASE_URL')
print(f"DATABASE_URL found: {bool(DATABASE_URL)}")

if not DATABASE_URL:
    print("DATABASE_URL environment variable not found.")
    try:
        with open('/run/secrets/DATABASE_URL', 'r') as f:
            DATABASE_URL = f.read().strip()
            print("Retrieved DATABASE_URL from /run/secrets")
    except FileNotFoundError:
        print("Could not find DATABASE_URL in /run/secrets")
        sys.exit(1)

try:
    # Create engine and test connection
    print(f"Attempting to connect to database...")
    engine = create_engine(DATABASE_URL)
    
    # Execute simple query to test connection
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        for row in result:
            print(f"Database connection successful! Test query result: {row[0]}")
    
    print("Database connection test completed successfully!")
except Exception as e:
    print(f"Error connecting to database: {str(e)}")
    sys.exit(1)