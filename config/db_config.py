try:
    import psycopg2
except ModuleNotFoundError:
    import psycopg2_binary as psycopg2  # Fix incorrect import

from config.settings import DATABASE_CONFIG

def get_db_connection():
    return psycopg2.connect(**DATABASE_CONFIG)