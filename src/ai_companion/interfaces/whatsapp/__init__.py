"""
Init Module is responsible for:

1. Setting up the postgres database.
"""
import logging

from langgraph.checkpoint.postgres import PostgresSaver

from ai_companion.settings import settings

logger = logging.getLogger(__name__)

with PostgresSaver.from_conn_string(
    settings.DB_URI,
) as checkpointer:
    checkpointer.setup()
    logger.info("Postgres database setup complete.")
