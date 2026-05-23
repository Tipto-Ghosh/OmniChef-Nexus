from retrieval.db_client import get_qdrant_client, get_collection_info
client = get_qdrant_client()
get_collection_info(client)
