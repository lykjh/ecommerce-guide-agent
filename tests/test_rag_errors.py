import json
import unittest
from unittest.mock import MagicMock, PropertyMock, patch, mock_open
from agent.rag.retriever import Retriever
from agent.rag.build_db import VectorStoreService


class RagErrorsTests(unittest.TestCase):
    def test_collection_count_failure_propagates(self):
        with patch("agent.rag.retriever.DashScopeEmbeddings"), patch("agent.rag.retriever.Chroma") as chroma:
            chroma.return_value._collection.count.side_effect = RuntimeError("database unavailable")
            with self.assertRaises(RuntimeError):
                Retriever()

    def test_metadata_failure_is_not_assumed_l2(self):
        retriever = Retriever.__new__(Retriever)
        retriever.db = MagicMock()
        type(retriever.db._collection).metadata = PropertyMock(side_effect=RuntimeError("metadata unavailable"))
        with self.assertRaises(RuntimeError):
            retriever._metric_space()

    def test_bad_record_does_not_trigger_full_reembedding(self):
        service = VectorStoreService.__new__(VectorStoreService)
        service.hash_file_path = "unused.json"
        with patch("agent.rag.build_db.os.path.exists", return_value=True), patch("builtins.open", mock_open(read_data="invalid")):
            with self.assertRaises(json.JSONDecodeError):
                service._load_md5_record()

    def test_write_failure_stops_without_recording_success(self):
        service = VectorStoreService.__new__(VectorStoreService)
        service.data_path, service.collection_name, service.persist_directory = "unused", "test", "unused"
        service.embeddings = MagicMock()
        service.get_all_files = MagicMock(return_value=["sample.txt"])
        service._allowed_types = MagicMock(return_value=(".txt",))
        service._load_md5_record = MagicMock(return_value={})
        service.get_file_md5 = MagicMock(return_value="hash")
        service._save_md5_record = MagicMock()
        service.splitter = MagicMock()
        service.splitter.split_documents.return_value = [MagicMock()]
        with patch("agent.rag.build_db.Chroma") as chroma, patch("agent.rag.build_db.TextLoader"):
            chroma.return_value.add_documents.side_effect = RuntimeError("write failed")
            with self.assertRaises(RuntimeError):
                service.build_db()
        service._save_md5_record.assert_not_called()
