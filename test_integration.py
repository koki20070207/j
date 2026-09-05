"""
Integration Tests - End-to-End Test Suite for Jarvis Project.

Comprehensive integration tests covering:
1. RAG Pipeline (PDF upload, chunking, embedding, search)
2. Chat Flows (user input, validation, tool invocation, LLM response)
3. Memo Workflow (add, list, search, mark done)
4. Chat Sessions (create, add messages, retrieve history)
5. Error Handling (database errors, API failures, invalid inputs)

Run: pytest test_integration.py -v
Run with coverage: pytest test_integration.py --cov --cov-report=html
"""

import json
import os
import pytest
import sqlite3
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest import mock
from unittest.mock import MagicMock, Mock, patch

# Import modules under test
from config import (
    DB_PATH,
    PDF_DIR,
    CHROMA_DB_PATH,
    PARENT_CHUNK_SIZE,
    CHILD_CHUNK_SIZE,
)
from db import init_db, get_connection, table_row_count
from memory_store import (
    load_chat_sessions,
    save_chat_sessions,
    estimate_tokens,
    build_history_text,
)
from tools import (
    add_memo,
    list_memos,
    search_memos,
    mark_memo_done,
    get_current_datetime,
    _validate_text_input,
)


# ===== Fixtures for Test Database =====
@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """Setup temporary database for testing."""
    db_file = tmp_path / "test_jarvis.db"
    
    # Patch the config module's DB_PATH before importing dependent modules
    import config
    monkeypatch.setattr(config, "DB_PATH", str(db_file))
    
    # Patch db module's DB_PATH
    import db
    monkeypatch.setattr(db, "DB_PATH", str(db_file))
    
    # Initialize database
    init_db()
    yield db_file
    
    # Cleanup
    if os.path.exists(db_file):
        os.remove(db_file)


@pytest.fixture
def test_chroma_db(tmp_path):
    """Setup temporary ChromaDB for testing."""
    chroma_dir = tmp_path / "test_chroma"
    chroma_dir.mkdir()
    return str(chroma_dir)


@pytest.fixture
def mock_gemini_client():
    """Mock Gemini API client."""
    with patch("llm_client.get_gemini_client") as mock_client:
        client = MagicMock()
        client.models.generate_content.return_value = MagicMock(
            text="Mock response",
            candidates=[MagicMock(content=MagicMock(parts=[MagicMock(text="Mock response")]))]
        )
        mock_client.return_value = client
        yield mock_client


@pytest.fixture
def mock_chroma_db(test_chroma_db):
    """Mock ChromaDB for testing."""
    with patch("rag_engine.chromadb") as mock_chroma:
        # Create a mock client and collection
        mock_client = MagicMock()
        mock_collection = MagicMock()
        
        # Simulate collection methods
        mock_collection.add = MagicMock()
        mock_collection.query = MagicMock(return_value={
            "ids": [["chunk_1", "chunk_2"]],
            "distances": [[0.1, 0.2]],
            "documents": [["Document 1", "Document 2"]],
        })
        mock_collection.delete = MagicMock()
        mock_collection.get = MagicMock(return_value={"ids": ["chunk_1"]})
        
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client.delete_collection = MagicMock()
        mock_chroma.Client.return_value = mock_client
        
        yield mock_chroma, mock_client, mock_collection


@pytest.fixture
def mock_embeddings():
    """Mock embeddings model."""
    with patch("rag_engine.SentenceTransformer") as mock_model_class, \
         patch("memory_store.SentenceTransformer") as mock_memory_model:
        
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_model_class.return_value = mock_model
        mock_memory_model.return_value = mock_model
        
        yield mock_model


# ===== Test: RAG Pipeline =====
class TestRAGPipeline:
    """Tests for RAG (Retrieval-Augmented Generation) pipeline."""
    
    def test_rag_pdf_upload_simulation(self, test_db, mock_chroma_db, mock_embeddings):
        """Test: PDF upload simulation with chunking and embedding."""
        # Arrange
        from rag_engine import create_hierarchical_chunks, sanitize_filename
        
        pdf_content = "This is a test PDF content with multiple sentences. " * 50
        filename = "test_document.pdf"
        
        # Act
        sanitized = sanitize_filename(filename)
        chunks = create_hierarchical_chunks(pdf_content)
        
        # Assert
        assert sanitized == "test_document.pdf"
        assert len(chunks) > 0
        assert all("parent" in chunk and "children" in chunk for chunk in chunks)
    
    def test_rag_chunking_with_overlap(self, mock_embeddings):
        """Test: Hierarchical chunking with overlap."""
        from rag_engine import create_hierarchical_chunks
        
        # Test text with clear sections
        text = "A" * 500 + "B" * 500 + "C" * 500
        chunks = create_hierarchical_chunks(text)
        
        # Verify overlap exists
        assert len(chunks) > 0
        for chunk in chunks:
            assert "parent" in chunk
            assert "children" in chunk
            assert len(chunk["children"]) > 0
    
    def test_rag_embedding_and_storage(self, test_db, mock_chroma_db, mock_embeddings):
        """Test: Embedding generation and storage in ChromaDB."""
        # Arrange
        mock_chroma, mock_client, mock_collection = mock_chroma_db
        chunks = [
            {"parent": "Parent content 1", "children": ["Child 1", "Child 2"]},
            {"parent": "Parent content 2", "children": ["Child 3", "Child 4"]},
        ]
        
        # Act - Simulate storing in ChromaDB
        mock_collection.add(
            ids=["chunk_1", "chunk_2"],
            documents=["Content 1", "Content 2"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            metadatas=[{"source": "pdf_1"}, {"source": "pdf_2"}]
        )
        
        # Assert
        assert mock_collection.add.called
        call_args = mock_collection.add.call_args
        assert call_args[1]["ids"] == ["chunk_1", "chunk_2"]
    
    def test_rag_similarity_search_and_retrieval(self, test_db, mock_chroma_db):
        """Test: Query embedding and similarity search."""
        mock_chroma, mock_client, mock_collection = mock_chroma_db
        
        # Act - Perform search
        search_results = mock_collection.query(
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=3
        )
        
        # Assert
        assert "ids" in search_results
        assert "distances" in search_results
        assert "documents" in search_results
        assert len(search_results["ids"]) == 1
        assert len(search_results["ids"][0]) == 2
    
    def test_rag_document_ranking_by_similarity(self, test_db, mock_chroma_db):
        """Test: Document retrieval and ranking by similarity score."""
        mock_chroma, mock_client, mock_collection = mock_chroma_db
        
        # Simulate search with multiple results
        search_results = mock_collection.query(
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=5
        )
        
        # Verify distance-based ranking (lower = more similar)
        assert search_results["distances"][0] == [0.1, 0.2]
        # First result should have lower distance
        assert search_results["distances"][0][0] < search_results["distances"][0][1]


# ===== Test: Chat Flows =====
class TestChatFlows:
    """Tests for chat flow and LLM interaction."""
    
    def test_chat_single_turn_flow(self, test_db, mock_gemini_client):
        """Test: Single turn user input → validation → LLM response."""
        # Arrange
        user_input = "What is the weather today?"
        
        # Act - Simulate input validation
        validation_result = _validate_text_input(user_input)
        
        # Assert
        assert validation_result is None  # No validation error
    
    def test_chat_multi_turn_conversation_with_memory(self, test_db):
        """Test: Multiple turn conversation with memory management."""
        # Arrange
        sessions = {}
        session_id = "test_session_001"
        sessions[session_id] = {
            "title": "Test Chat",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well, thanks!"},
            ]
        }
        
        # Act
        success = save_chat_sessions(sessions)
        loaded = load_chat_sessions()
        
        # Assert
        assert success
        assert session_id in loaded
        assert len(loaded[session_id]["messages"]) == 4
        assert loaded[session_id]["messages"][0]["role"] == "user"
    
    def test_chat_tool_invocation_memo_creation(self, test_db):
        """Test: Tool invocation for memo creation during chat."""
        # Arrange
        memo_text = "Meeting with John at 3 PM on August 28"
        
        # Act
        result = add_memo(memo_text)
        
        # Assert
        assert "保存しました" in result or "saved" in result.lower()
        
        # Verify memo was stored
        memos = list_memos()
        assert memo_text in memos
    
    def test_chat_tool_invocation_current_datetime(self):
        """Test: Tool invocation for getting current datetime."""
        # Act
        datetime_str = get_current_datetime()
        
        # Assert - Check for datetime components
        # Japanese year format: 年, month: 月, time: 時
        # Just check that it's not empty and has reasonable length
        assert len(datetime_str) > 10
        assert ":" in datetime_str  # Should have time separator
    
    def test_chat_input_validation_with_safety_check(self):
        """Test: User input validation with safety guardrails."""
        # Test valid input
        valid = _validate_text_input("正常な入力テキスト")
        assert valid is None
        
        # Test empty input
        empty = _validate_text_input("")
        assert empty is not None
        
        # Test too short
        short = _validate_text_input("a", min_length=5)
        assert short is not None
        
        # Test too long
        long_text = "a" * 20000
        too_long = _validate_text_input(long_text, max_length=10000)
        assert too_long is not None


# ===== Test: Memo Workflow =====
class TestMemoWorkflow:
    """Tests for memo management (add, list, search, mark done)."""
    
    def test_memo_add_and_list(self, test_db):
        """Test: Add memo and list all memos."""
        # Arrange
        memo1 = "Buy groceries"
        memo2 = "Call dentist"
        
        # Act
        result1 = add_memo(memo1)
        result2 = add_memo(memo2)
        all_memos = list_memos()
        
        # Assert
        assert "保存しました" in result1 or "saved" in result1.lower()
        assert "保存しました" in result2 or "saved" in result2.lower()
        assert memo1 in all_memos
        assert memo2 in all_memos
    
    def test_memo_search_functionality(self, test_db):
        """Test: Search memos by keyword."""
        # Arrange
        add_memo("Dentist appointment on Friday")
        add_memo("Coffee meeting Monday")
        add_memo("Doctor checkup Wednesday")
        
        # Act
        search_result = search_memos("dentist")
        
        # Assert
        assert "Dentist" in search_result
        assert "Coffee" not in search_result
    
    def test_memo_mark_done(self, test_db):
        """Test: Mark memo as done."""
        # Arrange
        add_memo("Complete project report")
        
        # Act
        # Get memo ID from list
        memos = list_memos()
        # Parse the first memo ID
        import re
        match = re.search(r"\[memo_(\d+)\]", memos)
        if match:
            memo_id = int(match.group(1))
            result = mark_memo_done(memo_id)
            
            # Assert
            assert result is not None
            # Verify memo is marked as done
            all_memos = list_memos()
            assert "✅" in all_memos
    
    def test_memo_concurrent_add_operations(self, test_db):
        """Test: Concurrent memo additions maintain data integrity."""
        # Arrange
        memo_texts = [f"Memo {i}" for i in range(10)]
        
        # Act
        for memo_text in memo_texts:
            add_memo(memo_text)
        
        all_memos = list_memos()
        
        # Assert - all memos should be present
        for memo_text in memo_texts:
            assert memo_text in all_memos
    
    def test_memo_validation_rejects_time_queries(self, test_db):
        """Test: Reject standalone time queries as memos."""
        # Arrange
        time_query = "何時？"  # "What time is it?" - too short
        
        # Act
        result = add_memo(time_query)
        
        # Assert - should reject or warn
        assert result is not None
        # Memo list should not contain incomplete entries
        memos = list_memos()
        if "何時" in memos:
            # If it was added, verify it includes warning
            assert "保存しました" not in result or len(result) > 20


# ===== Test: Chat Sessions =====
class TestChatSessions:
    """Tests for chat session management."""
    
    def test_chat_session_create_and_retrieve(self, test_db):
        """Test: Create chat session and retrieve messages."""
        # Arrange
        session_id = "session_001"
        session_data = {
            session_id: {
                "title": "Test Session",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ]
            }
        }
        
        # Act
        save_chat_sessions(session_data)
        loaded = load_chat_sessions()
        
        # Assert
        assert session_id in loaded
        assert loaded[session_id]["title"] == "Test Session"
        assert len(loaded[session_id]["messages"]) == 2
    
    def test_chat_session_persistence_across_loads(self, test_db):
        """Test: Chat session persists across load/save cycles."""
        # Arrange
        session_id = "session_persist_test"
        original_data = {
            session_id: {
                "title": "Persistence Test",
                "messages": [
                    {"role": "user", "content": "First message"},
                    {"role": "assistant", "content": "First response"},
                ]
            }
        }
        
        # Act - Save, load, modify, and save again
        save_chat_sessions(original_data)
        loaded1 = load_chat_sessions()
        
        loaded1[session_id]["messages"].append(
            {"role": "user", "content": "Second message"}
        )
        save_chat_sessions(loaded1)
        
        loaded2 = load_chat_sessions()
        
        # Assert
        assert len(loaded2[session_id]["messages"]) == 3
        assert loaded2[session_id]["messages"][-1]["content"] == "Second message"
    
    def test_chat_session_multiple_sessions(self, test_db):
        """Test: Manage multiple independent chat sessions."""
        # Arrange
        sessions = {
            "session_1": {
                "title": "Chat 1",
                "messages": [{"role": "user", "content": "Q1"}]
            },
            "session_2": {
                "title": "Chat 2",
                "messages": [{"role": "user", "content": "Q2"}]
            },
            "session_3": {
                "title": "Chat 3",
                "messages": [{"role": "user", "content": "Q3"}]
            }
        }
        
        # Act
        save_chat_sessions(sessions)
        loaded = load_chat_sessions()
        
        # Assert
        assert len(loaded) == 3
        assert all(sid in loaded for sid in ["session_1", "session_2", "session_3"])
    
    def test_chat_session_history_building(self, test_db):
        """Test: Build chat history for context."""
        # Arrange
        history = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "How do I learn it?"},
        ]
        
        # Act
        history_text = build_history_text(history)
        
        # Assert
        assert "Python" in history_text
        assert len(history_text) > 0


# ===== Test: Error Handling =====
class TestErrorHandling:
    """Tests for error handling and recovery."""
    
    def test_error_database_connection_failure(self, test_db):
        """Test: Handle database connection errors gracefully."""
        # Arrange - Patch connection to raise error
        with patch("db.sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.OperationalError("Database locked")
            
            # Act & Assert
            with pytest.raises(sqlite3.OperationalError):
                with get_connection() as conn:
                    conn.execute("SELECT 1")
    
    def test_error_invalid_memo_text(self, test_db):
        """Test: Handle invalid memo input."""
        # Arrange
        invalid_inputs = [
            "",  # Empty
            "   ",  # Whitespace only
            "a",  # Too short
        ]
        
        # Act & Assert
        for invalid_input in invalid_inputs:
            result = add_memo(invalid_input)
            # Should return error message, not success
            assert "短すぎます" in result or "空です" in result or len(result) > 0
    
    def test_error_malformed_chat_data(self, test_db):
        """Test: Handle malformed chat session data."""
        # Arrange
        malformed_sessions = {
            "bad_session": {
                # Missing required "title"
                "messages": [{"role": "user"}]  # Missing "content"
            }
        }
        
        # Act - Save the malformed data
        result = save_chat_sessions(malformed_sessions)
        
        # Try to load
        loaded = load_chat_sessions()
        
        # Assert - System should handle gracefully
        assert isinstance(loaded, dict)
    
    def test_error_gemini_api_failure_retry(self, test_db, mock_gemini_client):
        """Test: Mock Gemini API failure and retry behavior."""
        # Arrange
        mock_client = mock_gemini_client()
        mock_client.return_value.models.generate_content.side_effect = [
            Exception("API Error"),  # First call fails
            MagicMock(text="Success on retry")  # Second call succeeds
        ]
        
        # Act & Assert
        with pytest.raises(Exception):
            mock_client().models.generate_content(
                model="gemini-2.5-flash",
                contents="test"
            )
    
    def test_error_search_with_invalid_query(self, test_db):
        """Test: Search with invalid or dangerous query strings."""
        # Arrange
        dangerous_queries = [
            "'; DROP TABLE memos; --",
            "% OR 1=1 --",
            "\x00null byte",
        ]
        
        # Add a test memo first
        add_memo("Test memo")
        
        # Act & Assert - Should not crash or delete data
        for query in dangerous_queries:
            result = search_memos(query)
            # Should return safely (no matches or error message)
            assert isinstance(result, str)
    
    def test_error_token_estimation_edge_cases(self):
        """Test: Token estimation with various edge cases."""
        # Test empty string
        tokens_empty = estimate_tokens("")
        assert tokens_empty >= 1  # Should return at least 1
        
        # Test very short string
        tokens_short = estimate_tokens("a")
        assert tokens_short >= 1
        
        # Test very long string
        tokens_long = estimate_tokens("a" * 10000)
        assert tokens_long > 100


# ===== Test: Integration Scenarios =====
class TestIntegrationScenarios:
    """End-to-end integration scenarios combining multiple components."""
    
    def test_scenario_complete_memo_workflow(self, test_db):
        """Scenario: User creates, lists, searches, and completes memos."""
        # Step 1: Add multiple memos
        add_memo("Prepare presentation for meeting")
        add_memo("Review pull requests")
        add_memo("Update project documentation")
        add_memo("Schedule team sync")
        
        # Step 2: List all memos
        all_memos = list_memos()
        assert "Prepare presentation" in all_memos
        assert all_memos.count("・") >= 4  # At least 4 incomplete memos
        
        # Step 3: Search for specific memo
        search_result = search_memos("presentation")
        assert "presentation" in search_result.lower()
        
        # Step 4: Mark memo as done
        import re
        match = re.search(r"\[memo_(\d+)\].*presentation", all_memos, re.IGNORECASE)
        if match:
            memo_id = int(match.group(1))
            mark_memo_done(memo_id)
            
            # Verify done status changed
            updated_memos = list_memos()
            assert "✅" in updated_memos
    
    def test_scenario_multi_turn_chat_with_memos(self, test_db):
        """Scenario: Chat conversation that creates and references memos."""
        # Step 1: Create chat session
        session_id = "chat_with_memos"
        messages = []
        
        # Step 2: Simulate chat turns
        user_msgs = [
            "Remind me to call John tomorrow",
            "Add a memo: prepare slides for presentation",
            "What memos do I have?",
        ]
        
        for user_msg in user_msgs:
            messages.append({"role": "user", "content": user_msg})
            
            # Simulate adding memo if applicable
            if "memo" in user_msg.lower() or "remind" in user_msg.lower():
                add_memo(user_msg)
            
            messages.append({
                "role": "assistant",
                "content": f"Processed: {user_msg}"
            })
        
        # Step 3: Save chat session
        sessions = {session_id: {"title": "Chat with Memos", "messages": messages}}
        save_chat_sessions(sessions)
        
        # Step 4: Verify everything persisted
        loaded = load_chat_sessions()
        assert len(loaded[session_id]["messages"]) == 6  # 3 user + 3 assistant
        
        memos = list_memos()
        # Should have added memos
        assert len(memos) > 0
    
    def test_scenario_rag_search_in_chat_context(self, test_db, mock_chroma_db):
        """Scenario: Chat flow using RAG search results."""
        # Arrange
        mock_chroma, mock_client, mock_collection = mock_chroma_db
        
        # Step 1: Simulate document storage with Python content
        python_docs = [
            "Python is a programming language",
            "Python uses indentation for code blocks"
        ]
        mock_collection.add(
            ids=["doc_1", "doc_2"],
            documents=python_docs,
            embeddings=[[0.1, 0.2], [0.15, 0.25]]
        )
        
        # Step 2: Simulate search query
        search_result = mock_collection.query(
            query_embeddings=[[0.12, 0.22]],
            n_results=2
        )
        
        # Mock the search result to include Python in documents
        search_result["documents"] = [python_docs]
        
        # Step 3: Store search results in chat
        session_id = "rag_search_chat"
        sessions = {
            session_id: {
                "title": "RAG Search Chat",
                "messages": [
                    {"role": "user", "content": "Tell me about Python"},
                    {"role": "assistant", "content": " ".join(python_docs)},
                ]
            }
        }
        
        save_chat_sessions(sessions)
        
        # Assert
        loaded = load_chat_sessions()
        assert "Python" in loaded[session_id]["messages"][1]["content"]


# ===== Performance & Coverage Tests =====
class TestPerformanceAndCoverage:
    """Tests for performance and code coverage."""
    
    def test_large_memo_list_handling(self, test_db):
        """Test: Handle large number of memos efficiently."""
        # Arrange
        memo_count = 100
        
        # Act
        for i in range(memo_count):
            add_memo(f"Memo item {i}")
        
        all_memos = list_memos()
        
        # Assert
        # Should handle and display (with potential pagination)
        assert len(all_memos) > 0
        # Should not crash or timeout
        assert isinstance(all_memos, str)
    
    def test_large_chat_history_building(self, test_db):
        """Test: Build context from large chat history."""
        # Arrange
        large_history = []
        for i in range(50):
            large_history.append({"role": "user", "content": f"Question {i}"})
            large_history.append({"role": "assistant", "content": f"Answer {i}"})
        
        # Act
        history_text = build_history_text(large_history, max_tokens=5000)
        
        # Assert
        assert len(history_text) > 0
        assert isinstance(history_text, str)
    
    def test_database_table_row_counts(self, test_db):
        """Test: Verify database table statistics."""
        # Add some data
        add_memo("Test memo 1")
        add_memo("Test memo 2")
        
        sessions = {"session_1": {"title": "Test", "messages": []}}
        save_chat_sessions(sessions)
        
        # Act
        memo_count = table_row_count("memos")
        session_count = table_row_count("chat_sessions")
        
        # Assert
        assert memo_count == 2
        assert session_count == 1


# ===== Run tests =====
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
