"""PalaceMiner — file mining + SessionDB session mining for the Memory Palace plugin.

Mines knowledge from project files and past sessions into Palace drawers.
Pure stdlib + PalaceStore dependency.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional

from .store import PalaceStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunking constants
# ---------------------------------------------------------------------------

CHUNK_SIZE = 800  # chars per drawer
CHUNK_OVERLAP = 100  # overlap between chunks
MIN_CHUNK_SIZE = 50  # skip tiny chunks

# ---------------------------------------------------------------------------
# File mining defaults
# ---------------------------------------------------------------------------

DEFAULT_EXTENSIONS = {
    '.py', '.md', '.yaml', '.yml', '.json', '.toml', '.cfg', '.ini',
    '.txt', '.sh', '.bash', '.zsh', '.fish', '.ps1',
}

DEFAULT_SKIP_PATTERNS = {
    '__pycache__', '.git', 'node_modules', '.venv', 'venv',
    '.mypy_cache', '.pytest_cache', '*.egg-info', '.tox',
    'dist', 'build', '.eggs',
}

DEFAULT_WING_ROOMS = {
    'config': ['config', 'setting', 'setup', 'init', 'env', 'dotenv'],
    'models': ['model', 'schema', 'entity', 'dataclass', 'orm', 'table'],
    'api': ['api', 'endpoint', 'route', 'handler', 'controller', 'view', 'middleware'],
    'services': ['service', 'logic', 'business', 'usecase', 'worker', 'job', 'task'],
    'storage': ['storage', 'database', 'repository', 'dao', 'cache', 'redis', 'postgres'],
    'auth': ['auth', 'login', 'register', 'permission', 'role', 'token', 'jwt', 'session'],
    'testing': ['test', 'spec', 'fixture', 'mock', 'stub', 'expect'],
    'frontend': ['component', 'page', 'layout', 'style', 'css', 'html', 'jsx', 'vue', 'svelte'],
    'infra': ['docker', 'deploy', 'ci', 'cd', 'terraform', 'kubernetes', 'nginx', 'aws', 'cloud'],
    'docs': ['readme', 'changelog', 'license', 'doc', 'guide', 'tutorial'],
    'general': [],  # fallback
}

# ---------------------------------------------------------------------------
# Session mining defaults
# ---------------------------------------------------------------------------

TOPIC_KEYWORDS = {
    'technical': ['code', 'python', 'function', 'bug', 'error', 'api', 'database', 'server',
                  'deploy', 'git', 'test', 'debug', 'refactor'],
    'architecture': ['architecture', 'design', 'pattern', 'structure', 'schema', 'interface',
                     'module', 'component', 'service', 'layer'],
    'planning': ['plan', 'roadmap', 'milestone', 'deadline', 'priority', 'sprint',
                 'backlog', 'scope', 'requirement', 'spec'],
    'decisions': ['decided', 'chose', 'picked', 'switched', 'migrated', 'replaced',
                  'trade-off', 'alternative', 'option', 'approach'],
    'problems': ['problem', 'issue', 'broken', 'failed', 'crash', 'stuck', 'workaround',
                 'fix', 'solved', 'resolved'],
}


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, respecting paragraph boundaries.

    Strategy:
    1. Try to split at paragraph boundaries (double newline)
    2. Fallback to sentence boundaries (period + space)
    3. Hard split at chunk_size if no boundary found
    4. Overlap between chunks for context continuity
    5. Skip chunks shorter than MIN_CHUNK_SIZE
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    chunks: list[str] = []
    start = 0
    last_end = -1  # track the last end position to prevent infinite loops

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # If we're not at the end, try to find a good break point
        if end < len(text):
            # Strategy 1: paragraph boundary (double newline)
            paragraph_break = text.rfind('\n\n', start, end)
            if paragraph_break > start:
                end = paragraph_break
            else:
                # Strategy 2: sentence boundary (period + space or newline)
                sentence_break = text.rfind('. ', start, end)
                if sentence_break > start:
                    end = sentence_break + 1  # include the period
                else:
                    # Strategy 3: newline boundary
                    newline_break = text.rfind('\n', start, end)
                    if newline_break > start:
                        end = newline_break
                    # else: hard split at chunk_size (end is already set)

        chunk = text[start:end].strip()

        # Skip chunks that are too small
        # (unless it's the last bit of text AND we had previous chunks)
        is_last = end >= len(text)
        if len(chunk) < MIN_CHUNK_SIZE:
            if is_last and not chunks:
                # Entire text is too small — skip it
                break
            if not is_last:
                # Middle chunk too small — skip
                if is_last:
                    break
                # Move past this small chunk
                last_end = end
                start = end
                continue
            # Last chunk but we already have chunks — include it for completeness
            if chunk:
                chunks.append(chunk)
        else:
            if chunk:  # don't add empty chunks
                chunks.append(chunk)

        # If we consumed all remaining text, we're done
        if end >= len(text):
            break

        # Move start forward, with overlap
        new_start = end - overlap

        # Prevent infinite loop: ensure we always advance
        if new_start <= last_end:
            new_start = end  # skip overlap, just move to end
        if new_start >= len(text):
            break

        last_end = end
        start = new_start

    return chunks


# ---------------------------------------------------------------------------
# Room detection (file mining)
# ---------------------------------------------------------------------------

def detect_room(filepath: Path, content: str = '',
                wing_rooms: dict = None) -> str:
    """Three-level priority room detection.

    1. Folder path matching — path_parts[:-1] vs room names/keywords
    2. Filename matching — filepath.stem vs room names
    3. Content keyword scoring — content[:2000].lower() vs room keywords
    4. Fallback: 'general'
    """
    if wing_rooms is None:
        wing_rooms = DEFAULT_WING_ROOMS

    path_parts = filepath.parts
    stem = filepath.stem.lower()

    # Level 1: Folder path matching
    for part in path_parts[:-1]:
        part_lower = part.lower()
        for room_name, keywords in wing_rooms.items():
            if room_name == 'general':
                continue
            # Check if the directory name matches the room name directly
            if part_lower == room_name:
                return room_name
            # Check if the directory name contains any keyword
            for kw in keywords:
                if kw in part_lower:
                    return room_name

    # Level 2: Filename matching
    for room_name, keywords in wing_rooms.items():
        if room_name == 'general':
            continue
        if stem == room_name:
            return room_name
        for kw in keywords:
            if kw in stem:
                return room_name

    # Level 3: Content keyword scoring
    if content:
        content_lower = content[:2000].lower()
        best_room = 'general'
        best_score = 0
        for room_name, keywords in wing_rooms.items():
            if room_name == 'general':
                continue
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > best_score:
                best_score = score
                best_room = room_name
        if best_score > 0:
            return best_room

    return 'general'


# ---------------------------------------------------------------------------
# File mining
# ---------------------------------------------------------------------------

def _should_skip_dir(dir_name: str, skip_patterns: set) -> bool:
    """Check if a directory should be skipped."""
    for pattern in skip_patterns:
        if fnmatch.fnmatch(dir_name, pattern):
            return True
    return False


def mine_project(store: PalaceStore, project_path: str,
                 wing: str = 'project', extensions: set = None,
                 skip_patterns: set = None) -> dict:
    """Scan a project directory, chunk files, store in Palace.

    Process:
    1. Walk project_path recursively
    2. Skip directories matching skip_patterns
    3. Skip files not in extensions set
    4. For each file: check file_already_mined (by source_key = file_path)
       - If already mined, compare mtime — re-mine if changed
    5. Read file content
    6. chunk_text()
    7. detect_room()
    8. For each chunk: store.add_drawer(...)
    9. mark_mined(source_key=file_path, mtime=os.path.getmtime(file_path))

    Returns: {'files_scanned': int, 'drawers_added': int, 'files_skipped': int}
    """
    if extensions is None:
        extensions = DEFAULT_EXTENSIONS
    if skip_patterns is None:
        skip_patterns = DEFAULT_SKIP_PATTERNS

    project = Path(project_path)
    if not project.is_dir():
        logger.warning("mine_project: %s is not a directory", project_path)
        return {'files_scanned': 0, 'drawers_added': 0, 'files_skipped': 0}

    files_scanned = 0
    drawers_added = 0
    files_skipped = 0

    for root, dirs, files in os.walk(project):
        # Filter out skipped directories in-place (modifies dirs for os.walk)
        dirs[:] = [d for d in dirs if not _should_skip_dir(d, skip_patterns)]

        for filename in files:
            filepath = Path(root) / filename
            ext = filepath.suffix.lower()

            # Skip files not in extensions
            if ext not in extensions:
                continue

            files_scanned += 1
            source_key = str(filepath)

            try:
                current_mtime = os.path.getmtime(filepath)
            except OSError:
                files_skipped += 1
                continue

            # Check if already mined and unchanged
            if store.file_already_mined(source_key):
                stored_mtime = store.get_mined_mtime(source_key)
                if stored_mtime > 0 and stored_mtime >= current_mtime:
                    files_skipped += 1
                    continue
                # File changed since last mine — re-mine

            # Read file content
            try:
                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
            except OSError:
                files_skipped += 1
                continue

            if not content.strip():
                files_skipped += 1
                continue

            # Chunk the text
            chunks = chunk_text(content)
            if not chunks:
                files_skipped += 1
                continue

            # Detect room
            room = detect_room(filepath, content)

            # Store each chunk as a drawer
            for idx, chunk in enumerate(chunks):
                store.add_drawer(
                    content=chunk,
                    wing=wing,
                    room=room,
                    source_file=source_key,
                    source_type='file',
                    chunk_index=idx,
                    importance=3.0,
                    memory_type='general',
                    source_mtime=current_mtime,
                )
                drawers_added += 1

            # Mark as mined
            store.mark_mined(source_key, mtime=current_mtime)
            logger.debug("Mined %d drawers from %s", len(chunks), source_key)

    return {
        'files_scanned': files_scanned,
        'drawers_added': drawers_added,
        'files_skipped': files_skipped,
    }


# ---------------------------------------------------------------------------
# Session mining helpers
# ---------------------------------------------------------------------------

def detect_convo_room(content: str) -> str:
    """Detect conversation room from content using keyword scoring.

    Score each topic's keywords against content, pick highest scoring topic.
    Fallback: 'general'.
    """
    if not content:
        return 'general'

    content_lower = content.lower()
    best_topic = 'general'
    best_score = 0

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in content_lower)
        if score > best_score:
            best_score = score
            best_topic = topic

    return best_topic


def _chunk_qa_pairs(messages: list) -> list[str]:
    """Split messages into Q&A pair strings.

    messages is a list of dicts: [{'role': 'user'|'assistant', 'content': str}, ...]

    - Pair consecutive user+assistant messages
    - Truncate assistant response to first 500 chars
    - Return list of "Q: {user}\\nA: {assistant}" strings
    """
    pairs: list[str] = []
    i = 0

    while i < len(messages) - 1:
        msg = messages[i]
        next_msg = messages[i + 1]

        if msg.get('role') == 'user' and next_msg.get('role') == 'assistant':
            user_content = (msg.get('content') or '').strip()
            assistant_content = (next_msg.get('content') or '').strip()

            # Truncate assistant response
            if len(assistant_content) > 500:
                assistant_content = assistant_content[:500] + '...'

            pair = f"Q: {user_content}\nA: {assistant_content}"
            pairs.append(pair)
            i += 2  # skip past both messages
        else:
            i += 1  # skip unpaired message

    return pairs


def _get_state_db_path() -> Path:
    """Get path to Hermes state.db."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "state.db"


# ---------------------------------------------------------------------------
# Session mining
# ---------------------------------------------------------------------------

def mine_sessions(store: PalaceStore, state_db_path: str = None,
                  wing: str = 'hermes_sessions',
                  max_sessions: int = 10) -> dict:
    """Mine recent sessions from Hermes SessionDB.

    Process:
    1. Open state.db (SessionDB)
    2. Query recent N sessions with message counts
    3. For each session: check file_already_mined(source_key=f"session:{session_id}")
    4. Fetch messages
    5. _chunk_qa_pairs(messages)
    6. detect_convo_room(pair)
    7. store.add_drawer(...)
    8. mark_mined(source_key=f"session:{session_id}")

    Returns: {'sessions_scanned': int, 'drawers_added': int}
    """
    sessions_scanned = 0
    drawers_added = 0

    if state_db_path is None:
        state_db_path = str(_get_state_db_path())

    db_file = Path(state_db_path)
    if not db_file.exists():
        return {
            'error': 'state.db not found',
            'sessions_scanned': 0,
            'drawers_added': 0,
        }

    try:
        conn = sqlite3.connect(str(db_file))
        conn.row_factory = sqlite3.Row
    except Exception as e:
        return {
            'error': f'failed to open state.db: {e}',
            'sessions_scanned': 0,
            'drawers_added': 0,
        }

    try:
        # Check that the required tables exist
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('sessions', 'messages')"
        ).fetchall()
        table_names = {row['name'] for row in tables}
        if 'sessions' not in table_names or 'messages' not in table_names:
            return {
                'error': 'state.db missing sessions or messages table',
                'sessions_scanned': 0,
                'drawers_added': 0,
            }

        # Query recent sessions with message counts
        try:
            rows = conn.execute("""
                SELECT s.id, s.title, COUNT(m.id) as msg_count
                FROM sessions s JOIN messages m ON m.session_id = s.id
                WHERE m.role IN ('user', 'assistant')
                GROUP BY s.id ORDER BY s.started_at DESC LIMIT ?
            """, (max_sessions,)).fetchall()
        except Exception:
            # Try alternate column names (created_at vs started_at)
            try:
                rows = conn.execute("""
                    SELECT s.id, s.title, COUNT(m.id) as msg_count
                    FROM sessions s JOIN messages m ON m.session_id = s.id
                    WHERE m.role IN ('user', 'assistant')
                    GROUP BY s.id ORDER BY s.created_at DESC LIMIT ?
                """, (max_sessions,)).fetchall()
            except Exception as e:
                return {
                    'error': f'failed to query sessions: {e}',
                    'sessions_scanned': 0,
                    'drawers_added': 0,
                }

        for row in rows:
            session_id = row['id']
            title = row['title'] or 'untitled'
            source_key = f"session:{session_id}"

            sessions_scanned += 1

            # Skip already-mined sessions
            if store.file_already_mined(source_key):
                continue

            # Fetch messages for this session
            try:
                msg_rows = conn.execute(
                    "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp",
                    (session_id,),
                ).fetchall()
            except Exception:
                continue

            messages = [{'role': r['role'], 'content': r['content'] or ''} for r in msg_rows]

            if not messages:
                continue

            # Chunk into Q&A pairs
            pairs = _chunk_qa_pairs(messages)
            if not pairs:
                # If no Q&A pairs, use the raw messages as a single chunk
                raw_text = '\n'.join(
                    f"{m['role']}: {m['content'][:300]}" for m in messages[:20]
                )
                if not raw_text.strip():
                    continue
                pairs = [raw_text]

            # Store each pair as a drawer
            for idx, pair in enumerate(pairs):
                room = detect_convo_room(pair)
                store.add_drawer(
                    content=pair,
                    wing=wing,
                    room=room,
                    source_file=f"session:{session_id}:{title}",
                    source_type='session',
                    chunk_index=idx,
                    importance=3.0,
                    memory_type='general',
                )
                drawers_added += 1

            # Mark session as mined
            store.mark_mined(source_key)
            logger.debug("Mined %d drawers from session %s", len(pairs), session_id)

    except Exception as e:
        logger.error("mine_sessions error: %s", e)
        return {
            'error': str(e),
            'sessions_scanned': sessions_scanned,
            'drawers_added': drawers_added,
        }
    finally:
        conn.close()

    return {
        'sessions_scanned': sessions_scanned,
        'drawers_added': drawers_added,
    }


# ---------------------------------------------------------------------------
# Mining status
# ---------------------------------------------------------------------------

def mine_status(store: PalaceStore) -> dict:
    """Return mining statistics.

    Returns:
    {
        'mined_sources_count': int,
        'drawers_by_source_type': {'file': N, 'session': N, 'manual': N, ...},
        'total_drawers': int,
    }
    """
    conn = store.conn

    # Count mined sources
    mined_count = conn.execute(
        "SELECT COUNT(*) FROM mined_sources"
    ).fetchone()[0]

    # Count drawers by source type
    rows = conn.execute(
        "SELECT source_type, COUNT(*) as cnt FROM drawers GROUP BY source_type"
    ).fetchall()

    drawers_by_type: dict[str, int] = {}
    for row in rows:
        drawers_by_type[row[0]] = row[1]

    # Total drawers
    total = conn.execute("SELECT COUNT(*) FROM drawers").fetchone()[0]

    return {
        'mined_sources_count': mined_count,
        'drawers_by_source_type': drawers_by_type,
        'total_drawers': total,
    }
