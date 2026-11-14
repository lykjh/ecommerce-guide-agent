"""Immutable session ownership, separate from checkpoint transactions."""
import asyncio
from contextlib import suppress
from time import time

import aiosqlite

from agent.identity import Identity


class SessionStoreError(Exception):
    pass


class SessionStore:
    def __init__(self, path):
        self.path = path
        self.connection = None
        self.lock = asyncio.Lock()

    async def open(self):
        try:
            self.connection = await aiosqlite.connect(self.path, timeout=1)
            await self.connection.execute('''CREATE TABLE IF NOT EXISTS chat_session_owners (
                session_id TEXT PRIMARY KEY NOT NULL,
                owner_kind TEXT NOT NULL CHECK(owner_kind IN ('guest','user')),
                user_id TEXT,
                login_fingerprint TEXT,
                created_at INTEGER NOT NULL,
                CHECK ((owner_kind='guest' AND user_id IS NULL AND login_fingerprint IS NULL)
                    OR (owner_kind='user' AND user_id IS NOT NULL AND user_id GLOB '[1-9]*'
                        AND user_id NOT GLOB '*[^0-9]*' AND login_fingerprint IS NOT NULL
                        AND length(login_fingerprint)=64 AND login_fingerprint NOT GLOB '*[^0-9a-f]*'))
            )''')
            await self.connection.commit()
        except aiosqlite.Error:
            await self.close()
            raise SessionStoreError() from None

    async def close(self):
        if self.connection is not None:
            await self.connection.close()
            self.connection = None

    async def _operation(self, sql, params, write):
        if self.connection is None:
            raise SessionStoreError()
        async with self.lock:
            try:
                async with self.connection.execute(sql, params) as cursor:
                    row = None if write else await cursor.fetchone()
                if write:
                    await self.connection.commit()
                return row
            finally:
                # Queue rollback behind any in-flight SQLite work before unlocking.
                if write:
                    await self.connection.rollback()

    async def _run(self, sql, params, write=False):
        task = asyncio.create_task(self._operation(sql, params, write))
        try:
            return await asyncio.wait_for(asyncio.shield(task), 1)
        except (TimeoutError, aiosqlite.Error):
            raise SessionStoreError() from None
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError, aiosqlite.Error):
                    await task

    async def create(self, session_id, owner):
        await self._run('INSERT INTO chat_session_owners VALUES (?,?,?,?,?)',
                        (session_id, owner.kind, owner.user_id, owner.login_fingerprint, int(time())), True)

    async def get(self, session_id):
        row = await self._run('SELECT owner_kind,user_id,login_fingerprint FROM chat_session_owners WHERE session_id=?', (session_id,))
        return Identity(*row) if row is not None else None
