"""Ephemeral SDK mirror used to test earlier native metadata delivery.

The SDK still owns its durable local transcript. This optional mirror keeps
the SDK's entries verbatim in memory and reads them through its public APIs.
It neither exports transcripts nor attempts to restore a session from them.
"""

from collections.abc import Mapping
from typing import Any


class NativeMetadataMirror:
    def __init__(self, sdk: Any, on_append: Any) -> None:
        self.store = sdk.InMemorySessionStore()
        self.on_append = on_append
        self.appends = 0
        self.metadata_entries = 0
        self.metadata_with_model = 0
        self.callback_errors = 0
        self.metadata = {}
        self.conflicted = set()

    async def append(self, key, entries):
        await self.store.append(key, entries)
        self.appends += 1
        for entry in entries:
            if isinstance(entry, Mapping) and entry.get("type") == "agent_metadata":
                self.metadata_entries += 1
                self.metadata_with_model += int(isinstance(entry.get("model"), str))
                subpath = key.get("subpath")
                if isinstance(subpath, str) and subpath.startswith("subagents/"):
                    identity = (key["session_id"], subpath)
                    metadata = {
                        field: entry.get(field) for field in ("toolUseId", "parentAgentId", "model")
                    }
                    if identity in self.metadata and self.metadata[identity] != metadata:
                        self.conflicted.add(identity)
                    else:
                        self.metadata[identity] = metadata
        try:
            await self.on_append(key)
        except Exception:
            # Observation must not abort the SDK's successful transcript write.
            # Keep cancellation visible, and retain no exception text or entries
            # in diagnostics. The exact-identity resolver still rejects conflicts.
            self.callback_errors += 1

    def metadata_for_agent(self, session, agent):
        # This is the public get_subagent_messages_from_store subkey contract.
        candidates = [value for (owner, path), value in self.metadata.items()
                      if owner == session and path.rsplit("/", 1)[-1] == "agent-" + agent]
        return candidates[0] if len(candidates) == 1 and not self.is_conflicted(session, agent) else None

    def is_conflicted(self, session, agent):
        paths = [key for key in self.metadata
                 if key[0] == session and key[1].rsplit("/", 1)[-1] == "agent-" + agent]
        return len(paths) > 1 or any(key in self.conflicted for key in paths)

    async def load(self, key):
        return await self.store.load(key)

    async def list_subkeys(self, key):
        return await self.store.list_subkeys(key)

    async def list_sessions(self, project_key):
        return await self.store.list_sessions(project_key)
