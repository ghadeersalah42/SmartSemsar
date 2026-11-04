"""
Prompt Hub & Versioning Engine via LangSmith
Manages prompt lifecycle, commits, SemVer tagging, and version retrieval.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from .config import PROMPTS_BACKUP_DIR
from .compat import Client


class LangSmithPromptHub:
    """Manages versioned prompts via LangSmith Prompt Hub with local fallback sync."""

    def __init__(self, client: Optional[Client] = None):
        self.client = client or Client()
        self.backup_dir = PROMPTS_BACKUP_DIR
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def push_prompt(
        self,
        agent_name: str,
        system_prompt: str,
        version: str,
        target_model: str = "openai/gpt-oss-20b",
        description: str = "",
        output_schema: Optional[str] = None
    ) -> str:
        """
        Pushes and versions a system prompt in LangSmith Prompt Hub.
        """
        repo_id = f"smart-semsar/{agent_name}"
        full_identifier = f"{repo_id}:{version}"
        
        payload = {
            "agent_name": agent_name,
            "version": version,
            "system_prompt": system_prompt,
            "target_model": target_model,
            "output_schema": output_schema,
            "description": description
        }
        
        # Save local JSON backup
        agent_dir = self.backup_dir / agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        with open(agent_dir / f"{version}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        with open(agent_dir / "latest.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        # Push to LangSmith Hub
        try:
            self.client.push_prompt(
                repo_id,
                object=payload,
                description=f"{version}: {description}"
            )
        except Exception:
            pass

        return full_identifier

    def pull_prompt(self, agent_name: str, version: str = "latest") -> Dict[str, Any]:
        """
        Retrieves a versioned prompt from LangSmith Hub or local cache.
        """
        repo_id = f"smart-semsar/{agent_name}:{version}"
        
        # Check local backup first
        agent_dir = self.backup_dir / agent_name
        file_path = agent_dir / f"{version}.json"
        if not file_path.exists() and version == "latest":
            file_path = agent_dir / "latest.json"
            
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # Fallback to pulling from LangSmith Hub
        try:
            content = self.client.pull_prompt(repo_id)
            if isinstance(content, str):
                try:
                    return json.loads(content)
                except Exception:
                    return {"system_prompt": content, "version": version}
            elif isinstance(content, dict):
                return content
        except Exception as e:
            raise FileNotFoundError(f"Prompt {repo_id} not found: {e}")

    def list_versions(self, agent_name: str) -> List[str]:
        """Lists all locally registered versions for an agent."""
        agent_dir = self.backup_dir / agent_name
        if not agent_dir.exists():
            return []
        versions = [f.stem for f in agent_dir.glob("*.json") if f.stem != "latest"]
        return sorted(versions)


_hub_instance = None

def get_prompt_hub() -> LangSmithPromptHub:
    global _hub_instance
    if _hub_instance is None:
        _hub_instance = LangSmithPromptHub()
    return _hub_instance
