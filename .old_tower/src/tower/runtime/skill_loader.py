"""Skill loader — parses skill.yaml and loads hook modules."""
import importlib
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class Skill:
    """A loaded skill pack with metadata and optional hook instances.

    Hook fields are None if the skill doesn't provide that hook.
    """
    name: str
    version: str = "0.1.0"
    description: str = ""
    system_prompt: str = ""
    rules: list[str] = field(default_factory=list)
    mcp_servers: list[dict] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)
    # Hook instances (None if not provided by skill)
    plan_hooks: Any = None
    act_hooks: Any = None
    observe_hooks: Any = None
    refine_hooks: Any = None
    respond_hooks: Any = None


class SkillLoader:
    """Load a skill pack from a directory containing skill.yaml."""

    @staticmethod
    def load(skill_dir: str) -> Skill | None:
        """Load skill from directory.

        Args:
            skill_dir: Path to skill directory (e.g. 'skills/dmrg').

        Returns:
            Skill object with hooks loaded, or None if skill.yaml not found.
        """
        skill_path = Path(skill_dir)
        yaml_path = skill_path / "skill.yaml"

        if not yaml_path.exists():
            return None

        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
        except (yaml.YAMLError, OSError):
            return None

        if not isinstance(data, dict) or "name" not in data:
            return None

        skill = Skill(
            name=data["name"],
            version=data.get("version", "0.1.0"),
            description=data.get("description", ""),
            system_prompt=data.get("system_prompt", ""),
            rules=data.get("rules", []) or [],
            mcp_servers=data.get("mcp_servers", []) or [],
            tools=data.get("tools", []) or [],
        )

        # Load hook modules if present
        hooks_dir = skill_path / "hooks"
        if hooks_dir.is_dir():
            SkillLoader._load_hooks(skill, str(hooks_dir), skill_path.name)

        return skill

    @staticmethod
    def _load_hooks(skill: Skill, hooks_dir: str, skill_pkg: str):
        """Dynamically import hook modules from skills/<name>/hooks/.

        Each hook module (plan.py, act.py, etc.) should export a single
        class that implements the corresponding hook protocol.
        The first non-imported class in the module is instantiated.
        """
        # Ensure the skill directory's parent is on sys.path for imports
        skill_parent = str(Path(hooks_dir).parent.parent)
        if skill_parent not in sys.path:
            sys.path.insert(0, skill_parent)

        hook_files = {
            "plan": "plan_hooks",
            "act": "act_hooks",
            "observe": "observe_hooks",
            "refine": "refine_hooks",
            "respond": "respond_hooks",
        }

        for module_name, attr_name in hook_files.items():
            module_path = Path(hooks_dir) / f"{module_name}.py"
            if not module_path.exists():
                continue

            try:
                mod = importlib.import_module(f"{skill_pkg}.hooks.{module_name}")
                hook_instance = SkillLoader._find_hook_instance(mod)
                if hook_instance is not None:
                    setattr(skill, attr_name, hook_instance)
            except Exception:
                # Hook loading failure should not crash the skill loader
                pass

    @staticmethod
    def _find_hook_instance(module):
        """Find the first locally-defined class in a hook module and instantiate it.

        Skips imported classes and private classes (starting with _).
        Returns the instance, or None if no suitable class found.
        """
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Only consider classes defined in this module (not imported)
            if obj.__module__ != module.__name__:
                continue
            if name.startswith("_"):
                continue
            try:
                return obj()
            except Exception:
                pass
        return None
