import json
import os
import sys

PLUGINS_DIR = os.path.join(os.path.expanduser("~"), ".claude", "plugins")

REQUIRED = [
    ("ponytail@ponytail", "ponytail", "DietrichGebert/ponytail"),
    ("agent-skills@addy-agent-skills", "addy-agent-skills", "addyosmani/agent-skills"),
    ("superpowers@claude-plugins-official", "claude-plugins-official", None),  # built-in marketplace
]


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


installed = load(os.path.join(PLUGINS_DIR, "installed_plugins.json")).get("plugins", {})
marketplaces = load(os.path.join(PLUGINS_DIR, "known_marketplaces.json"))

missing = [(plugin_id, mp_name, mp_repo) for plugin_id, mp_name, mp_repo in REQUIRED if plugin_id not in installed]

if not missing:
    sys.exit(0)

lines = ["This project expects some Claude Code plugins that aren't installed on this machine yet:", ""]
for plugin_id, mp_name, mp_repo in missing:
    if mp_repo and mp_name not in marketplaces:
        lines.append(f"  /plugin marketplace add {mp_repo}")
    lines.append(f"  /plugin install {plugin_id}")
lines.append("")
lines.append("Run these once to match this project's setup (see CLAUDE.md).")

print(json.dumps({"systemMessage": "\n".join(lines)}))
