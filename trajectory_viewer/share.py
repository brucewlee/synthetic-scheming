#!/usr/bin/env python3
"""
Create a shareable package of trajectory viewer + data.

This script creates a self-contained directory with the viewer and all data
that can be shared with others or archived.
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Source data directory (relative to this script)
# Should contain stage3_trajectories_* folders and stage2_tasks/
DATA_DIR = "../synthetic_data/demo"

# Output directory name (will be created in current directory)
OUTPUT_DIR = f"share_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# ============================================================================
# MAIN
# ============================================================================

def create_share_package():
    """Create a shareable package with viewer + data."""

    script_dir = Path(__file__).parent
    output_path = script_dir / OUTPUT_DIR

    print(f"Creating shareable package at: {output_path}")

    # Create output directory
    output_path.mkdir(exist_ok=True)

    # Copy data directories first
    print("Copying data directories...")
    data_dir_source = (script_dir / DATA_DIR).resolve()
    data_dir_dest = output_path / "data"
    data_dir_dest.mkdir(exist_ok=True)

    trajectory_count = 0
    model_count = 0
    models_data = []
    trajectories_data = {}

    # Copy only essential files, keep directory structure
    essential_files = ['trajectory.jsonl', 'validation.json', 'final_component_states.json', 'component_states_history.json']

    if data_dir_source.exists():
        for item in data_dir_source.iterdir():
            if item.is_dir() and item.name.startswith('stage3_trajectories_'):
                print(f"  Copying {item.name}...")
                model_name = item.name.replace('stage3_trajectories_', '')
                models_data.append({
                    'id': model_name,
                    'name': model_name,
                    'path': item.name
                })

                model_trajectories = []
                # Find all trajectory directories and copy only essential files
                for trajectory_file in item.rglob("trajectory.jsonl"):
                    trajectory_dir = trajectory_file.parent
                    rel_path = trajectory_dir.relative_to(item)
                    trajectory_path_str = str(rel_path).replace(os.sep, '/')
                    model_trajectories.append(trajectory_path_str)
                    trajectory_count += 1

                    # Create destination directory
                    dest_dir = data_dir_dest / item.name / rel_path
                    dest_dir.mkdir(parents=True, exist_ok=True)

                    # Copy only essential files
                    for file in trajectory_dir.iterdir():
                        if file.is_file() and file.name in essential_files:
                            shutil.copy2(file, dest_dir / file.name)

                trajectories_data[model_name] = sorted(model_trajectories)
                model_count += 1
                print(f"    {len(model_trajectories)} trajectories")

        # Copy only environment.json files from stage2_tasks
        stage2_source = data_dir_source / "stage2_tasks"
        if stage2_source.exists():
            print("  Copying stage2_tasks...")
            for env_dir in stage2_source.iterdir():
                if env_dir.is_dir():
                    env_json = env_dir / "environment.json"
                    if env_json.exists():
                        dest_env_dir = data_dir_dest / "stage2_tasks" / env_dir.name
                        dest_env_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(env_json, dest_env_dir / "environment.json")

    models_data = sorted(models_data, key=lambda x: x['id'])
    print(f"\n  Total: {trajectory_count} trajectories across {model_count} models")

    # Create data.js with manifest and fetch interceptor
    print("\nCreating data.js...")
    data_js = f"""window.TRAJECTORY_MANIFEST = {json.dumps({'models': models_data, 'trajectories': trajectories_data}, indent=2)};

// Intercept fetch calls for static site
(function() {{
  const originalFetch = window.fetch;
  window.fetch = function(url, options) {{
    const urlStr = url.toString();

    // /models endpoint
    if (urlStr.includes('/models') || urlStr === 'models') {{
      return Promise.resolve({{
        ok: true,
        json: () => Promise.resolve(window.TRAJECTORY_MANIFEST.models)
      }});
    }}

    // /trajectories?model=... endpoint
    if (urlStr.includes('trajectories?model=')) {{
      const modelMatch = urlStr.match(/model=([^&]+)/);
      if (modelMatch) {{
        const modelId = decodeURIComponent(modelMatch[1]);
        return Promise.resolve({{
          ok: true,
          json: () => Promise.resolve(window.TRAJECTORY_MANIFEST.trajectories[modelId] || [])
        }});
      }}
    }}

    // /trajectory/MODEL_ID/path endpoint - load from data/ directory
    const trajectoryMatch = urlStr.match(/trajectory\\/([^/]+)\\/(.+)$/);
    if (trajectoryMatch) {{
      const modelId = decodeURIComponent(trajectoryMatch[1]);
      const trajectoryPath = decodeURIComponent(trajectoryMatch[2]);

      // Load trajectory.jsonl
      const trajectoryUrl = `data/stage3_trajectories_${{modelId}}/${{trajectoryPath}}/trajectory.jsonl`;

      return originalFetch(trajectoryUrl)
        .then(response => response.text())
        .then(text => {{
          const data = JSON.parse(text.split('\\n')[0]);

          // Load validation.json and environment.json
          const validationUrl = `data/stage3_trajectories_${{modelId}}/${{trajectoryPath}}/validation.json`;
          const environmentId = trajectoryPath.split('/')[0];
          const envUrl = `data/stage2_tasks/${{environmentId}}/environment.json`;

          return Promise.all([
            originalFetch(validationUrl).then(r => r.ok ? r.json() : null).catch(() => null),
            originalFetch(envUrl).then(r => r.ok ? r.json() : null).catch(() => null)
          ]).then(([validation, envData]) => {{
            if (validation) data.validation = validation;
            if (envData) data.component_initializations = envData.component_initializations || {{}};
            return {{ ok: true, json: () => Promise.resolve(data) }};
          }});
        }});
    }}

    // Fall back to original fetch
    return originalFetch(url, options);
  }};
}})();
"""
    with open(output_path / "data.js", 'w') as f:
        f.write(data_js)

    # Copy viewer.html and inject data.js
    print("Creating viewer.html...")
    with open(script_dir / "viewer.html", 'r') as f:
        viewer_content = f.read()

    # Inject data.js before closing head tag
    viewer_content = viewer_content.replace('</head>', '  <script src="data.js"></script>\n</head>')

    with open(output_path / "viewer.html", 'w') as f:
        f.write(viewer_content)

    # Create README
    print("\nCreating README...")
    readme_content = f"""# Trajectory Viewer Package

Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Contents

- `viewer.html` - Static trajectory viewer
- `data.js` - Trajectory manifest (models and paths)
- `data/` - Trajectory data (essential files only)
  - `stage3_trajectories_*/` - Trajectories by model
  - `stage2_tasks/` - Environment data

## Usage

**Local:** Just open `viewer.html` in your browser

**GitHub Pages:** Push this directory to a GitHub repo and enable Pages

## Removing Trajectories

Delete directories in `data/stage3_trajectories_MODEL_NAME/` then refresh browser

## Requirements

Modern web browser
"""

    with open(output_path / "README.md", 'w') as f:
        f.write(readme_content)

    # Create .gitignore to exclude from git if desired
    with open(output_path / ".gitignore", 'w') as f:
        f.write("# This is a generated share package\n")

    # Print summary
    print("\n" + "="*60)
    print("Package created successfully!")
    print("="*60)
    print(f"\nLocation: {output_path}")
    print(f"\nTo use locally:")
    print(f"  Open {output_path}/viewer.html in your browser")
    print(f"\nFor GitHub Pages:")
    print(f"  Push this folder to GitHub and enable Pages")

    # Calculate total size
    total_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
    size_mb = total_size / (1024 * 1024)
    print(f"\nTotal size: {size_mb:.1f} MB")

    # Optional: Create zip archive
    create_zip = input("\nCreate zip archive? (y/n): ").strip().lower()
    if create_zip == 'y':
        print("\nCreating zip archive...")
        zip_path = script_dir / f"{OUTPUT_DIR}.zip"
        shutil.make_archive(str(output_path), 'zip', script_dir, OUTPUT_DIR)
        zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"Zip created: {zip_path}")
        print(f"Zip size: {zip_size_mb:.1f} MB")
        print(f"\nYou can now share: {zip_path.name}")

if __name__ == "__main__":
    create_share_package()
