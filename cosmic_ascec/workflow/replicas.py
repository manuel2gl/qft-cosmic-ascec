"""Annealing replicas — independent re-runs of the same ``.asc`` input.

The COSMIC ASCEC protocol opens with a *replication* stage: the same annealing
search is run several times from independent random seeds, widening the pool of
candidate minima before clustering.

**R6b re-port.** The pre-R0 v05 carried a clean-refactor (``ReplicaSpec`` /
``create_replicas`` / ``write_launcher`` / ``merge_launchers``) whose own
docstring admitted the launcher scripts were "*simplified*". **D-039** (faithful
decomposition) rules that out: this module is now a **verbatim port** of
``ascec-v04.py``:

* ``PROTOCOL_MARKER_RE`` / ``is_protocol_marker_line`` — lines 99-117
* ``strip_protocol_from_content`` — lines 1926-1947
* ``create_launcher_script`` — lines 4795-4867
* ``merge_launcher_scripts`` — lines 4870-4979
* ``update_box_size_in_input`` — lines 6978-7012
* ``create_replicated_runs`` — lines 7014-7082

The directory layout (``annealing/<stem>_<i>/<stem>_<i>.asc``) and the launcher
script bodies are reproduced byte for byte — the ``index.html`` GUI and user
scripts depend on them.

**The one mechanical adaptation** (permitted by the gold rule — moving code
across file boundaries): v04's launcher writers read ``os.path.abspath(__file__)``
to locate the running ``ascec-v04.py`` and embed ``python <that path> ...`` in
the launcher. v05 is a package; the equivalent "main script" is the thin root
``ascec.py`` shim (assembled in R7/R8). ``_REPO_ROOT`` / ``_ASCEC_SCRIPT`` below
resolve to it, the two ``ascec_script_path`` / ``ascec_directory`` lines in
``create_launcher_script`` / ``merge_launcher_scripts`` read those instead of
``__file__``, and ``merge_launcher_scripts`` matches the v05 launcher line
(``ascec.py``) instead of v04's ``ascec-v04.py``. No other logic is changed.
"""

from __future__ import annotations

import os
import re
import sys
from typing import List, Optional

# Mechanical adaptation: v04 embeds the running ``ascec-v04.py`` path; v05
# embeds the thin root ``ascec.py`` shim.  ``replicas.py`` lives at
# ``<repo>/cosmic_ascec/workflow/replicas.py`` — the repo root is three levels up.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_ASCEC_SCRIPT = os.path.join(_REPO_ROOT, "ascec-v04.py")

# v04 line 99-102: supports the placeholder marker (``.asc,``) and explicit
# input-file markers ending with ``.asc`` (e.g. ``formic_annealing.asc,``).
PROTOCOL_MARKER_RE = re.compile(
    r'^\s*(?:\.asc|[^,\s#]+\.asc)\s*,',
    re.IGNORECASE,
)

__all__ = [
    "PROTOCOL_MARKER_RE",
    "is_protocol_marker_line",
    "strip_protocol_from_content",
    "input_box_is_cubic",
    "update_box_size_in_input",
    "create_launcher_script",
    "merge_launcher_scripts",
    "create_replicated_runs",
]


def is_protocol_marker_line(raw_line: str) -> bool:
    """Return True when a line starts an embedded protocol block."""
    if not raw_line:
        return False
    stripped = raw_line.strip().lstrip('\ufeff')
    if not stripped:
        return False
    # Ignore inline comments while preserving the leading command token.
    if '#' in stripped:
        stripped = stripped.split('#', 1)[0].strip()
    if not stripped:
        return False
    return bool(PROTOCOL_MARKER_RE.match(stripped))


def strip_protocol_from_content(content: str) -> str:
    """Remove the embedded .asc protocol section and everything after it.

    Annealing replicas only need the raw ASCEC configuration parameters
    (box size, temperature, molecules …). The protocol block and any
    embedded QM templates that follow it are not needed and must not be
    copied – otherwise each replica would re-trigger the workflow.
    """
    lines = content.splitlines(keepends=True)
    for i, line in enumerate(lines):
        # Strip inline comments to get the bare content of this line
        bare = line.strip()
        if '#' in bare:
            bare = bare.split('#', 1)[0].strip()
        # Stop as soon as we hit the embedded protocol marker
        if is_protocol_marker_line(bare):
            # Drop trailing blank lines so the file ends cleanly
            trimmed = lines[:i]
            while trimmed and not trimmed[-1].strip():
                trimmed.pop()
            return ''.join(trimmed)
    return content


def _box_values(line: str) -> List[str]:
    """The numeric fields of a Line-2 box declaration, comments stripped."""
    return line.split('#', 1)[0].split('!', 1)[0].split()


def input_box_is_cubic(input_file_path: str) -> bool:
    """Whether Line 2 of an ``.asc`` declares a cube rather than a prism.

    ``--box<percent>`` re-derives one cube edge from a packing percentage, so it
    only applies to a cubic box; this is the check every caller of it makes
    first. Text-level on purpose — it answers the question without requiring the
    rest of the file to parse.
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except OSError:
        return True

    line_count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        line_count += 1
        if line_count == 2:
            return len(_box_values(line)) < 3
    return True


def update_box_size_in_input(input_file_path: str, new_box_size: float) -> str:
    """
    Updates the box size (Line 2) in an ASCEC input file.
    
    Args:
        input_file_path (str): Path to the input file
        new_box_size (float): New box size in Angstroms
    
    Returns:
        str: Modified content with updated box size
    """
    with open(input_file_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    # Find and update Line 2 (Simulation Cube Length)
    # Skip comment lines and empty lines
    line_count = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Skip empty lines and pure comment lines
        if not stripped or stripped.startswith('#'):
            continue
        line_count += 1
        
        # Line 2 is the box size
        if line_count == 2:
            # A rectangular-prism box (``Lx Ly Lz``) is bound to the geometry
            # sitting in it — a substrate's footprint comes from the slab, so
            # there is no single edge to substitute and no way to re-place the
            # slab from here, where all we have is the text of the file.
            # Leave it exactly as it is: callers warn before getting this far
            # (see :func:`input_box_is_cubic`), and silently returning the file
            # unchanged is safer than writing one number over three.
            if len(_box_values(line)) >= 3:
                return ''.join(lines)
            # Preserve any inline comment
            if '#' in line:
                comment_part = '#' + line.split('#', 1)[1]
                lines[i] = f"{new_box_size:.1f}           {comment_part}"
            else:
                lines[i] = f"{new_box_size:.1f}           # Line 2: Simulation Cube Length (Angstroms)\n"
            break
    
    return ''.join(lines)


def create_launcher_script(replicated_files: List[str], input_dir: str, script_name: Optional[str] = None) -> str:
    """
    Creates a launcher script for sequential execution of replicated runs.
    Generates a .sh bash script on Linux/macOS and a .bat script on Windows.

    Args:
        replicated_files (List[str]): List of paths to the replicated input files
        input_dir (str): Directory where the launcher script should be created
        script_name (str): Name of the launcher script (auto-detected from OS if None)

    Returns:
        str: Path to the created launcher script
    """
    is_windows = sys.platform == 'win32'
    if script_name is None:
        script_name = "launcher_ascec.bat" if is_windows else "launcher_ascec.sh"

    launcher_path = os.path.join(input_dir, script_name)

    # Get the directory where ascec-v04.py is located
    ascec_script_path = _ASCEC_SCRIPT
    ascec_directory = _REPO_ROOT

    try:
        # newline='': write bytes exactly as given. The Windows branch below
        # emits explicit \r\n; with the default newline=None those \n would be
        # translated a second time on Windows, producing \r\r\n on every line.
        # The POSIX branch writes \n and must keep \n even when generated on
        # Windows, so that the .sh still runs under WSL or Git Bash.
        with open(launcher_path, 'w', newline='', encoding='utf-8') as f:
            if is_windows:
                f.write("@echo off\r\n\r\n")
                f.write("rem Configuration for ASCEC v04\r\n")
                f.write("rem Set ASCEC_ROOT to the directory containing ascec-v04.py\r\n")
                f.write(f'set ASCEC_ROOT={ascec_directory}\r\n\r\n')
                f.write("rem Add the ASCEC directory to the system PATH for direct execution\r\n")
                f.write('set PATH=%ASCEC_ROOT%;%PATH%\r\n\r\n')
                f.write('echo ASCEC v04 environment is now active via direct script setup.\r\n')
                f.write('echo ASCEC_ROOT set to: %ASCEC_ROOT%\r\n\r\n')
                f.write("rem Run ASCEC using the full path\r\n")
                for i, replicated_file in enumerate(replicated_files):
                    rel_path = os.path.relpath(replicated_file, input_dir)
                    output_name = os.path.splitext(rel_path)[0] + ".out"
                    if i > 0:
                        f.write('echo ==================================================================\r\n')
                    f.write(f'python "{ascec_script_path}" "{rel_path}" > "{output_name}" 2>&1\r\n')
            else:
                f.write("#!/bin/bash\n\n")
                f.write("# Configuration for ASCEC v04\n")
                f.write("# Set ASCEC_ROOT to the directory containing ascec-v04.py\n")
                f.write(f'export ASCEC_ROOT="{ascec_directory}"\n\n')
                f.write("# Save original environment paths\n")
                f.write('_SYSTEM_PATH="$PATH"\n\n')
                f.write("# Add the ASCEC directory to the system PATH for direct execution\n")
                f.write('export PATH="$ASCEC_ROOT:$_SYSTEM_PATH"\n\n')
                f.write('echo "ASCEC v04 environment is now active via direct script setup."\n')
                f.write('echo "ASCEC_ROOT set to: $ASCEC_ROOT"\n\n')
                f.write("# Run ASCEC using the full path\n")
                commands = []
                for i, replicated_file in enumerate(replicated_files):
                    rel_path = os.path.relpath(replicated_file, input_dir)
                    output_name = os.path.splitext(rel_path)[0] + ".out"
                    if i > 0:
                        commands.append('echo "=================================================================="')
                    commands.append(f"python {ascec_script_path} {rel_path} > {output_name} 2>&1")
                f.write(" ; \\\n".join(commands))
                f.write("\n")

        # Make the script executable (Unix only)
        if not is_windows:
            os.chmod(launcher_path, 0o755)

        print(f"Created launcher script: {script_name}")
        return launcher_path

    except IOError as e:
        print(f"Error creating launcher script '{launcher_path}': {e}")
        return ""


def merge_launcher_scripts(working_dir: str = ".") -> str:
    """
    Finds all launcher_ascec scripts in the working directory and subfolders,
    and merges them into a single launcher script.
    Generates a .sh bash script on Linux/macOS and a .bat script on Windows.

    Args:
        working_dir (str): Working directory to search for launcher scripts

    Returns:
        str: Path to the merged launcher script
    """
    is_windows = sys.platform == 'win32'
    launcher_filename = "launcher_ascec.bat" if is_windows else "launcher_ascec.sh"

    working_dir_full = os.path.abspath(working_dir)
    merged_launcher_path = os.path.join(working_dir_full, launcher_filename)

    # Find all launcher scripts
    launcher_scripts = []
    for root, dirs, files in os.walk(working_dir_full):
        for file in files:
            if file == launcher_filename:
                launcher_scripts.append(os.path.join(root, file))

    if not launcher_scripts:
        print(f"No {launcher_filename} scripts found in the working directory or subfolders.")
        return ""

    print(f"Found {len(launcher_scripts)} launcher scripts:")
    for script in launcher_scripts:
        rel_path = os.path.relpath(script, working_dir_full)
        print(f"  {rel_path}")

    # Merge all launcher scripts
    all_commands = []

    try:
        for script_path in launcher_scripts:
            with open(script_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()

            # Extract python commands (skip shebang, comments, echo, set/export lines)
            commands = []
            for line in lines:
                line = line.strip()
                if line and 'python' in line and 'ascec-v04.py' in line:
                    # Remove trailing " ; \\" if present (bash style)
                    line = line.rstrip(' \\;').strip()
                    commands.append(line)

            if commands:
                all_commands.extend(commands)
                if script_path != launcher_scripts[-1]:
                    all_commands.append("###")

        # Get the directory where ascec-v04.py is located
        ascec_script_path = _ASCEC_SCRIPT
        ascec_directory = _REPO_ROOT

        # Write merged launcher script
        # newline='': see create_launcher_script above.
        with open(merged_launcher_path, 'w', newline='', encoding='utf-8') as f:
            if is_windows:
                f.write("@echo off\r\n\r\n")
                f.write("rem Configuration for ASCEC v04\r\n")
                f.write("rem Set ASCEC_ROOT to the directory containing ascec-v04.py\r\n")
                f.write(f'set ASCEC_ROOT={ascec_directory}\r\n\r\n')
                f.write("rem Add the ASCEC directory to the system PATH for direct execution\r\n")
                f.write('set PATH=%ASCEC_ROOT%;%PATH%\r\n\r\n')
                f.write('echo ASCEC v04 environment is now active via direct script setup.\r\n')
                f.write('echo ASCEC_ROOT set to: %ASCEC_ROOT%\r\n\r\n')
                f.write("rem Run ASCEC using the full path\r\n")
                for cmd in all_commands:
                    if cmd == "###":
                        f.write('echo ==================================================================\r\n')
                    else:
                        f.write(cmd + "\r\n")
            else:
                f.write("#!/bin/bash\n\n")
                f.write("# Configuration for ASCEC v04\n")
                f.write("# Set ASCEC_ROOT to the directory containing ascec-v04.py\n")
                f.write(f'export ASCEC_ROOT="{ascec_directory}"\n\n')
                f.write("# Save original environment paths\n")
                f.write('_SYSTEM_PATH="$PATH"\n\n')
                f.write("# Add the ASCEC directory to the system PATH for direct execution\n")
                f.write('export PATH="$ASCEC_ROOT:$_SYSTEM_PATH"\n\n')
                f.write('echo "ASCEC v04 environment is now active via direct script setup."\n')
                f.write('echo "ASCEC_ROOT set to: $ASCEC_ROOT"\n\n')
                f.write("# Run ASCEC using the full path\n")
                for i, cmd in enumerate(all_commands):
                    if cmd == "###":
                        f.write(" ; \\\n###\n")
                    else:
                        f.write(cmd)
                        if i < len(all_commands) - 1 and all_commands[i + 1] != "###":
                            f.write(" ; \\\n")
                        elif i == len(all_commands) - 1:
                            f.write("\n")

        # Make the script executable (Unix only)
        if not is_windows:
            os.chmod(merged_launcher_path, 0o755)

        print(f"\nCreated merged launcher script: {launcher_filename}")
        print(f"Total commands: {len([cmd for cmd in all_commands if cmd != '###'])}")
        return merged_launcher_path

    except IOError as e:
        print(f"Error creating merged launcher script: {e}")
        return ""


def create_replicated_runs(input_file_path: str, num_replicas: int, create_launcher: bool = True, box_size: Optional[float] = None, verbose: bool = True) -> List[str]:
    """
    Creates replicated folders and input files for multiple annealing runs.
    
    Args:
        input_file_path (str): Path to the original input file
        num_replicas (int): Number of replicas to create
        create_launcher (bool): Whether to create launcher script (default: True)
        box_size (Optional[float]): If provided, updates the box size in replicated files
    
    Returns:
        List[str]: List of paths to the replicated input files
    """
    input_file_path_full = os.path.abspath(input_file_path)
    input_dir = os.path.dirname(input_file_path_full)
    input_basename = os.path.basename(input_file_path_full)
    input_name, input_ext = os.path.splitext(input_basename)
    
    replicated_files = []
    
    # Create parent directory: annealing
    parent_folder_name = "annealing"
    parent_folder_path = os.path.join(input_dir, parent_folder_name)
    os.makedirs(parent_folder_path, exist_ok=True)
    if verbose:
        print(f"Creating {num_replicas} replicated runs in '{parent_folder_name}/'...")
    
    for i in range(1, num_replicas + 1):
        # Create folder name: e.g., example_1, example_2, example_3
        folder_name = f"{input_name}_{i}"
        folder_path = os.path.join(parent_folder_path, folder_name)
        
        # Create the folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)
        
        # Create the replicated input file name: e.g., example_1.in, example_2.in, example_3.in
        replicated_input_name = f"{input_name}_{i}{input_ext}"
        replicated_input_path = os.path.join(folder_path, replicated_input_name)
        
        # Copy the original input file to the new location
        try:
            with open(input_file_path_full, 'r', encoding='utf-8', errors='replace') as src:
                content = src.read()
            
            # Update box size if specified
            if box_size is not None:
                content = update_box_size_in_input(input_file_path_full, box_size)

            content = strip_protocol_from_content(content)
            
            with open(replicated_input_path, 'w', encoding='utf-8') as dst:
                dst.write(content)
            
            replicated_files.append(replicated_input_path)
            if verbose:
                print(f"  Created: {folder_name}/{replicated_input_name}")
            
        except IOError as e:
            print(f"Error creating replicated file '{replicated_input_path}': {e}")
            continue
    
    # Create launcher script only if requested
    if create_launcher and replicated_files:
        create_launcher_script(replicated_files, input_dir)
        if verbose:
            print("\nTo run all simulations sequentially, use:")
            print("  ./launcher_ascec.sh")
    
    return replicated_files
