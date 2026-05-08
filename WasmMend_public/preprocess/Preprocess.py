"""
    This is for preprocessing the repo to instrument the code to get more information.
"""

import os
import shutil
import json
from datetime import datetime
from analysis.AST_builder import ProjectASTBuilder

class Preprocess:
    def __init__(self, project_path, compile_command_json_path):
        # Convert to absolute paths to avoid issues with subprocess cwd
        self.project_path = os.path.abspath(project_path)
        compile_command_json_path = os.path.abspath(compile_command_json_path)
        self.metadata_path = os.path.join(self.project_path, "metadata.json")
        print(f"Metadata path: {self.metadata_path}")
        self.backup_dir = os.path.join(self.project_path, ".instrumentation_backups")
        self.backup_manifest = os.path.join(self.backup_dir, "manifest.json")

        # Validate paths
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        if not os.path.exists(compile_command_json_path):
            raise FileNotFoundError(f"Compile commands not found: {compile_command_json_path}")

        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Validate metadata structure
        test_info = self.metadata.get("Test Case Failure Info", {})
        if not test_info:
            raise ValueError("Metadata missing 'Test Case Failure Info'")

        self.root_file = test_info.get("file_path", "")
        self.root_line = test_info.get("failed line", 1)

        if not self.root_file:
            raise ValueError("Metadata missing file_path in Test Case Failure Info")
        if not os.path.exists(self.root_file):
            raise FileNotFoundError(f"Root file does not exist: {self.root_file}")
        if int(self.root_line) < 1:
            raise ValueError(f"Invalid line number: {self.root_line}")

        self.compile_command_json_path = compile_command_json_path
        self.modified_files = []  # Track what we've modified for rollback

    def create_backup(self, file_path):
        """Create a backup of a file before modifying it."""
        os.makedirs(self.backup_dir, exist_ok=True)

        rel_path = os.path.relpath(file_path, self.project_path)
        backup_path = os.path.join(self.backup_dir, rel_path)
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)

        shutil.copy2(file_path, backup_path)
        print(f"  Created backup: {backup_path}")
        return backup_path

    def save_backup_manifest(self):
        """Save manifest of all backed up files"""
        manifest = {
            "timestamp": datetime.now().isoformat(),
            "root_file": self.root_file,
            "root_line": self.root_line,
            "modified_files": self.modified_files
        }
        with open(self.backup_manifest, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Backup manifest saved: {self.backup_manifest}")

    def restore_backups(self):
        """Restore all files from backup"""
        if not os.path.exists(self.backup_manifest):
            print("No backup manifest found - nothing to restore")
            return

        with open(self.backup_manifest, "r") as f:
            manifest = json.load(f)

        print(f"\nRestoring files from backup (created: {manifest['timestamp']})...")
        for file_path in manifest["modified_files"]:
            rel_path = os.path.relpath(file_path, self.project_path)
            backup_path = os.path.join(self.backup_dir, rel_path)

            if os.path.exists(backup_path):
                shutil.copy2(backup_path, file_path)
                print(f"  Restored: {file_path}")
            else:
                print(f"  Warning: Backup not found for {file_path}")

    def write_back_entire_file(self, file_path, new_content):
        """Write instrumented content to file with validation"""
        if file_path is None or new_content is None:
            return

        # Create backup before modifying
        self.create_backup(file_path)

        # Write the new content
        try:
            with open(file_path, "w") as f:
                f.write(new_content)
            self.modified_files.append(file_path)
            print(f"  Wrote instrumented code to: {file_path}")
        except Exception as e:
            print(f"  ERROR writing to {file_path}: {e}")
            raise

    def pre_analysze(self):
        """Main preprocessing function with error handling and rollback"""
        try:
            print("="*60)
            print("Starting instrumentation preprocessing...")
            print(f"  Root file: {self.root_file}")
            print(f"  Root line: {self.root_line}")
            print("="*60)

            project_builder = ProjectASTBuilder(
                compile_commands_path=self.compile_command_json_path,
                root_file=self.root_file,
                root_line=self.root_line,
                root_func_name=""
            )

            modified_files = project_builder.process_project()

            if not modified_files:
                print("\nNo files need instrumentation")
                return

            print(f"\n{len(modified_files)} file(s) to be instrumented:")
            for file_path, _ in modified_files:
                print(f"  - {file_path}")

            # Write all modified files
            for file_path, instrumented_code in modified_files:
                if instrumented_code:
                    self.write_back_entire_file(file_path, instrumented_code)

            # Save manifest of changes
            self.save_backup_manifest()

            print("\n" + "="*60)
            print("Instrumentation completed successfully!")
            print(f"Backups saved to: {self.backup_dir}")
            print("="*60)

        except Exception as e:
            print(f"\n{'='*60}")
            print(f"ERROR during preprocessing: {e}")
            print("Attempting to restore from backups...")
            print("="*60)

            try:
                self.restore_backups()
                print("Rollback completed successfully")
            except Exception as restore_error:
                print(f"ERROR during rollback: {restore_error}")
                print("Manual restoration may be required!")

            raise  # Re-raise the original exception
        return modified_files

    def merge_both_program_states(self):
        """
        Execute both native and wasm builds, collect their runtime states,
        and merge them side-by-side into source files for comparison.

        Returns:
            tuple: (program_states_native, program_states_wasm)
        """
        print("\n" + "="*60)
        print("Collecting runtime states from both Native and Wasm")
        print("="*60)

        # Collect states from both builds (without writing to files yet)
        print("\n--- Collecting Native states ---")
        program_states_native = self._collect_program_state_no_write(build_type="native")

        print("\n--- Collecting Wasm states ---")
        program_states_wasm = self._collect_program_state_no_write(build_type="wasm")

        # Now merge both states together into source files
        self._merge_comparative_states(program_states_native, program_states_wasm)

        # Save combined comparison to JSON file
        comparison_file = os.path.join(self.project_path, "program_states_comparison.json")
        try:
            comparison_data = {
                "native": program_states_native,
                "wasm": program_states_wasm
            }
            with open(comparison_file, 'w') as f:
                json.dump(comparison_data, f, indent=2)
            print(f"\nComparative program states saved to: {comparison_file}")
        except Exception as e:
            print(f"Warning: Failed to save comparison states: {e}")

        return program_states_native, program_states_wasm

    def _collect_program_state_no_write(self, build_type="native"):
        """
        Helper method: collects program states but doesn't write to files.
        Returns the program_states dict.
        """
        import subprocess
        import re
        import hashlib

        # Step 1: Load the instrumentation report
        report_path = os.path.join(os.path.dirname(self.compile_command_json_path), "instrumentation_report.json")
        if not os.path.exists(report_path):
            print(f"Error: Instrumentation report not found at {report_path}")
            return {}

        with open(report_path, "r") as f:
            report = json.load(f)

        # Build marker map
        marker_map = {}
        for point in report.get("instrumentation_points", []):
            file_path = point["file"]
            line = point["line"]
            marker = hashlib.md5(f"{file_path}:{line}".encode()).hexdigest()[:8]

            instr_text = point.get("instrumentation", "")
            vars_match = re.search(r"Variables to print:\s*(\{[^}]*\})", instr_text)
            variables = vars_match.group(1) if vars_match else "{}"

            marker_map[marker] = {
                "file": file_path,
                "line": line,
                "variables": variables
            }

        # Step 2: Compile
        compile_script = os.path.join(self.project_path, "compile.sh")
        if not os.path.exists(compile_script):
            print(f"Error: compile.sh not found")
            return {}

        try:
            print(f"Compiling {build_type} build...")
            if getattr(self, 'fixed_time', False):
                env_script = os.path.join(
                    os.environ.get('CONDA_PREFIX', ''),
                    'etc', 'conda', 'activate.d', 'env_vars.sh')
                cmd = ["bash", "-c",
                       f"source {env_script} 2>/dev/null; {compile_script} {build_type}"]
            else:
                cmd = [compile_script, build_type]
            result = subprocess.run(
                cmd,
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                print(f"Compilation failed with return code {result.returncode}")
                print(f"stderr: {result.stderr[-500:]}")
                return {}
            print("Compilation successful!")
        except Exception as e:
            print(f"Error during compilation: {e}")
            return {}

        # Step 3: Run and collect output
        run_script = os.path.join(self.project_path, "run.sh")
        if not os.path.exists(run_script):
            print(f"Error: run.sh not found")
            return {}

        try:
            print(f"Running {build_type} tests...")
            if getattr(self, 'fixed_time', False):
                env_script = os.path.join(
                    os.environ.get('CONDA_PREFIX', ''),
                    'etc', 'conda', 'activate.d', 'env_vars.sh')
                cmd = ["bash", "-c",
                       f"source {env_script} 2>/dev/null; with_faketime {run_script} {build_type}"]
            else:
                cmd = [run_script, build_type]
            # Merge stderr into stdout so instrumentation markers
            # (which may be written to either stream) appear in
            # chronological order in the execution log.
            # Use surrogateescape because uint8_t values printed via
            # operator<< produce raw bytes that may not be valid UTF-8.
            result = subprocess.run(
                cmd,
                cwd=self.project_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors='surrogateescape',
                timeout=300
            )
            execution_output = result.stdout
            print(f"Execution completed with return code {result.returncode}")

            # Stash the returncode on self so callers (e.g. phase2_execution)
            # can read it without re-running run.sh.
            setattr(self, f'_exec_returncode_{build_type}',
                    result.returncode)

            output_file = os.path.join(self.project_path, f"execution_output_{build_type}.log")
            with open(output_file, "w", errors='surrogateescape') as f:
                f.write(execution_output)
            print(f"Execution output saved to: {output_file}")

        except Exception as e:
            print(f"Error during execution: {e}")
            return {}

        # Step 4: Parse output
        program_states = {}
        marker_pattern = r'@@INST_START_([a-f0-9]{8})@@(.*?)@@INST_END_\1@@'
        matches = re.findall(marker_pattern, execution_output, re.DOTALL)

        print(f"Found {len(matches)} instrumentation output blocks")

        for marker, content in matches:
            if marker not in marker_map:
                continue

            info = marker_map[marker]
            file_path = info["file"]
            line = info["line"]
            variables = info["variables"]

            value_lines = [line.strip() for line in content.strip().split('\n') if line.strip()]

            # Store the runtime values - collect ALL executions
            if file_path not in program_states:
                program_states[file_path] = {}

            if line not in program_states[file_path]:
                program_states[file_path][line] = {
                    "variables": variables,
                    "marker": marker,
                    "executions": []
                }

            # Append this execution to the list
            program_states[file_path][line]["executions"].append({
                "values": value_lines
            })

        # Print summary
        for file_path, line_states in program_states.items():
            for line, state_info in line_states.items():
                exec_count = len(state_info["executions"])
                print(f"  Captured state at {os.path.basename(file_path)}:{line} ({exec_count} executions)")

        # Save program states to JSON file
        states_file = os.path.join(self.project_path, f"program_states_{build_type}.json")
        try:
            with open(states_file, 'w') as f:
                json.dump(program_states, f, indent=2)
            print(f"Program states saved to: {states_file}")
        except Exception as e:
            print(f"Warning: Failed to save program states: {e}")

        return program_states

    def _merge_comparative_states(self, states_native, states_wasm):
        """
        Merge both native and wasm states into ORIGINAL source files side-by-side for comparison.
        """
        print("\n" + "="*60)
        print("Merging comparative runtime states into ORIGINAL source files...")
        print("="*60)

        # Get all files that have states in either native or wasm
        all_files = set(states_native.keys()) | set(states_wasm.keys())

        for file_path in all_files:
            # Get the backup (original uninstrumented) file path
            rel_path = os.path.relpath(file_path, self.project_path)
            backup_path = os.path.join(self.backup_dir, rel_path)

            if not os.path.exists(backup_path):
                print(f"Warning: Backup not found for {file_path}, skipping")
                continue

            # Read the ORIGINAL file (from backup)
            try:
                with open(backup_path, 'r') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"Error reading {backup_path}: {e}")
                continue

            # Get all line numbers from both builds
            native_lines = states_native.get(file_path, {})
            wasm_lines = states_wasm.get(file_path, {})
            all_line_nums = set(native_lines.keys()) | set(wasm_lines.keys())

            # Add comparative inline comments at the end of lines
            for line_num in sorted(all_line_nums):
                native_state = native_lines.get(line_num, None)
                wasm_state = wasm_lines.get(line_num, None)

                # Get variable info
                variables = ""
                if native_state:
                    variables = native_state['variables']
                elif wasm_state:
                    variables = wasm_state['variables']

                # Get execution counts
                native_exec_count = len(native_state['executions']) if native_state else 0
                wasm_exec_count = len(wasm_state['executions']) if wasm_state else 0

                # Build comparison for all executions
                if native_exec_count == 1 and wasm_exec_count == 1:
                    # Single execution on both sides - simple comparison
                    native_vals = ", ".join(native_state['executions'][0]['values']) if native_state and native_state['executions'][0]['values'] else "N/A"
                    wasm_vals = ", ".join(wasm_state['executions'][0]['values']) if wasm_state and wasm_state['executions'][0]['values'] else "N/A"
                    inline_comment = f"  // [Line {line_num}][COMPARE] {variables} | Native: {native_vals} | Wasm: {wasm_vals}"
                else:
                    # Multiple executions - show all
                    native_execs = []
                    if native_state:
                        for i, execution in enumerate(native_state['executions']):
                            exec_vals = ", ".join(execution['values']) if execution['values'] else "(no values)"
                            native_execs.append(f"#{i+1}: {exec_vals}")

                    wasm_execs = []
                    if wasm_state:
                        for i, execution in enumerate(wasm_state['executions']):
                            exec_vals = ", ".join(execution['values']) if execution['values'] else "(no values)"
                            wasm_execs.append(f"#{i+1}: {exec_vals}")

                    native_str = " | ".join(native_execs) if native_execs else "N/A"
                    wasm_str = " | ".join(wasm_execs) if wasm_execs else "N/A"

                    inline_comment = f"  // [Line {line_num}][COMPARE] {variables} (N:{native_exec_count} W:{wasm_exec_count} execs) | Native: {native_str} | Wasm: {wasm_str}"

                # Append comment to the end of the target line
                if 0 < line_num <= len(lines):
                    lines[line_num - 1] = lines[line_num - 1].rstrip('\n') + inline_comment + '\n'

            # Write back to the actual source file (replacing instrumented version with original + comments)
            try:
                with open(file_path, 'w') as f:
                    f.writelines(lines)
                print(f"  ✓ Restored original + merged comparative states into: {file_path}")
                if file_path not in self.modified_files:
                    self.modified_files.append(file_path)
            except Exception as e:
                print(f"Error writing {file_path}: {e}")
                continue

        # Save manifest
        self.save_backup_manifest()

        print("\n" + "="*60)
        print("Comparative state merge completed!")
        print(f"Files modified: {len(all_files)}")
        print("="*60)

if __name__ == "__main__":
    # Preprocess.py is a library consumed by diff_trace_analysis.py --
    # it has no standalone CLI. Running it directly used to go through a
    # legacy demo block with hard-coded developer paths that would fail
    # on any other machine and invoke a since-broken repair API. Route
    # users to the actual entry point instead of doing partial work.
    import sys
    sys.stderr.write(
        "Preprocess.py is a library module, not a standalone script.\n"
        "Use diff_trace_analysis.py as the entry point:\n"
        "  python diff_trace_analysis.py <project_path> "
        "<compile_commands.json>\n"
    )
    sys.exit(2)
