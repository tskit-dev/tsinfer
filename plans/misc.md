- Suppress debug output from numba.

- Make the config paths relative to the current working directory, not the config file.

- Move TestComputeGroups to test_groups.py

- Don't make compute_groups importable from matching.py

- Keep all imports to the top of the file, unless absolutely necesary.

- Don't use the vcztools binary to run vcztools for tests, use python -m vcztools.
