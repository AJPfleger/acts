#!/usr/bin/env python3

from pathlib import Path
import os
import argparse
from fnmatch import fnmatch
import re
import sys

type_list = [
    "size_t",
    "ptrdiff_t",
    "nullptr_t",
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    "max_align_t",
]

# Initialize an empty dictionary to store compiled regex patterns
regex_patterns = {}

# Create regex patterns for each type in type_list
for TYPE in type_list:
    # Create the regex pattern dynamically using f-string
    pattern = rf"\b(?<!std::{TYPE}){TYPE}\b"
    # Compile the regex pattern
    regex_patterns[TYPE] = re.compile(pattern)

github = "GITHUB_ACTIONS" in os.environ


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument(
        "--fix", action="store_true", help="Attempt to fix any license issues found."
    )
    p.add_argument("--exclude", "-e", action="append", default=[])

    args = p.parse_args()

    # walk over all files
    exit_code = 0
    for root, _, files in os.walk("."):
        root = Path(root)
        for filename in files:
            # get the full path of the file
            filepath = root / filename
            if filepath.suffix not in (
                ".hpp",
                ".cpp",
                ".ipp",
                ".h",
                ".C",
                ".c",
                ".cu",
                ".cuh",
            ):
                continue

            if any([fnmatch(str(filepath), e) for e in args.exclude]):
                continue

            changed_lines = handle_file(filepath, fix=args.fix)
            if len(changed_lines) > 0:
                exit_code = 1
                print()
                print(filepath)
                for i, oline in changed_lines:
                    print(f"{i}: {oline}")

                    if github:
                        TYPE = next((t for t in type_list if t in oline), None)
                        if TYPE:
                            ex = regex_patterns[TYPE]
                            print(
                                f"::error file={filepath},line={i+1},title=Do not use C-style types::Replace {oline.strip()} with std::{TYPE}"
                            )

    return exit_code


def handle_file(file: Path, fix: bool) -> list[tuple[int, str]]:
    content = file.read_text()
    lines = content.splitlines()

    changed_lines = []

    for i, oline in enumerate(lines):
        for TYPE in type_list:
            ex = regex_patterns[TYPE]

            def repl_func(match):
                # Check if the match is already prefixed with std::
                if match.group(0).startswith(f"std::{TYPE}"):
                    return match.group(0)
                else:
                    return f"std::{TYPE}"

            # Replace matches in the line using a lambda function
            line, n_subs = ex.subn(repl_func, oline)

            if n_subs > 0:
                lines[i] = line
                changed_lines.append((i, oline))
                break  # Move to the next line after the first replacement

    if fix and len(changed_lines) > 0:
        file.write_text("\n".join(lines) + "\n")

    return changed_lines


if __name__ == "__main__":
    sys.exit(main())
