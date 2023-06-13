# mypy: ignore-errors
import argparse
import subprocess
import sys


def run_test_local():
    try:
        subprocess.check_call(["coverage run -m pytest"], shell=True)
        return True
    except subprocess.CalledProcessError:
        print(
            "FATAL: `coverage run -m pytest` completed with a non-zero "
            "exit code. Did some test(s) fail?"
        )
        return False


def write_summary_file(test_passed, path):
    with open(path, "w") as f:
        if test_passed is not None:
            res = "✅ (passed)" if test_passed else "❌ (failed)"
        else:
            res = "⏭ (skipped)"
        f.write(
            f"{res} PyDESeq2 tests - python version "
            f"{sys.version_info.major}.{sys.version_info.minor} \n"
        )


def main(summary_file):
    test_passed = run_test_local()
    if summary_file:
        write_summary_file(test_passed, summary_file)

    sys.exit(0 if test_passed else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-output-file-path",
        type=str,
        default=None,
        help="Write a summary of the results to the given filename",
    )
    args = parser.parse_args()
    main(args.summary_output_file_path)
