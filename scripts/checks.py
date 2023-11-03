"""Run all checks locally."""
import subprocess


def run_checks():
    """Run all checks locally."""
    checks = [
        "pytest",
        "mypy sparse_autoencoder",
        "pylint sparse_autoencoder",
        "black --check sparse_autoencoder",
        "build",
    ]

    for check in checks:
        # Split the command to pass it properly into subprocess
        command = check.split()
        if command[0] == "build":  # Replace 'build' with the actual poetry command
            command = ["poetry", "build"]

        else:
            # Prefix non-build commands with 'poetry' and 'run'
            command.insert(0, "poetry")
            command.insert(1, "run")

        print(f"Running {' '.join(command)}...")
        result = subprocess.run(command, check=True)

        # Check if the command was successful
        if result.returncode != 0:
            print(f"Check failed: {' '.join(command)}")
            break


if __name__ == "__main__":
    run_checks()
