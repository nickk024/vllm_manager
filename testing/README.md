# vLLM Manager Testing

This directory contains all tests for the vLLM Manager project.

## Test Structure

The tests are organized into the following categories:

- **Unit Tests** (`testing/unit/`): Tests for individual functions and classes in isolation.
- **Integration Tests** (`testing/integration/`): Tests for interactions between components.
- **System Tests** (`testing/system/`): Tests for the entire system, including concurrency and performance tests.
- **Frontend Tests** (`testing/frontend/`): Tests for the Flask frontend application.

## Running Tests

You can run all tests using the `run_all_tests.sh` script in the project root:

```bash
./run_all_tests.sh
```

Or run specific test categories:

```bash
# Run only unit tests
python -m pytest testing/unit/

# Run only integration tests
python -m pytest testing/integration/

# Run only system tests
python -m pytest testing/system/

# Run only frontend tests
python -m pytest testing/frontend/
```

## Test Environment

The tests automatically detect the environment they're running in:
- macOS with Apple Silicon
- macOS with Intel
- Linux with NVIDIA GPUs
- Linux without NVIDIA GPUs

Some tests are skipped based on the environment. For example, NVIDIA-specific tests are skipped on non-NVIDIA environments.

## Coverage Reports

After running the tests, a coverage report is generated in the `htmlcov/` directory. You can open `htmlcov/index.html` in a browser to view the coverage report.