# vLLM Manager Testing

This directory contains all tests for the vLLM Manager project.

## Test Structure

The tests are organized into the following categories:

- **Unit Tests** (`testing/unit/`): Tests for individual functions and classes in isolation.
- **Integration Tests** (`testing/integration/`): Tests for interactions between components.
- **System Tests** (`testing/system/`): Tests for the entire system, including concurrency and performance tests.
- **Frontend Tests** (`testing/frontend/`): Tests for the Flask frontend application.

## Running Tests

You can run all tests using the `test.sh` script in the project root:

```bash
./test.sh
```

For more advanced testing options, use the scripts in the `testing/scripts/` directory:

```bash
# Run tests in production environment (Debian with NVIDIA GPUs)
./testing/scripts/test_prod_env.sh

# Run enhanced tests with additional features
./testing/scripts/test_enhanced.sh [options]
```

The enhanced test script supports the following options:
- `--skip-stress`: Skip stress tests
- `--skip-benchmark`: Skip performance benchmarks
- `--skip-coverage`: Skip coverage reporting
- `--verbose`, `-v`: Enable verbose output

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

# Run NVIDIA-specific tests
python -m pytest testing/system/test_nvidia_compat.py testing/system/test_nvidia_specific.py
```

## Test Environment

The tests automatically detect the environment they're running in:
- macOS with Apple Silicon
- macOS with Intel
- Linux with NVIDIA GPUs
- Linux without NVIDIA GPUs

Some tests are skipped based on the environment. For example, NVIDIA-specific tests are skipped on non-NVIDIA environments.

## Test Requirements

The test requirements are specified in `testing/requirements-test.txt`. These include:
- pytest
- pytest-cov
- pytest-asyncio
- httpx

Additional requirements for enhanced testing:
- pytest-benchmark

## Coverage Reports

After running the tests, a coverage report is generated in the `htmlcov/` directory. You can open `htmlcov/index.html` in a browser to view the coverage report.

## Troubleshooting

If you encounter test failures, check the following:

1. **Environment-specific tests**: Some tests are designed for specific environments (e.g., NVIDIA GPUs). Make sure you're running the appropriate tests for your environment.

2. **Timing-sensitive tests**: Some tests may be sensitive to timing. If you see failures in tests that involve concurrency or performance, try running them again or adjusting the timing thresholds.

3. **Missing dependencies**: Make sure you have installed all the required dependencies, including the test requirements.

4. **Test isolation**: Some tests may interfere with each other. Try running the failing tests in isolation.