# Changelog

## Milestone 4 - Peer and Instructor Feedback Addressed

### Reproducibility Issues Fixed

**Feedback:** Docker and Conda environments failed to build due to missing packages and incorrect configuration.

**Actions Taken:**
- Fixed `python-json-logger` package location - moved from pip to conda dependencies with version 2.0.*
- Added `make` to Dockerfile to support Makefile execution in containers
- Fixed package version specifiers in environment.yml - removed all `>=` operators,
- Added `pytest=8.3.*` and `pytest-cov=6.0.*` to dependencies
- Versioned Docker image as `v4.0.0` instead of `:latest` tag
- Regenerated conda-lock files for all platforms (linux-64, osx-64, osx-arm64, win-64)
- Fixed README conda installation command from incorrect syntax to proper `conda-lock install`

### Test Organization

**Feedback:** Tests need to be in tests/ directory, not scattered in scripts/.

**Actions Taken:**
- Moved `scripts/test_download_data.py` to `tests/test_download_data.py`
- Ensured all test files follow pytest naming convention (test_*.py)
- Added pytest dependencies to environment.yml

### Community Guidelines Expanded

**Feedback:** CONTRIBUTING.md incomplete - missing "report issues" and "seek support" sections.

**Actions Taken:**
- Added "Reporting Issues or Problems" section with GitHub Issues workflow
- Added "Getting Support" section with response timeline expectations
- Clarified Code of Conduct reference and expectations
- Fixed confusing sentence about code of conduct compliance

### Code Structure Improvements

**Feedback:** Functions should be in src/ directory; unnecessary use of classes; functions doing multiple things.

**Actions Taken:**
- Created `src/` package directory with proper module structure:
  - `src/data_download.py` - data fetching functions
  - `src/data_processing.py` - preprocessing pipeline functions
  - `src/data_validation.py` - validation functions
  - `src/eda.py` - exploratory analysis functions
  - `src/model_training.py` - model training and evaluation functions

