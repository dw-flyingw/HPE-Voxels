# HPE Voxels Documentation

This directory contains detailed documentation for the HPE Voxels project.

## Documentation Index

### Quality & Optimization

- **[Quality Guide](QUALITY_GUIDE.md)** - Comprehensive guide to understanding and optimizing conversion quality
  - Pipeline quality analysis
  - Parameter reference and recommendations
  - Use case-specific settings
  - Best practices and troubleshooting

### Component Documentation

- **[Backend Setup](../backend/README.md)** - FLUX.1-dev API server setup and configuration
- **[Docker Setup](../backend/DOCKER.md)** - Docker deployment guide
- **[Quick Start (Docker)](../backend/QUICKSTART_DOCKER.md)** - Quick Docker setup
- **[Frontend Usage](../frontend/USAGE.md)** - Web viewer usage instructions

### Project Overview

- **[Main README](../README.md)** - Project overview and quick start guide

## Quick Links

### Getting Started
1. Read the [Main README](../README.md) for project overview
2. Check the [Quality Guide](QUALITY_GUIDE.md) to understand conversion parameters
3. Review [Frontend Usage](../frontend/USAGE.md) for viewing models

### Common Tasks

**Converting NIfTI files:**
```bash
# See Quality Guide for parameter details
python nifti2obj.py -i ./input/nifti -o ./output/obj [OPTIONS]
python obj2glb.py -i ./output/obj -o ./output/glb
python glb2model.py
```

**Viewing models:**
```bash
python frontend/model_viewer.py
```

**Setting up FLUX server:**
- See [Backend README](../backend/README.md)
- See [Docker Setup](../backend/DOCKER.md) for containerized deployment

## Documentation Philosophy

Our documentation follows these principles:

1. **Practical** - Focused on real-world usage and examples
2. **Comprehensive** - Covers both basic and advanced topics
3. **Searchable** - Clear headings and table of contents
4. **Up-to-date** - Maintained alongside code changes

## Contributing to Documentation

When updating documentation:

1. Keep examples practical and tested
2. Include command-line examples where applicable
3. Update this index when adding new documentation
4. Use clear headings for searchability
5. Include troubleshooting sections for common issues

## File Organization

```
docs/
├── README.md              # This file - documentation index
└── QUALITY_GUIDE.md       # Quality optimization guide

backend/
├── README.md              # Backend/FLUX server setup
├── DOCKER.md              # Docker deployment
├── QUICKSTART_DOCKER.md   # Quick Docker start
└── DOCKER_SETUP_SUMMARY.md # Docker setup summary

frontend/
└── USAGE.md               # Web viewer usage

root/
└── README.md              # Main project README
```

## Need Help?

If you can't find what you're looking for:

1. Check the [Quality Guide](QUALITY_GUIDE.md) for conversion-related questions
2. Check the [Backend README](../backend/README.md) for server setup issues
3. Check the [Main README](../README.md) for general project information
4. Review the inline documentation in the Python scripts themselves

---

*Last updated: October 2025*

