# Implementation Guide

This document provides guidelines for developing the Standalone Photometry Pipeline.

## General Rules
- use poetry to maintain environment, keep pyproject.toml current, never directly edit poetry.lock
- Prefer using the following packages:
   - **numpy**, **scipy**, for general numerical and scientific 
   - **astropy** and **photutils** for astronomical operations. Specifically use advanced photutils photometry and segmentation functionality where beneficial (e.g. , units). Dont reimplement unless necessary
- Structure  project in a modular way to promote reusability and clear separation of concerns
- Organize code under the `dotfit` package with clear module boundaries.
- Use `@dataclass` for  structured data
- use object oriented design where abstraction is appropriate and makes the code easier to maintain / extend 
- Keep functions pure when reasonable and document expected input shapes.
- Write unit tests alongside new functionality using `pytest` and produce an insightful diagnostic image / figure
