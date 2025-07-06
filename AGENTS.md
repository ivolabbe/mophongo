## Coding Agent
- always check IMPLEMENTATION_GUIDE.md
- keep CHECKLIST.md up-to-date when implementing code


# AGENTS.md – Coding Agent Instructions 

This document provides high-level operating instructions for the Codex-based coding agent contributing to the **Mophongo** astrophysical photometry and template-fitting library.

Mophongo is a scientific Python package for PSF modeling, image registration, and template-fitting photometry.

---

## 🧱 Environment Setup

- Always use **Poetry** to manage the environment:
  - Install with `poetry install`
  - Never modify `poetry.lock` directly
  - Add dependencies only via `pyproject.toml` using `poetry add`

---

## 🔧 Implementation Workflow

1. **Read [`GUIDE.md`](./GUIDE.md)** before writing or modifying code
2. Add new code only inside the `mophongo/` package
3. Avoid project-wide refactoring unless specifically instructed
4. Ensure type hints and Google-style docstrings are applied
5. Prefer **pure functions** when possible
6. Use `logging` (never `print`) for runtime reporting

---

## ✅ Update Checklist

Update `CHECKLIST.md` if any of the following occur:

- A CHECKLIST item is completed
- A new feature is added
- Behavior or file layout changes

Use consistent format and accurate status notes.

---

## 🧪 Testing

- All added functionality must have accompanying `pytest`-based unit tests
- Test files must go under the `tests/` directory
- Use temporary file mocks for FITS I/O if needed
- Avoid network access or external resources in tests

---

## 🔁 Pull Request Generation

When preparing a pull request:

- Ensure full test suite passes: `pytest`
- Ensure CHECKLIST is updated
- Verify changes are scoped to relevant modules
- Do **not** refactor unrelated modules unless instructed
- PR title should summarize the core logic implemented
- PR body must include:
  - Summary of logic
  - Modules modified or added
  - Links to related checklist or design notes (if any)

---

## 📂 Project Layout (Internal Reference)

Get latest project structure map from https://uithub.com/ivolabbe/mophongo?accept=text%2Fplain&maxTokens=200