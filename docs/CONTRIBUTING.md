# Contributing Guide

## Git Workflow

**Important: Always use feature branches and pull requests. Never push directly to `main`.**

### Standard Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   # or
   git checkout -b docs/documentation-update
   ```

2. **Make your changes**
   - Write code
   - Add tests
   - Update documentation

3. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: Add new feature description"
   ```

4. **Push your branch**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**
   - Go to GitHub repository
   - Click "New Pull Request"
   - Select your branch
   - Add description and reviewers
   - Submit PR

6. **Wait for review and approval**
   - Address review comments
   - Update PR as needed

7. **Merge via Pull Request**
   - Only maintainers merge PRs
   - Use "Squash and merge" or "Merge commit"
   - Delete branch after merge

### Branch Naming Conventions

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions/updates
- `chore/` - Maintenance tasks

### Commit Message Format

Use conventional commits:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `style:` - Formatting
- `refactor:` - Code refactoring
- `test:` - Tests
- `chore:` - Maintenance

Example:
```
feat: Add agentic AI framework with tool support

- Implement base agent classes
- Add RAG agent with tool use
- Create built-in tools (retrieval, ingestion, refinement)
- Add agent execution trace tracking
```

### Pull Request Guidelines

1. **Title**: Clear, descriptive title
2. **Description**: 
   - What changes were made
   - Why the changes were needed
   - How to test
   - Screenshots (if UI changes)
3. **Checklist**:
   - [ ] Code follows style guidelines
   - [ ] Tests added/updated
   - [ ] Documentation updated
   - [ ] No breaking changes (or documented)

### Code Review Process

1. At least one approval required
2. All CI checks must pass
3. Address all review comments
4. Maintainer merges PR

---

## Quick Reference

```bash
# Start new feature
git checkout main
git pull origin main
git checkout -b feature/my-feature

# Work and commit
git add .
git commit -m "feat: Description"

# Push and create PR
git push origin feature/my-feature
# Then create PR on GitHub

# After PR is merged, clean up
git checkout main
git pull origin main
git branch -d feature/my-feature
```

