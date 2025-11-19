# Claude Code Preferences for Webcam Filters Project

## ⚠️ CRITICAL: Git Workflow Rules ⚠️

**NEVER EVER run `git commit` or `git push` unless the user EXPLICITLY asks you to commit.**

### Examples of explicit requests:
- ✓ "commit this"
- ✓ "create a commit"
- ✓ "commit these changes"
- ✓ "push this commit"

### Examples that are NOT explicit requests to commit:
- ✗ "add a license"
- ✗ "create a new file"
- ✗ "fix this bug"
- ✗ "push to github" (when setting up remote)

### What TO do:
- Use `git add` to stage files (this is OK)
- Suggest commits when appropriate (after significant features)
- Ask "Would you like me to commit this?" if unclear

### What NOT to do:
- DO NOT run `git commit` just because files changed
- DO NOT run `git commit` after creating files like LICENSE, README, etc.
- DO NOT assume the user wants to commit
- DO NOT commit without being explicitly told to commit

## Documentation
- NEVER create documentation files (*.md, README, etc.) unless explicitly requested
- Exception: Source code comments and docstrings are always appropriate
- This includes:
  - No README files
  - No CHANGELOG files
  - No documentation markdown files
  - No API documentation
- But DO include:
  - Inline code comments when helpful
  - Function/class docstrings
  - Module-level docstrings

## Working Style
- Focus on code implementation over documentation
- Ask before making structural changes
- Prioritize working code over perfect documentation
