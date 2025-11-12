# Deployment Checklist

Use this checklist when preparing to distribute `mlflow-eval-tools` to other teams.

## Pre-Build Checklist

### Code Quality
- [ ] All tests pass (`poetry run pytest`)
- [ ] No linting errors (`poetry run ruff check` or similar)
- [ ] Code is documented
- [ ] No debug code or print statements

### Version Management
- [ ] Version updated in `pyproject.toml`
- [ ] Version updated in `src/mlflow_eval_tools/__init__.py`
- [ ] Version updated in `src/mlflow_eval_tools/cli.py` (--version option)
- [ ] CHANGELOG.md updated (if exists) or created

### Documentation
- [ ] README.md is up-to-date
- [ ] PACKAGE_README.md is complete
- [ ] QUICK_START_TEAMS.md has latest instructions
- [ ] BUILD_GUIDE.md is current
- [ ] All code examples tested
- [ ] Screenshots/examples updated if UI changed

### Dependencies
- [ ] All dependencies listed in `pyproject.toml`
- [ ] Version constraints are appropriate
- [ ] No unnecessary dependencies
- [ ] Dependencies are secure and maintained

### Configuration
- [ ] `.env.example` file exists and is complete
- [ ] Environment variables documented
- [ ] Default values are sensible
- [ ] No secrets in code

## Build Checklist

### Build Process
- [ ] Clean previous builds: `rm -rf dist/`
- [ ] Install latest dependencies: `poetry install`
- [ ] Run tests: `poetry run pytest`
- [ ] Build package: `poetry build`
- [ ] Verify wheel created: Check `dist/` directory

### Build Verification
- [ ] Wheel file exists: `mlflow_eval_tools-X.X.X-py3-none-any.whl`
- [ ] Source tarball exists: `mlflow_eval_tools-X.X.X.tar.gz`
- [ ] File sizes are reasonable
- [ ] Inspect wheel contents:
  ```bash
  unzip -l dist/mlflow_eval_tools-X.X.X-py3-none-any.whl
  ```
- [ ] All necessary files included
- [ ] No sensitive files included (secrets, credentials, etc.)

### Test Installation
- [ ] Create clean virtual environment
- [ ] Install from wheel: `pip install dist/mlflow_eval_tools-X.X.X-py3-none-any.whl`
- [ ] Run installation test: `python test_installation.py`
- [ ] Test CLI: `mlflow-eval-tools --version`
- [ ] Test main commands work
- [ ] Verify imports: `python -c "import mlflow_eval_tools; print(mlflow_eval_tools.__version__)"`

## Distribution Checklist

### Prepare Distribution Package
- [ ] Create distribution folder
- [ ] Include wheel file
- [ ] Include README.md or PACKAGE_README.md
- [ ] Include LICENSE file
- [ ] Include .env.example
- [ ] Include QUICK_START_TEAMS.md
- [ ] Include test_installation.py
- [ ] Create distribution archive (zip or tar.gz)

### Documentation for Teams
- [ ] Installation instructions clear
- [ ] Requirements listed
- [ ] Quick start examples tested
- [ ] Troubleshooting section complete
- [ ] Support contact information included
- [ ] Links to issue tracker provided

### Version Control
- [ ] Changes committed to git
- [ ] Tag created: `git tag -a vX.X.X -m "Release vX.X.X"`
- [ ] Tag pushed: `git push origin --tags`
- [ ] Release notes created (GitHub releases or CHANGELOG)

## Deployment Methods

Choose your deployment method:

### Method 1: Direct Distribution
- [ ] Copy wheel to distribution location
- [ ] Copy documentation
- [ ] Send installation instructions to teams
- [ ] Provide support channel information

### Method 2: Internal PyPI
- [ ] Configure internal PyPI credentials
- [ ] Test upload to test PyPI (optional)
- [ ] Upload to internal PyPI: `poetry publish -r internal`
- [ ] Verify package is available
- [ ] Update team documentation with installation command

### Method 3: Git Repository
- [ ] Code pushed to repository
- [ ] Release created on GitHub/GitLab
- [ ] Wheel attached to release
- [ ] Installation instructions in README
- [ ] Teams can install via: `pip install git+https://github.com/...`

### Method 4: Network Share
- [ ] Wheel copied to network share
- [ ] Permissions set correctly
- [ ] Path documented for teams
- [ ] Verify access from different machines

## Post-Deployment Checklist

### Communication
- [ ] Announce to teams (email, Slack, etc.)
- [ ] Share installation instructions
- [ ] Provide quick start guide
- [ ] Announce support channels
- [ ] Schedule demo/training session (optional)

### Monitoring
- [ ] Monitor support channels for issues
- [ ] Track adoption metrics
- [ ] Collect feedback from teams
- [ ] Document common issues
- [ ] Plan for next version based on feedback

### Support
- [ ] Support channel ready (Slack, email, etc.)
- [ ] Issue tracker configured
- [ ] Response time expectations set
- [ ] Escalation path defined
- [ ] Documentation for common issues prepared

## Rollback Plan

In case of issues:

- [ ] Previous version available
- [ ] Rollback instructions documented
- [ ] Communication plan for rollback
- [ ] Root cause analysis process defined

## Security Checklist

### Code Security
- [ ] No hardcoded secrets or credentials
- [ ] API keys loaded from environment
- [ ] Input validation in place
- [ ] Dependencies checked for vulnerabilities

### Distribution Security
- [ ] Wheel integrity can be verified
- [ ] Distribution channel is secure
- [ ] Only authorized users can deploy
- [ ] Audit trail for distribution

## Compliance Checklist

### Licensing
- [ ] LICENSE file included
- [ ] License is appropriate (MIT, Apache, etc.)
- [ ] Third-party licenses documented
- [ ] Copyright notices correct

### Documentation
- [ ] All required documentation included
- [ ] No proprietary information in public docs
- [ ] Attribution for third-party code
- [ ] Terms of use clear (if applicable)

## Quality Assurance

### Testing
- [ ] Unit tests pass
- [ ] Integration tests pass (if applicable)
- [ ] Manual testing completed
- [ ] Edge cases tested
- [ ] Error handling verified

### Performance
- [ ] No obvious performance issues
- [ ] Resource usage is reasonable
- [ ] Large datasets handled gracefully
- [ ] Timeouts configured appropriately

### Usability
- [ ] CLI is intuitive
- [ ] Error messages are helpful
- [ ] Documentation is clear
- [ ] Examples are practical

## Success Criteria

Define what success looks like:

- [ ] X teams successfully installed
- [ ] X datasets created
- [ ] X evaluations run
- [ ] Positive feedback from early adopters
- [ ] No critical bugs reported
- [ ] Support load is manageable

## Timeline

Set expectations for:

- [ ] Build date: __________
- [ ] Internal testing: __________
- [ ] Pilot deployment: __________
- [ ] General availability: __________
- [ ] First review: __________

## Notes

Use this section for deployment-specific notes:

```
Date: ___________
Version: ___________
Deployed by: ___________
Deployment method: ___________
Special considerations: ___________
```

## Final Sign-Off

- [ ] Technical lead approval
- [ ] Product owner approval (if applicable)
- [ ] Security review completed
- [ ] Documentation review completed
- [ ] Ready for deployment

---

## Quick Reference Commands

```bash
# Clean and build
rm -rf dist/
poetry install
poetry run pytest
poetry build

# Test installation
python -m venv test_env
source test_env/bin/activate
pip install dist/mlflow_eval_tools-X.X.X-py3-none-any.whl
python test_installation.py
deactivate
rm -rf test_env

# Tag and release
git tag -a vX.X.X -m "Release vX.X.X"
git push origin --tags

# Publish (if using PyPI)
poetry publish -r internal
```

## Troubleshooting Deployment Issues

Common issues and solutions:

1. **Import errors after installation**
   - Check `packages` in pyproject.toml
   - Verify all modules are included

2. **CLI not found**
   - Check `[tool.poetry.scripts]` configuration
   - Verify PATH includes Python scripts directory

3. **Dependency conflicts**
   - Review version constraints
   - Test in clean environment

4. **Large package size**
   - Check for unnecessary files
   - Review MANIFEST.in exclusions
   - Verify no large data files included

---

Use this checklist for every deployment to ensure consistency and quality!
