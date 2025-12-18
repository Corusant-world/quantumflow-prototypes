# Zenodo DOI Setup Instructions

## Overview

Zenodo provides persistent DOIs for GitHub releases, making it easy to cite this repository in academic papers and research.

## Setup Steps

1. **Create Zenodo account**
   - Go to https://zenodo.org/
   - Sign up with GitHub (recommended for automatic integration)

2. **Connect GitHub repository**
   - Go to https://zenodo.org/account/settings/github/
   - Click "Enable" next to your repository
   - Zenodo will now automatically create a DOI for each GitHub release

3. **Create a GitHub Release**
   - Go to your repository on GitHub
   - Click "Releases" → "Create a new release"
   - Tag: `v0.1.0`
   - Title: `v0.1.0 — First Public Release`
   - Description: Copy from `docs/RELEASE_NOTES_TEMPLATE.md`
   - Click "Publish release"

4. **Wait for Zenodo processing**
   - Zenodo will process the release (usually takes 5-10 minutes)
   - You'll receive an email when the DOI is ready
   - The DOI will be in format: `10.5281/zenodo.XXXXXXX`

5. **Update README.md with DOI**
   - Replace placeholder in Citation section:
   ```bibtex
   @software{quantumflow2025,
     author = {Tech Eldorado},
     title = {QuantumFlow: GPU-Accelerated Prototypes Ecosystem},
     year = {2025},
     url = {https://github.com/<ORG>/<REPO>},
     note = {DOI: 10.5281/zenodo.XXXXXXX}
   }
   ```

## Notes

- Zenodo creates a new DOI for each release (even patch versions)
- The DOI is permanent and will always point to that specific release
- You can update the Zenodo record metadata if needed (but not the DOI itself)
- For best practices, create releases for major/minor versions, not every commit

## Troubleshooting

- **Release not appearing in Zenodo**: Check that the repository is enabled in Zenodo settings
- **DOI not generated**: Wait 10-15 minutes, then check Zenodo dashboard
- **Wrong metadata**: You can edit the Zenodo record after creation

## References

- Zenodo GitHub Integration: https://guides.github.com/activities/citable-code/
- Zenodo FAQ: https://help.zenodo.org/

