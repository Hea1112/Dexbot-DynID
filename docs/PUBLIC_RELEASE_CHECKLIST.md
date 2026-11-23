# Publication checklist

- [ ] Confirm author/institution permission and intended code license.
- [ ] Confirm no experimental CSV, CAD, model URDF, proprietary controller, calibration or credentials are staged.
- [ ] Run `python3 -m pytest -q` from a clean checkout.
- [ ] Check `git status --short`, `git ls-files`, and commit history for private material.
- [ ] Review `data/INPUT_FORMAT.md` and `docs/REGRESSOR_INTERFACE.md` for data/input availability.
- [ ] If original 84→59 numerical reproducibility is claimed, publish or cite the actual 60-column regressor and required raw/processed recordings with authorization.
- [ ] Independent validation should use a genuinely different experiment and matching model-computed matrix.
- [ ] Insert a **real, verified** public repository URL and release commit into the reviewer response.
- [ ] Make manuscript Code/Data Availability statements match what is actually public.
