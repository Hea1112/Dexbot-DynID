# GitHub publication

After approval to release the source and license, run from the project root:

```bash
bash scripts/publish_github.sh
```

The script checks tests, lists the staged files, requests confirmation and creates the public `Dexbot-DynID` repository with GitHub CLI (`gh`). Authenticate with `gh auth login` beforehand. It does not upload trajectory or model files.

Alternatively, create an empty repository named `Dexbot-DynID` on GitHub, then initialize and push your source tree using Git. Review the staged files before publishing. Use the verified public URL in the paper's Code Availability statement.
