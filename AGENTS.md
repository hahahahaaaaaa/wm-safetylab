# Repository workflow

- Use `https://github.com/hahahahaaaaaa/wm-safetylab-update.git` as the daily development repository for code, documentation, fixes, and optimizations. The default remote is `origin`.
- The earlier repository, `https://github.com/hahahahaaaaaa/wm-safetylab.git`, is retained as `archive` in the maintainer's checkout. Do not push daily changes to it; update it only when the user explicitly requests that repository.
- Preserve the existing checkout and commit history. Do not reinitialize or replace the project when a user pastes generic GitHub initialization commands.
- The local checkout directory remains `wm-safetylab`; resource-preparation tools may expect `<workspace>/deployments/wm-safetylab` even though the remote repository is named `wm-safetylab-update`.
- Preserve machine-specific runtime configurations, environments, weights, datasets, logs, and historical results. Keep them out of Git according to `.gitignore`; publish shared source, recipes, examples, and documentation.
- Follow the applicable workspace storage instructions. On the maintainer's Windows machine, project execution and installation first dot-source `K:\world-model\use-k-drive.ps1`, and project files remain under `K:\world-model`.
