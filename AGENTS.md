# AGENTS.md

## Platform and build

Production targets Linux x86-64 AMD Strix Halo (`gfx1151`) only. CMake owns
compiler flags, dependencies and installation; Nix supplies the pinned toolchain.
Linux source-build prerequisites are in [README.md](README.md#build-from-source).

```sh
nix build                              # production package, no tests/tools
./result/bin/gufo diagnose
nix build .#checks.x86_64-linux.pr      # bounded hosted CPU/repository checks
nix develop                            # GPU development and reference tools

# Same production build without Nix, with the documented dependencies installed
cmake --preset release
cmake --build --preset release --parallel 4
```

Stage only task-owned paths before Nix builds; flakes include tracked files.
Measure performance with `result/bin/gufo` or `build/release/gufo`. Preserve
compiler/dependency versions when comparing results.

## Focused tests

Formatting and Python repository checks may run on the editing host; they do
not need the remote GPU. Before committing C++ changes, run the shared CI check:

```sh
nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py
# Add --fix to apply formatting, then rerun the check.
```

Run the smallest check covering the change. `gpu-test` is RelWithDebInfo with
assertions enabled. Build only the affected target during iteration:

```sh
nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test --target <test-target>
nix develop -c ctest --preset gpu-full -R '^<test-name>$' --output-on-failure
```

The same CMake/CTest commands work outside Nix. `cmake --build --preset pr`
runs the hosted contract suite after configuring `cpu-test`. Full CPU checks,
sanitisers, GPU/operator/model checks and H3 quality tools remain available
locally; see [docs/TESTING.md](docs/TESTING.md). Do not run full model sweeps,
video generation or duplicate suites for routine edits. A missing-model skip
is not a quality pass. Broaden checks when shared behavior changes or failures
expose risk.

## Profiling and kernels

Apply [.agents/skills/optimize-kernel/SKILL.md](.agents/skills/optimize-kernel/SKILL.md).
Use `tools/bench/build.sh` for standalone HIP experiments,
`tools/bench/gfx1151_peak.hip` for measured hardware ceilings,
`tools/prof/prof.py` for pipeline/wall-time profiles, and
`tools/prof/isa_mix.py` for instruction analysis. Production paths must retain
quality; successful optimizations become the default, without extra switches.

## Development

- Keep model code, tests, tools and numerical contracts with their model.
- Use one canonical long option and backend name per behavior; avoid aliases.
- Use `gh` for GitHub operations after checking `gh auth status`.
- Push as `jaacMusic <juan.avila97@gmail.com>`: the repo-local `user.name` and
  `user.email` are already set to it, and `git log -1 --format="%an <%ae>"` shows
  whether the newest commit still matches before pushing.
- Push to `origin` (`https://github.com/jaacMusic/gufo-win.git`). After rewriting
  history use `git push --force-with-lease origin main` and confirm the remote tip
  with `gh api repos/jaacMusic/gufo-win/commits/main --template "{{.commit.author.name}} {{.commit.author.email}}"`.
- Never publish a personal machine path or a build log: notes and scripts name
  directories with placeholders such as `%TEMP%` and `C:\dev`. `git config
  core.hooksPath .githooks` installs the hooks that run
  `tools/ci/check-local-paths.py`: `--staged` before a commit and `--pushed`
  before a push, which also covers `--no-verify`, rewritten history and commits
  other tools make.
- Follow Conventional Commits with a single-line message.
- Prefer `jj` when available (`jj version`); otherwise use Git.
- Follow the user's remote workflow and preserve unrelated work.
- Windows performance-port plan and status: [docs/LINUX-PERF-PORT.md](docs/LINUX-PERF-PORT.md).
  A port with a small measured gain is kept and recorded with its numbers, never
  reverted unilaterally; report the delta and let the user decide.
