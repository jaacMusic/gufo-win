# Release procedure

Releases are tagged `vX.Y.Z` on `main` and published from
https://github.com/jaacMusic/gufo-win/releases. The release body is the
hand-written notes file (headline, measured tables, What changed), in the
style of `release-0.4.0-notes.md` / `release-0.6.0-notes.md` at the repo root.

## Asset naming convention

Exactly one zip per release, named `gufo-windows-v<version>.zip`
(`gufo-windows-v0.5.0.zip`, `gufo-windows-v0.6.0.zip`). No date-stamped
names for tagged releases.

## Steps

1. Finish and **commit everything the release claims**: the release binary
   is built from the working tree, so uncommitted changes (code or docs)
   make the tag lie about what shipped. Verify with `git status --short`.
2. Build: `tools\windows\build.ps1` (preset release).
3. Package: `powershell -ExecutionPolicy Bypass -File tools\windows\package.ps1 -Version v0.6.0`
   -> `dist\gufo-windows-v0.6.0.zip`. The staged `README.txt` records the
   commit it was built from.
4. Push `main` (fast-forward; `--force-with-lease` only after history
   rewrites, then confirm the remote tip author is
   `jaacMusic <juan.avila97@gmail.com>`).
5. Tag and push: `git tag -a v0.6.0 -m "v0.6.0" <commit> && git push origin v0.6.0`.
   Re-pointing a published tag: delete remote and local tag, re-tag, push —
   and note GitHub turns the release into a **draft** when its tag is deleted;
   republish with `gh release edit v0.6.0 -R jaacMusic/gufo-win --draft=false`.
6. Publish with gh, targeting the fork explicitly (a bare `gh` command can
   resolve to the upstream `gufo-org/gufo` repo):
   `gh release create v0.6.0 -R jaacMusic/gufo-win --title "Gufo-win v0.6.0" --notes-file release-0.6.0-notes.md --verify-tag`
7. Attach the zip: `gh release upload v0.6.0 -R jaacMusic/gufo-win dist\gufo-windows-v0.6.0.zip --clobber`.
8. Verify asset name and release body at the release URL.

Benchmarks quoted in the notes follow [BENCHMARKS.md](BENCHMARKS.md):
one engine process at a time, same snapshot and bench command on both
sides, and the comparison table refreshed in the model's `BENCHMARKS.md`.
