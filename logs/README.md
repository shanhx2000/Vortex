# Run records

One directory per invocation of `run_simulation.sh`. The point is
provenance: given a figure, these files say which run produced its input data,
when, from which commit, and what that data hashed to.

```
logs/
  index.md                    one row per run, newest last  -- start here
  runs/<run id>/
    manifest.json             command, timing, git state, output checksums
    run.log                   driver output (the [ok] / [FAIL] lines)
    jobs/<group>/<tag>.log    per-job stdout          (git-ignored, ~5 MB/run)
```

Run ids are `<YYYYMMDD-HHMMSS>_<groups>`, so they sort chronologically.

## What the manifest is for

`outputs[]` records the size, row count, mtime and **md5** of every canonical
file the run was responsible for. That is what makes the record useful later:

- **Which run produced this CSV?** md5 the file, grep the manifests. Five of the
  six files in `stats/ref/simulation/` resolve this way; the sixth,
  `vortex_forceflow_evaluation_results.csv`, was assembled from more than one
  run and matches none of them exactly.
- **Is a figure built on current data?** Compare the CSV's md5 against the
  newest manifest that lists it.
- **Which machine ran it?** `environment.host` is an 8-hex blake2s digest of
  the hostname, not the hostname itself: runs made on one machine share a value
  without the record naming anybody's infrastructure. Every run here reads
  `497c2e49` — one 48-core machine produced all of them.

**`environment.git_commit` does not resolve in this repository.** These runs
were made in the private development tree this artifact was split out of, so
every hash in them refers to a history that is not published here, and every run
is recorded `git_dirty: true` — made from a working tree that never existed as a
commit even there. Treat the commit field as a note-to-self, not as something
you can check out; the md5s are the part that still works.

## Committed vs. ignored

`index.md` and `manifest.json` are committed. `run.log` is committed (it is
small). `jobs/` is **git-ignored** — tqdm progress bars make it megabytes per
run, and it is only useful for debugging a failure while it is still fresh.

## Reading a failure

`run.log` names the failing job and its log path. Groups are independent, so a
failure in one does not abort the others; the manifest lists them under
`failed_groups` and the index row is marked `failed`.

A run that is *killed* rather than failed — the OS reaping it, or the terminal
going away — never reaches the point where the manifest and the index row are
written, so it leaves a directory holding only `run.log`. There is one such
record here, `runs/20260808-172111_batch_size_sweep+rebuttal/`: an attempted
re-run of `batch_size_sweep`, OOM-killed part-way (`terminated by signal 9` at
the end of its log). It produced nothing that any figure uses — BF's data comes
from the `20260806-204612_...` run, which the index does list — and it is kept
only so the gap in the id sequence has an explanation.
