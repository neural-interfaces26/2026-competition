# Maintainer tools

These scripts build and reproduce the competition infrastructure. Participants
normally need only the workflows documented under [`tracks/`](../tracks/).

| Tool | Purpose |
|---|---|
| `make_starting_kit.py` | build one participant kit with its benchmark and derived example ZIPs |
| `create_bundle.py` | build one complete Codabench competition bundle per track |
| `run_docker.py` | replay Codabench ingestion and scoring locally on a submission |
| `Dockerfile`, `build_image.sh` | define and publish the shared worker environment |
| `setup_data.py` | stage public evaluation data for workers |
| `debug_submission.py` | train, reload-check, and ZIP a tiny synthetic model with the real Codabench tensor contract |
| `neuralbench_config.json` | pin the NeuralBench task configuration used by workers |
| `staged_hf_models.txt` | list Hugging Face assets cached in the worker image |

Common commands from the repository root:

```bash
python tools/make_starting_kit.py --track sleep_onset
python tools/create_bundle.py --all
python tools/run_docker.py --track sleep_onset
python tools/debug_submission.py --track sleep_onset
tools/build_image.sh --push
```

The debug command downloads no dataset. Its synthetic score is meaningless,
but the exported model has the current warm-up dimensions and its ZIP can be
uploaded to verify the complete submission path. Use `--all` for four tracks
or `--skip-install` after the first dependency installation.

The first two commands produce ZIP files at the repository root. Generated
archives, prepared data, caches, and benchmark outputs are not source files and
must not be committed.

## Worker-image replay

One Docker image supplies the environment for all four tracks and both phases.
The selected benchmark and phase config are mounted with the Codabench task,
while prepared data remain outside the image under `/app/data`.
`run_docker.py` reproduces ingestion and scoring locally:

```bash
python tools/run_docker.py --track sleep_onset \
  --submission my_submission.zip --data ~/neural-data
```

The first real-data replay prepares the cache in `--data`; later runs reuse it.
