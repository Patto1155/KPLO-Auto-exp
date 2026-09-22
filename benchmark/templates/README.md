# Environment benchmark templates

These files are protected experimental protocols, not bundled game installations.
They define what a fair run means before an agent sees results.

Every environment adapter must accept a JSON request on stdin and emit one JSON object
per episode plus a final aggregate object:

```json
{"type":"episode","episode_id":"...","seed":1,"metrics":{"success":1}}
{"type":"aggregate","metrics":{"success_rate":0.5},"artifacts":["..."]}
```

The request contains the immutable template, stage, candidate checkpoint, starting
checkpoint, seeds/opponents, and budget. Adapters must reject unsupported versions.
Environment logs, replays and checksums are retained for audit. The harness will only
mark a template `runnable` after its adapter passes conformance tests.

Commands:

```bash
python research.py template list
python research.py template show minecraft_ender_dragon
python research.py template validate
```
