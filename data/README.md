# Data

This directory is reserved for research datasets and derived artifacts.

See [the acquisition guide](../docs/DATA.md), [provenance manifest](MANIFEST.json),
and [verification report](../docs/DATA_VERIFICATION.txt) for the four published
dataset families. Raw files are never modified or committed.

Do **not** commit restricted, licensed, private, or personally identifiable mental-health data.

Suggested structure:
- `raw/` — source datasets or download scripts/metadata
- `processed/` — harmonized labels and derived benchmark files

For each dataset, record:
- source
- license/terms
- version/date
- original taxonomy
- harmonized mapping
- preprocessing steps
- whether labels are human/clinician/LLM generated
