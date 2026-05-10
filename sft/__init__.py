"""Modal+Unsloth SFT pipeline for BIRD text-to-SQL.

This package is opt-in: importing `sft.train_unsloth` only requires Modal
(for `modal.App`/`modal.Image`); heavy deps (`unsloth`, `torch`, `trl`,
`transformers`, `datasets`, `peft`) are lazy-imported *inside* the Modal
function so they're only resolved on the remote GPU container.
"""
