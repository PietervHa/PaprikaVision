# Models

Put the trained pose weights here as `paprika_pose.pt`, then set
`paprika.backend: "pose"` in `config/default.yaml`.

Until then `paprika.backend: "shape"` runs without any model at all, which is
what lets the rest of the system be commissioned before the dataset is ready.

Weights are gitignored - keep the training run's `best.pt` somewhere backed up,
along with the dataset version it came from. A model whose training data you
cannot identify is a model you cannot debug later.
