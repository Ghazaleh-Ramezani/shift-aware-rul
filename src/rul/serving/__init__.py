"""Serving: a SageMaker-compatible inference container with built-in drift detection.

``bundle``     train the source ensembles once and save them with their conformal quantiles
``drift``      population stability index (PSI) of incoming sensors against the source fleet
``predictor``  windows, normalization (none / anchored / auto), ensemble, conformal intervals
``app``        FastAPI app implementing the SageMaker contract: GET /ping, POST /invocations
"""
