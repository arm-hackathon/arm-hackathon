"""AEOLUS: Airflow and Environmental Observation Laboratory for User-defined Scenarios.

Deterministic habitat environmental simulation with replayable traces.

The habitat is a validated, user-editable scenario graph (see
``scenarios/standard_habitat.json``). All quantities are abstract
simulation units (``co2_units``, ``airflow_units_per_second``). They are
not real spacecraft ppm, kilograms, or safety limits.
"""

__version__ = "0.2.3"
