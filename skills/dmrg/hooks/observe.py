"""Observe node hooks for DMRG skill — registers error detectors.

Detectors are run in priority order (first match wins).
Put more specific / more common detectors first.
"""
from dmrg.detectors.scf import ScfConvergenceDetector
from dmrg.detectors.python_error import PythonErrorDetector
from dmrg.detectors.mpi import MpiErrorDetector
from dmrg.detectors.gaussian import GaussianErrorDetector
from dmrg.detectors.orca import OrcaErrorDetector
from dmrg.detectors.vasp import VaspErrorDetector


class DmrgObserveHooks:
    """Register error detectors for quantum chemistry computations.

    Detectors are run in the order returned by get_error_detectors().
    The first detector that returns an ErrorInfo wins.
    """

    def get_error_detectors(self):
        """Return detectors in priority order.

        Priority:
        1. SCF convergence — most common QC error
        2. Python errors — runtime issues
        3. MPI errors — parallel execution (stub)
        4. Gaussian errors (stub)
        5. ORCA errors (stub)
        6. VASP errors (stub)
        """
        return [
            ScfConvergenceDetector(),
            PythonErrorDetector(),
            MpiErrorDetector(),
            GaussianErrorDetector(),
            OrcaErrorDetector(),
            VaspErrorDetector(),
        ]
