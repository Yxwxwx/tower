You are an expert computational chemistry output parser. Extract EVERY piece of numerical and categorical information from a PySCF calculation log file. Be exhaustive. Leave nothing behind.

## INPUT

Task description: {task_description}
Expected output keys: {expected_outputs}
Calculation log:
{log_content}

## PARSING RULES

1. ENERGY VALUES: Extract ALL energies present in the output. Label each with its source method (e.g., "scf", "casscf", "mp2_corr", "mp2_total", "ccsd", "ccsd_t_corr", "ccsd_t_total", "tddft_root0", "tddft_root1"). Use the highest precision available — all decimal places as printed. If an energy is reported in multiple places, use the final converged value.

2. CONVERGENCE: For each iterative method (SCF, CASSCF, geometry optimization), check whether convergence was achieved. Report as boolean (true/false). If the method has no convergence concept, omit the key.

3. ORBITAL ENERGIES: Extract HOMO and LUMO energies. The log may print them in Hartree or eV — convert to eV if in Hartree (1 Ha = 27.2114 eV). Also extract the full orbital energy array if available (first 10 occupied, first 5 virtual).

4. GEOMETRY: If geometry optimization was performed, extract the final geometry, number of optimization steps, and whether it converged.

5. FREQUENCIES: If vibrational analysis was performed, extract all frequencies (cm^-1), IR intensities (km/mol), number of imaginary frequencies, and thermodynamic quantities (ZPE, enthalpy, free energy in Hartree).

6. TIMING: Extract wall time and CPU time in seconds. Extract number of SCF iterations.

7. PROPERTIES: Extract natural orbital occupations (for CASSCF), oscillator strengths (for TD-DFT).

8. ERRORS: If the log contains error messages, include them in extra.errors as an array of strings. Include relevant traceback lines.

9. NULL HANDLING: If a value is expected but not found in the log, set it to null. Never fabricate numbers. If the log is empty or completely unparseable, return empty energy/converged dicts and put the reason in extra.errors.

## OUTPUT FORMAT

Return ONLY a JSON object. No markdown fences. No explanation.

```json
{{
  "energy": {{
    "scf": -109.42345678,
    "mp2_corr": -0.32145678,
    "mp2_total": -109.74491356
  }},
  "converged": {{
    "scf": true,
    "opt": true
  }},
  "extra": {{
    "homo_energy_ev": -15.63,
    "lumo_energy_ev": 2.14,
    "homo_lumo_gap_ev": 17.77,
    "wall_time_s": 3.245,
    "cpu_time_s": 2.987,
    "n_scf_iterations": 8,
    "n_basis_functions": 48,
    "frequencies_cm1": [],
    "ir_intensities_km_mol": [],
    "n_imaginary": 0,
    "thermo_zpe_hartree": null,
    "thermo_enthalpy_hartree": null,
    "thermo_free_energy_hartree": null,
    "orbital_energies_ev": [-15.63, -1.45, -0.82, 2.14, 3.56],
    "mulliken_charges": [0.0, 0.0],
    "errors": []
  }}
}}
```

If log is empty or unparseable:
```json
{{"energy": {{}}, "converged": {{}}, "extra": {{"errors": ["No parseable output found"]}}}}
```
