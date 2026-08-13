

# Welcome to the Overtoner, a quantum computation tool for deriving molar extinction coefficients for IR/NIR peaks.

## This CLI performs the following quantum chemistry and Morse Model-based anharmonicty calulations to determine the molar extinction coefficent ε in $\text{M·cm}^{-1}$ for any organic molecule’s IR (or NIR) overtone peak. Its algorithm can even compute ε values for fundamnetal peaks, at full anharmonic accuracy.

## Inputs required

- Molar mass (amu) of element A in an A-B stretch
- Molar mass (amu) of element B in an A-B stretch
- Fundamental frequency of molecule in $\text{cm}^{-1}$ (wavenumber)
- Observed frequency of molecule in $\text{cm}^{-1}$ (wavenumber)
- Approximate integer overtone order of the molecule’s observed
  wavenumber relative to the fundamental wavenumber

## Morse model parameters used

For an IR (or NIR) overtone transition from $v=0$ to $v=n$, the solver
assumes the standard Morse-oscillator relation between the fundamental
frequency $\nu_e$, the anharmonicity constant $x_e$, and the observed
overtone wavenumber $\nu_{0\to n}$:

$$
\nu_{0\to n} \approx n\,\nu_e - n(n+1)\,\nu_e x_e.
$$

Solving for $x_e$ in terms of the fundamental and observed overtone
gives

$$
x_e = \frac{n\,\nu_e - \nu_{0\to n}}{n(n+1)\,\nu_e}.
$$

From $\nu_e$ and $x_e$, the dissociation energy in wavenumber units is
taken as the magnitude

$$
D_e^{\mathrm{cm}^{-1}} = \frac{\nu_e}{4\,|x_e|},
$$

which avoids sign-convention issues that can make $D_e$ negative. This
$D_e^{\mathrm{cm}^{-1}}$ is then converted to Joules and used, together
with the reduced mass, to construct the Morse parameters $a$ and
$\lambda$ that enter the high-precision overlap and intensity
calculations.

## Allowed organic stretches:

- C–H
- C=O
- C–N
- N–H
- O–H

## Installation

### Prerequisites

- **Conda** or **Miniconda**: Required for environment management
- Download from [Anaconda](https://www.anaconda.com/download) or
  [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- **Git**: Required for cloning the repository

### Step-by-Step Installation

1.  **Clone the Repository**

<!-- -->

    git clone https://github.com/lukelevensaler/Organic-Morse-Solver.git
    cd Organic-Morse-Solver

2.  **Create the Conda Environment**

The repository includes an `environment.yml` file that specifies all
required dependencies including: - **PySCF**: Quantum chemistry
calculations (SCF-level theory in the current implementation) -
**NumPy/SciPy**: Numerical computations and special functions -
**Typer**: CLI framework - **PyBerny**: Geometry optimization -
**H5PY**: Data storage for quantum chemistry results - **High-Precision
Libraries**: Optimized BLAS/LAPACK for numerical stability - **Parallel
Computing**: MPI support for distributed quantum chemistry calculations

Create the environment named `morse_solver`:
`conda env create -f environment.yml`

3.  **Activate the Environment**

<!-- -->

    conda activate morse_solver

4.  **Verify Installation**

Test that the CLI works correctly: `python run_morse_model.py stretches`

You should see the list of allowed organic stretches:
`Allowed organic stretches:    - C–H    - C=O    - C–N    - N–H    - O–H`

### Usage After Installation

Once installed, you can run the solver from the repository directory:

    # Activate the environment (if not already active)
    conda activate morse_solver

    # Run the CLI
    python3 run_morse_model.py compute --help

### Troubleshooting

**Common Issues:**

1.  **Conda environment creation fails**: Ensure you have sufficient
    disk space (~2GB) and internet connectivity
2.  **PySCF import errors**: The environment includes all required
    quantum chemistry dependencies
3.  **Permission errors**: Ensure you have write access to your conda
    installation directory
    

------------------------------------------------------------------------

### Geometry Input Options

#### Option 1: Direct molecular information input via the interactive CLI

``` bash
# Example of how the inetractive prompt handles:
Enter molecular coordinates (Element x y z format):
C 0.000000 0.000000 0.000000
H 1.100000 0.000000 0.000000
O -1.200000 0.000000 0.000000
H -1.800000 0.800000 0.000000
[blank line to finish]
```

#### Option 2: File input (+ other necesary molecular parameters in interactive mode)

``` bash
# Create coordinates file (e.g., molecule.xyz):
cat > molecule.xyz << EOF
C 0.000000 0.000000 0.000000
H 1.100000 0.000000 0.000000
O -1.200000 0.000000 0.000000
H -1.800000 0.800000 0.000000
EOF

# Then use in batch mode:
python3 run_morse_model.py compute --coords molecule.xyz [other parameters...]
```

### Option 3: Direct input of all coordinates and parameters (Advanced)

#### Examples:

#### C-H Stretch in Methane:

``` bash
python3 run_morse_model.py compute \
 --m1 12.011 --m2 1.008 \
 --fundamental 2917 --observed 8750 --overtone 3 \
 --coords "C 0.0 0.0 0.0\nH 1.09 0.0 0.0\nH -0.36 1.03 0.0\nH -0.36 -0.51 0.89\nH -0.36 -0.51 -0.89" \
 --specified-spin 0 --bond "0,1"
```

#### O-H Stretch in Water:

``` bash
python3 run_morse_model.py compute \
 --m1 15.999 --m2 1.008 \
 --fundamental 3657 --observed 10935 --overtone 3 \
 --coords "O 0.0 0.0 0.0\nH 0.757 0.587 0.0\nH -0.757 0.587 0.0" \
 --specified-spin 0 --bond "0,1"
```

#### Dual Bond System:

For molecules with symmetric stretching modes (the semicolon between
bond axes is CRUCIAL):

``` bash
python3 run_morse_model.py compute \
 --dual-bonds "(0,2);(1,2)" \
 --m1 12.011 --m2 15.999 \
 [other parameters...]
```

#### Fundamental peak ε values can also be determined with this software, if the overtone order is set to 0 and the observed frequency input is the same as the fundamental frequency:

``` bash
python3 run_morse_model.py compute \
 --m1 15.999 --m2 1.008 \
 --fundamental 3657 --observed 3657 --overtone 0 \
 --coords "O 0.0 0.0 0.0\nH 0.757 0.587 0.0\nH -0.757 0.587 0.0" \
 --specified-spin 0  --bond "0,1"
```

#### *NOTE: ALL of the above examples are just arbitrary numbers, not actual valid data, including the hypothetical methane example. DO NOT USE THOSE DEMONSTRATION NUMBERS IN ACTUAL SCIENTIFIC RESEARCH!*

### What Is Basis Set Selection?

The `--basis` flag allows you to control the quantum chemistry basis set
used for SCF calculations:

**Default (Recommended):** `aug-cc-pVTZ` - High accuracy for most
organic molecules - Good balance of precision and computational cost -
Suitable for production calculations

**Higher Accuracy:** `aug-cc-pVQZ` - Maximum precision for critical
applications - Significantly longer computation time - Recommended for
benchmarking or when highest accuracy is needed

**Faster Computation:** `cc-pVDZ` - Reduced accuracy but much faster -
Useful for testing, debugging, or large systems - Not recommended for
final results

**Example with custom basis set:**

``` bash
python3 run_morse_model.py compute --basis aug-cc-pVQZ [other parameters...]
```

### Output

The CLI provides detailed output including: - SCF geometry optimization
results - Computed dipole derivatives - Morse model parameters - Final
molar extinction coefficient

------------------------------------------------------------------------

## How To Use the CLI

The Morse solver provides two usage modes: **batch mode** (all
parameters at once) and **interactive mode** (step-by-step prompts).

### Batch Mode (All Parameters at Once)

Provide all required parameters in a single command for automated
workflows:

``` bash
python3 \
 --m1 12.011 \
 --m2 1.008 \
 --fundamental 2900.0 \
 --observed 8700.0 \
 --overtone 3 \
 --coords "C 0.0 0.0 0.0\nH 1.1 0.0 0.0" \
 --specified-spin 0 \
 --bond "0,1" \
 --delta 0.005 \
 --basis aug-cc-pVTZ \
 --fwhm 75.0
```

**Parameters:** - `--m1`, `--m2`: Atomic masses (amu) for elements A and
B - `--fundamental`: Fundamental frequency (cm⁻¹) - `--observed`:
Observed overtone frequency (cm⁻¹) - `--overtone`: Integer overtone
number (n for 0→n transition) - `--coords`: Molecular geometry in XYZ
format (quoted multiline string) - `--specified-spin`: Spin multiplicity
(0 for singlet, 1 for doublet, etc.) - `--bond`: Bond atom indices as
“i,j” (0-based) - `--delta`: Finite difference displacement (Angstrom,
default: 0.005) - `--basis`: Quantum chemistry basis set (default:
aug-cc-pVTZ). Can be overridden with higher quality sets like
aug-cc-pVQZ or aug-cc-pV5Z for maximum accuracy. - `--fwhm`: Line width
for peak extinction (cm⁻¹, default: 75.0)

### Interactive Mode (Step-by-Step)

Run without parameters for guided input:

``` bash
python3 run_morse_solver
```

The CLI will prompt for each parameter:

    Morse Solver for IR (or NIR) Overtone Extinction Coefficients

    Enter atomic mass of element A (amu): 12.011
    Enter atomic mass of element B (amu): 1.008
    Enter fundamental frequency (cm⁻¹): 2900.0
    Enter observed frequency (cm⁻¹): 8700.0
    Enter overtone number (integer): 3

    📐 Molecular Geometry Input
    Choose input method:
    1. Type coordinates directly
    2. Load from file
    Selection: 1

    Enter molecular coordinates (Element x y z format, blank line to finish):
    C 0.0 0.0 0.0
    H 1.1 0.0 0.0
    [blank line]

    Enter spin multiplicity: 0
    Enter bond atom indices (i,j format): 0,1

    Advanced Options (press Enter for defaults)
    Finite difference step size (Å) [0.005]:
    Basis set [aug-cc-pVTZ]: aug-cc-pVQZ
