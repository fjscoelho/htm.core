# neoslam_loop_example.py
"""
Simple example in the style of `fabio_example.py`, now with 2 cells per column.

HTM network:
    cellsPerColumn = 2, minThreshold = 4, activationThreshold = 8,
    initialPermanence = 0.5

Spot representation (1:1 with the integer number representation):
    spot 0 -> columns [ 0.. 7]
    spot 1 -> columns [ 8..15]
    ...
    spot 5 -> columns [40..47]
    (6 spots x 8 bits = 48 HTM columns)

Each column has CELLS_PER_COLUMN (= 2) cells. The global cell index is
    index = column * CELLS_PER_COLUMN + cell_within_column
Therefore, for each state (Active / Predicted / Winner), CELLS_PER_COLUMN
lines of 48 bits are printed:
    "cell 0" -> shows the 1st cell of each of the 48 columns
    "cell 1" -> shows the 2nd cell of each of the 48 columns

The "Input" line (above "Active") shows the input SDR of the iteration,
i.e. the 48 active mini-columns fed to the TM.

Sequence: [0, 1, 2, 3, 4, 5] * 2 + [2, 1, 0, 5]
"""

import numpy as np
from htm.bindings.sdr import SDR
from htm.algorithms import TemporalMemory as TM


# ------------------------------------------------------------------
# Formatting: `sdr.dense` of a cell SDR has shape
# (n_columns, cells_per_column). So `dense[:, layer]` gives the n_columns
# bits of that cell -> one line of 48 bits per cell.
# ------------------------------------------------------------------
def formatBits(bits, bits_per_spot=8):
    """Bit string with a space every `bits_per_spot` bits."""
    result = ''
    for i, bit in enumerate(bits):
        if i > 0 and i % bits_per_spot == 0:
            result += ' '
        result += str(int(bit))
    return result


def formatSdrLayer(sdr, layer, cells_per_column, bits_per_spot=8):
    dense = np.asarray(sdr.dense).reshape(-1, cells_per_column)
    return formatBits(dense[:, layer], bits_per_spot)


def print_cell_lines(label, sdr, cells_per_column):
    """Print one line of 48 bits for each cell within the column."""
    for layer in range(cells_per_column):
        tag = f"(layer {layer})"
        print(f"  {label:<10}{tag:<12}: "
              f"{formatSdrLayer(sdr, layer, cells_per_column)}")


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
BITS_PER_SPOT = 8
N_SPOTS = 6
arraySize = N_SPOTS * BITS_PER_SPOT      # 48 columns
CELLS_PER_COLUMN = 2                     # 2 cells per column

# Short sequence: 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 0
# cycleArray = [0, 1, 2, 3, 4, 5, 0]
# To repeat the loop several times, use for example:
cycleArray = ([0, 1, 2, 3, 4, 5] * 2) + [2, 1, 0, 5]

inputSDR = SDR(arraySize)

print("spots     :", list(range(N_SPOTS)))
print("columns   :", arraySize, f"({BITS_PER_SPOT} bits per spot)")
print("cells/col :", CELLS_PER_COLUMN)
print("sequence  :", cycleArray)
print()
print("Bit legend (columns):")
for spot in range(N_SPOTS):
    lo = spot * BITS_PER_SPOT
    hi = lo + BITS_PER_SPOT - 1
    print(f"  spot {spot} -> columns {lo:>2}..{hi:>2}")
print()

# ------------------------------------------------------------------
# HTM: 2 cells per column
# ------------------------------------------------------------------
tm = TM(columnDimensions=(inputSDR.size,),
        cellsPerColumn=CELLS_PER_COLUMN,   # 2
        minThreshold=4,          # default: 10
        activationThreshold=8,   # default: 13
        initialPermanence=0.5,   # default: 0.21
        )

# ------------------------------------------------------------------
# Run the sequence
# ------------------------------------------------------------------
for sensorValue in cycleArray:
    # 1. Encode the spot as an SDR (block of 8 bits) and feed it to the HTM
    sensorValueBits = np.zeros(arraySize)
    sensorValueBits[sensorValue * BITS_PER_SPOT:
                    sensorValue * BITS_PER_SPOT + BITS_PER_SPOT] = 1
    inputSDR.dense = sensorValueBits

    tm.compute(inputSDR, learn=True)

    # 2. Print the input SDR and the bits of Active / Predicted / Winner
    #    (one line of 48 bits per cell within the column)
    print(f"=== Spot: {sensorValue:>2} ===")
    print(f"  {'Input':<10}{'(columns)':<12}: {formatBits(sensorValueBits)}")
    print(f"  -----------------------------------------------------------------------------")
    print_cell_lines("Active", tm.getActiveCells(), CELLS_PER_COLUMN)
    print(f"  -----------------------------------------------------------------------------")
    tm.activateDendrites(True)
    print_cell_lines("Predicted", tm.getPredictiveCells(), CELLS_PER_COLUMN)
    print(f"  -----------------------------------------------------------------------------")
    print_cell_lines("Winner", tm.getWinnerCells(), CELLS_PER_COLUMN)
    print(f"  -----------------------------------------------------------------------------")
    # anomaly = fraction of active columns that were NOT predicted (0..1)
    print(f"  {'anomaly':<10}{'':<12}: {tm.anomaly:.2f}")
    print()
