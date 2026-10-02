# neoslam_loop_example.py
"""
Versao simples, no estilo do `fabio_example.py`, agora com 2 celulas por coluna.

Rede HTM:
    cellsPerColumn = 2, minThreshold = 4, activationThreshold = 8,
    initialPermanence = 0.5

Representacao dos spots (1:1 com a representacao do numero inteiro):
    spot 0 -> colunas [ 0.. 7]
    spot 1 -> colunas [ 8..15]
    ...
    spot 5 -> colunas [40..47]
    (6 spots x 8 bits = 48 colunas do HTM)

Cada coluna tem CELLS_PER_COLUMN (= 2) celulas. O indice global da celula e
    indice = coluna * CELLS_PER_COLUMN + celula_da_coluna
Por isso, para cada estado (Active / Predicted / Winner) sao impressas
CELLS_PER_COLUMN linhas de 48 bits:
    "celula 0" -> mostra a 1a celula de cada uma das 48 colunas
    "celula 1" -> mostra a 2a celula de cada uma das 48 colunas

Sequencia: 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0
"""

import numpy as np
from htm.bindings.sdr import SDR
from htm.algorithms import TemporalMemory as TM


# ------------------------------------------------------------------
# Formatacao: `sdr.dense` de um SDR de celulas tem shape
# (n_colunas, cells_per_column). Entao `dense[:, layer]` da os n_colunas
# bits daquela celula -> uma linha de 48 bits por celula.
# ------------------------------------------------------------------
def formatSdrLayer(sdr, layer, cells_per_column, bits_per_spot=8):
    dense = np.asarray(sdr.dense).reshape(-1, cells_per_column)
    bits = dense[:, layer]
    result = ''
    for i, bit in enumerate(bits):
        if i > 0 and i % bits_per_spot == 0:
            result += ' '
        result += str(int(bit))
    return result


def print_cell_lines(label, sdr, cells_per_column):
    """Imprime uma linha de 48 bits para cada celula dentro da coluna."""
    for layer in range(cells_per_column):
        tag = f"(celula {layer})"
        print(f"  {label:<10}{tag:<12}: "
              f"{formatSdrLayer(sdr, layer, cells_per_column)}")


# ------------------------------------------------------------------
# Configuracao
# ------------------------------------------------------------------
BITS_PER_SPOT = 8
N_SPOTS = 6
arraySize = N_SPOTS * BITS_PER_SPOT      # 48 colunas
CELLS_PER_COLUMN = 2                     # 2 celulas por coluna

# Sequencia curta: 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 0
# cycleArray = [0, 1, 2, 3, 4, 5, 0]
# Para repetir o loop varias vezes, use por exemplo:
cycleArray = ([0, 1, 2, 3, 4, 5] * 2) + [2, 1, 0, 5]

inputSDR = SDR(arraySize)

print("spots     :", list(range(N_SPOTS)))
print("colunas   :", arraySize, f"({BITS_PER_SPOT} bits por spot)")
print("cells/col :", CELLS_PER_COLUMN)
print("sequencia :", cycleArray)
print()
print("Legenda dos bits (colunas):")
for spot in range(N_SPOTS):
    lo = spot * BITS_PER_SPOT
    hi = lo + BITS_PER_SPOT - 1
    print(f"  spot {spot} -> colunas {lo:>2}..{hi:>2}")
print()

# ------------------------------------------------------------------
# HTM: 2 celulas por coluna
# ------------------------------------------------------------------
tm = TM(columnDimensions=(inputSDR.size,),
        cellsPerColumn=CELLS_PER_COLUMN,   # 2
        minThreshold=4,          # default: 10
        activationThreshold=8,   # default: 13
        initialPermanence=0.5,   # default: 0.21
        )

# ------------------------------------------------------------------
# Executa a sequencia
# ------------------------------------------------------------------
for sensorValue in cycleArray:
    # 1. Codifica o spot como SDR (bloco de 8 bits) e entrega ao HTM
    sensorValueBits = np.zeros(arraySize)
    sensorValueBits[sensorValue * BITS_PER_SPOT:
                    sensorValue * BITS_PER_SPOT + BITS_PER_SPOT] = 1
    inputSDR.dense = sensorValueBits

    tm.compute(inputSDR, learn=True)

    # 2. Imprime os bits de Active / Predicted / Winner
    #    (uma linha de 48 bits por celula dentro da coluna)
    print(f"=== V: {sensorValue:>2} ===")
    print_cell_lines("Active", tm.getActiveCells(), CELLS_PER_COLUMN)

    tm.activateDendrites(True)
    print_cell_lines("Predicted", tm.getPredictiveCells(), CELLS_PER_COLUMN)
    print_cell_lines("Winner", tm.getWinnerCells(), CELLS_PER_COLUMN)
    print(f"  {'anomaly':<10}{'':<12}: {tm.anomaly:.2f}")
    print()
