# neoslam_loop_example.py
"""
Exemplo inspirado no `fabio_example.py`, modelando o "problema conceitual"
da figura `NeoSLAM__problema_conceitual.png`:

    O robô parte de 0 e percorre um loop fechado com 6 spots:
        0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 0 -> 1 -> ...

Diferenças em relação ao `fabio_example.py`:
    * 6 spots (0 a 5). Cada spot recebe um SDR próprio (um bloco de
      BITS_PER_SPOT bits ativos), ou seja, uma representação 1:1
      equivalente à sua representação de número inteiro;
    * o HTM recebe inicialmente a travessia completa do loop
      0, 1, 2, 3, 4, 5, 0 (e depois continua girando o loop);
    * a cada iteração são impressas as células ATIVAS, as WINNER cells
      e as células PREDITAS (mais o score de anomalia).

Observação sobre o `TemporalMemory`:
    `tm.compute(x)` roda `activateDendrites()` ANTES de `activateCells()`,
    então as células preditas que interessam (a expectativa para o PRÓXIMO
    input) são obtidas chamando `tm.activateDendrites()` depois do compute —
    é exatamente o que o `fabio_example.py` faz.
"""

import numpy as np

from htm.bindings.sdr import SDR
from htm.algorithms import TemporalMemory as TM


# ================================================================
# 1. Configuração
# ================================================================
# Spots do loop da figura.
SPOT_IDS = [0, 1, 2, 3, 4, 5]

# Quantos bits ativos representam cada spot (bloco contíguo, 1:1 com o inteiro).
BITS_PER_SPOT = 8

# Espaço de mini-colunas do HTM: 1 mini-coluna por bit do SDR.
N_COLUMNS = len(SPOT_IDS) * BITS_PER_SPOT          # 6 * 8 = 48

# Células por mini-coluna. O `fabio_example.py` usa 1; aqui usamos > 1 para
# que as winner cells sejam realmente distintas das células ativas e para que
# o TM consiga representar o "0" em contextos diferentes (início vs. após 5).
CELLS_PER_COLUMN = 4

# Uma volta completa no loop: 0, 1, 2, 3, 4, 5, 0
ONE_LAP = SPOT_IDS + [SPOT_IDS[0]]

# Número de voltas. O HTM começa exatamente pela sequência 0,1,2,3,4,5,0.
N_LAPS = 5

# Stream contínuo (loop sem emenda artificial entre voltas):
#   0,1,2,3,4,5,0,1,2,3,4,5,0,...
STREAM = (SPOT_IDS * N_LAPS) + [SPOT_IDS[0]]

# Parâmetros do TM (iguais aos do fabio_example.py).
TM_PARAMS = dict(
    minThreshold=4,          # default: 10
    activationThreshold=8,   # default: 13
    initialPermanence=0.5,   # default: 0.21
    seed=42,
)


# ================================================================
# 2. Encoder: inteiro (spot) -> SDR
# ================================================================
def sdr_for_spot(spot: int, sdr: SDR) -> SDR:
    """
    Escreve no `sdr` a representação do spot: um bloco contíguo de
    BITS_PER_SPOT bits ativos começando em `spot * BITS_PER_SPOT`.

        spot 0 -> bits [ 0.. 7]
        spot 1 -> bits [ 8..15]
        ...
        spot 5 -> bits [40..47]
    """
    bits = np.zeros(sdr.size, dtype=np.uint8)
    start = spot * BITS_PER_SPOT
    bits[start:start + BITS_PER_SPOT] = 1
    sdr.dense = bits
    return sdr


# ================================================================
# 3. Utilitários de impressão
# ================================================================
def format_bits(sdr: SDR, group: int) -> str:
    """String de bits do SDR, com um espaço a cada `group` bits."""
    dense = np.asarray(sdr.dense).ravel()
    out = []
    for i, bit in enumerate(dense):
        if i > 0 and i % group == 0:
            out.append(" ")
        out.append(str(int(bit)))
    return "".join(out)


def cells_by_spot_lines(sdr: SDR, group: int) -> list:
    """
    Bits de um SDR de CÉLULAS agrupados por spot. Devolve uma linha por spot
    com bits não-nulos, p.ex. ``"spot 0 = 11111111111111111111111111111111"``.
    """
    dense = np.asarray(sdr.dense).ravel()
    lines = []
    for start in range(0, dense.size, group):
        chunk = dense[start:start + group]
        if np.any(chunk):
            spot = start // group
            lines.append(f"spot {spot} = {''.join(str(int(b)) for b in chunk)}")
    return lines or ["(vazio)"]


def spots_from_cells(cell_sdr: SDR, tm: TM) -> list:
    """
    Converte um SDR de CÉLULAS no conjunto de spots correspondentes.

    Célula -> coluna -> spot:  spot = coluna // BITS_PER_SPOT
    """
    if cell_sdr.getSum() == 0:
        return []
    columns_sdr = tm.cellsToColumns(cell_sdr)
    return sorted({int(col) // BITS_PER_SPOT for col in columns_sdr.sparse})


def print_step_header(step: int, lap: int, spot: int) -> None:
    print(f"\n--- passo {step:02d} | volta {lap} | spot de entrada: {spot} "
          f"---")


# ================================================================
# 4. Main
# ================================================================
def main() -> None:
    print("=" * 78)
    print(" NeoSLAM - problema conceitual: loop fechado com 6 spots")
    print("=" * 78)
    print(f"Spots            : {SPOT_IDS}")
    print(f"Bits por spot    : {BITS_PER_SPOT}")
    print(f"Mini-colunas     : {N_COLUMNS}")
    print(f"Células por col. : {CELLS_PER_COLUMN}  "
          f"(total de células = {N_COLUMNS * CELLS_PER_COLUMN})")
    print(f"Sequência        : {STREAM}")
    print(f"Voltas           : {N_LAPS}")
    print("\nMapa spot -> bits ativos -> colunas:")
    for spot in SPOT_IDS:
        lo = spot * BITS_PER_SPOT
        hi = lo + BITS_PER_SPOT - 1
        print(f"  spot {spot} -> bits [{lo:>2}..{hi:>2}] -> colunas [{lo:>2}..{hi:>2}]")

    # ---- Cria o Temporal Memory ----
    tm = TM(
        columnDimensions=(N_COLUMNS,),
        cellsPerColumn=CELLS_PER_COLUMN,
        **TM_PARAMS,
    )
    input_sdr = SDR(N_COLUMNS)

    # Grupo de bits que corresponde a um spot dentro de um SDR de células:
    # (8 colunas) x (CELLS_PER_COLUMN células por coluna)
    cell_group = BITS_PER_SPOT * CELLS_PER_COLUMN

    print("\nLegenda: active/winner/predicted estão agrupados por spot "
          f"({cell_group} bits por grupo).\n")

    # Para medir se a predição do passo t acerta o spot do passo t+1.
    # pred_ok[t] = True se a predição feita no passo t acertou o spot do passo t+1
    pred_ok = []
    anomaly_per_step = []

    for step, spot in enumerate(STREAM):
        lap = min(step // len(SPOT_IDS) + 1, N_LAPS)

        # ---- 1. Codifica o spot como SDR e entrega ao HTM ----
        sdr_for_spot(spot, input_sdr)
        tm.compute(input_sdr, learn=True)
        tm.activateDendrites(True)

        # ---- 2. Coleta os estados pedidos ----
        active = tm.getActiveCells()
        winner = tm.getWinnerCells()
        predicted = tm.getPredictiveCells()
        anomaly = float(tm.anomaly)

        active_spots = spots_from_cells(active, tm)
        winner_spots = spots_from_cells(winner, tm)
        predicted_spots = spots_from_cells(predicted, tm)

        anomaly_per_step.append(anomaly)

        # ---- 3. Impressão ----
        print_step_header(step, lap, spot)
        print(f"  input     : {format_bits(input_sdr, BITS_PER_SPOT)}")
        print(f"  active    : {active.getSum():>4} células | spots={active_spots}")
        for line in cells_by_spot_lines(active, cell_group):
            print(f"      {line}")
        print(f"  winner    : {winner.getSum():>4} células | spots={winner_spots}")
        for line in cells_by_spot_lines(winner, cell_group):
            print(f"      {line}")
        print(f"  predicted : {predicted.getSum():>4} células | "
              f"spots={predicted_spots}")
        for line in cells_by_spot_lines(predicted, cell_group):
            print(f"      {line}")
        print(f"  anomaly   : {anomaly:.3f}")

        # ---- 4. A predição do passo t acerta o spot do passo t+1? ----
        if step + 1 < len(STREAM):
            pred_ok.append(STREAM[step + 1] in predicted_spots)

    # ---- Resumo final ----
    print("\n" + "=" * 78)
    print(" RESUMO")
    print("=" * 78)

    n = len(anomaly_per_step)
    first_lap_len = len(SPOT_IDS)
    n_checks = len(pred_ok)
    n_hits = int(sum(pred_ok))
    after = pred_ok[first_lap_len:]

    print(f"Passos executados             : {n}")
    if n_checks:
        print(f"Acerto de predição (total)    : {n_hits}/{n_checks} "
              f"({100.0 * n_hits / n_checks:.1f} %)")
    if after:
        print(f"Acerto após a 1ª volta        : {sum(after)}/{len(after)} "
              f"({100.0 * sum(after) / len(after):.1f} %)")
    print(f"Anomalia média (1ª volta)     : "
          f"{np.mean(anomaly_per_step[:first_lap_len]):.3f}")
    print(f"Anomalia média (demais)       : "
          f"{np.mean(anomaly_per_step[first_lap_len:]):.3f}")

    # Ponto central do "problema conceitual": o PRIMEIRO retorno ao spot 0
    # (transição 5 -> 0, nunca vista antes) é anômalo; depois o TM aprende a
    # sequência do loop e passa a PREVER o retorno (anomalia ~0).
    closure_steps = [s for s, spot in enumerate(STREAM)
                     if spot == SPOT_IDS[0] and s > 0]
    if closure_steps:
        print("\nRetorno ao spot 0 (loop closures) e anomalia:")
        for i, s in enumerate(closure_steps, start=1):
            tag = "   <- 1º loop closure (5->0 inédito)" if i == 1 else ""
            print(f"  retorno {i} (passo {s:>2}): "
                  f"anomalia = {anomaly_per_step[s]:.3f}{tag}")

    print("\nSequência aprendida pelo HTM (loop fechado):")
    print("  " + " -> ".join(str(s) for s in ONE_LAP))


if __name__ == "__main__":
    main()
