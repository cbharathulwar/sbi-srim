import os
import time
import pandas as pd
from src.utils.data_generator import run_and_parse_energy

SRIM_DIR    = "/Users/cbharathulwar/Documents/Research/Walsworth/SRIM-2013"
OUTPUT_ROOT = "/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/output_full"
NUM_IONS    = 200
MASTER_CSV  = os.path.join(OUTPUT_ROOT, "vacancies_full_dataset.csv")
LOG_PATH    = os.path.join(OUTPUT_ROOT, "gen_log.txt")

os.makedirs(OUTPUT_ROOT, exist_ok=True)

def energy_schedule():
    # Start at 1 keV, step by 2, up to (and including) 100 keV
    energies = [round(e, 1) for e in range(1, 101, 2)]
    return energies

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

def main():
    energies = energy_schedule()
    dfs = []

    log(f"START | energies={energies} | ions/energy={NUM_IONS}")

    for E in energies:
        per_csv = os.path.join(OUTPUT_ROOT, f"{E:.1f}_with_energy.csv")

        # Resume-skip
        if os.path.exists(per_csv):
            log(f"SKIP  | {E:.1f} keV (found {per_csv})")
            try:
                dfs.append(pd.read_csv(per_csv))
            except Exception as e:
                log(f"WARN  | Could not read {per_csv}: {e}")
            continue

        # Run with one retry guard
        for attempt in (1, 2):
            try:
                log(f"RUN   | {E:.1f} keV | attempt {attempt}")
                df = run_and_parse_energy(E, SRIM_DIR, OUTPUT_ROOT, number_ions=NUM_IONS)
                # sanity: ion count check (optional)
                uniq_ions = df["ion #"].nunique()
                if uniq_ions < NUM_IONS:
                    log(f"WARN  | {E:.1f} keV: ions seen={uniq_ions} < target={NUM_IONS}")
                dfs.append(df)
                log(f"DONE  | {E:.1f} keV | rows={len(df)}")
                break
            except Exception as e:
                log(f"FAIL  | {E:.1f} keV attempt {attempt}: {e}")
                if attempt == 2:
                    log(f"GIVEUP| {E:.1f} keV")
                else:
                    time.sleep(3)

    # Combine everything we have
    if dfs:
        master = pd.concat(dfs, ignore_index=True)
        master.to_csv(MASTER_CSV, index=False)
        log(f"MASTER| saved -> {MASTER_CSV} | total_rows={len(master)}")
    else:
        log("ERROR | no dataframes produced")

if __name__ == "__main__":
    main()