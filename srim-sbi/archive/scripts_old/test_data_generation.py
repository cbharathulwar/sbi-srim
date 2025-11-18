import os
import time
import pandas as pd
from src.utils.data_generator import run_and_parse_energy

# --- CONFIG ---------------------------------------------------
SRIM_DIR    = "/Users/cbharathulwar/Documents/Research/Walsworth/SRIM-2013"
OUTPUT_ROOT = "/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/output_quicktest"
NUM_IONS    = 3   # ✅ just 3 ions per energy for a quick test
MASTER_CSV  = os.path.join(OUTPUT_ROOT, "vacancies_quicktest.csv")
LOG_PATH    = os.path.join(OUTPUT_ROOT, "gen_log.txt")

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# --- ENERGY RANGE (tiny subset) -------------------------------
def energy_schedule():
    # Only 3 quick energies — fast verification
    return [5.0, 10.0, 20.0]

# --- LOGGING --------------------------------------------------
def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

# --- MAIN LOOP ------------------------------------------------
def main():
    energies = energy_schedule()
    dfs = []

    log(f"START | energies={energies} | ions/energy={NUM_IONS}")

    for E in energies:
        per_csv = os.path.join(OUTPUT_ROOT, f"{E:.1f}_with_energy.csv")

        if os.path.exists(per_csv):
            log(f"SKIP  | {E:.1f} keV (found {per_csv})")
            try:
                dfs.append(pd.read_csv(per_csv))
            except Exception as e:
                log(f"WARN  | Could not read {per_csv}: {e}")
            continue

        for attempt in (1, 2):
            try:
                log(f"RUN   | {E:.1f} keV | attempt {attempt}")
                df = run_and_parse_energy(E, SRIM_DIR, OUTPUT_ROOT, number_ions=NUM_IONS)
                uniq_ions = df["ion #"].nunique()
                log(f"DONE  | {E:.1f} keV | rows={len(df)} | unique_ions={uniq_ions}")
                dfs.append(df)
                break
            except Exception as e:
                log(f"FAIL  | {E:.1f} keV attempt {attempt}: {e}")
                if attempt == 2:
                    log(f"GIVEUP| {E:.1f} keV")
                else:
                    time.sleep(2)

    # Combine everything
    if dfs:
        master = pd.concat(dfs, ignore_index=True)
        master.to_csv(MASTER_CSV, index=False)
        log(f"MASTER| saved -> {MASTER_CSV} | total_rows={len(master)}")
    else:
        log("ERROR | no dataframes produced")

# --- ENTRYPOINT -----------------------------------------------
if __name__ == "__main__":
    main()