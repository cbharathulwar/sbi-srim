import os
import time
import pandas as pd
from ..utils.data_generator import run_and_parse_energy

# --- CONFIG ---------------------------------------------------
SRIM_DIR = r"C:\Users\walsworth.admin\Desktop\SRIM-2013"
OUTPUT_ROOT = r"C:\Users\walsworth.admin\Documents\Chinmay\sbi-srim-main\srim_sbi\data\nov3srim"
NUM_IONS    = 200   # quick test
MASTER_CSV  = os.path.join(OUTPUT_ROOT, "vacancies.csv")
LOG_PATH    = os.path.join(OUTPUT_ROOT, "gen_log.txt")

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# --- ENERGY RANGE (tiny subset) -------------------------------
def energy_schedule():
    return [
        1, 2.5, 5, 7.5, 10,     # fine spacing at low keV
        15, 20, 25, 30, 35,     
        40, 45, 50, 55, 60, 65, 70, 75, 80, 85,     
        90, 95, 100                 
    ]

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
        # If a CSV already exists, load it
        if os.path.exists(per_csv):
            log(f"SKIP  | {E:.1f} keV (found {per_csv})")
            try:
                dfs.append(pd.read_csv(per_csv))
            except Exception as e:
                log(f"WARN  | Could not read {per_csv}: {e}")
            continue
        # Only one attempt — no retry
        try:
            log(f"RUN   | {E:.1f} keV")
            df = run_and_parse_energy(E, SRIM_DIR, OUTPUT_ROOT, number_ions=NUM_IONS)
            uniq_ions = df["ion #"].nunique()
            log(f"DONE  | {E:.1f} keV | rows={len(df)} | unique_ions={uniq_ions}")
            dfs.append(df)
        except Exception as e:
            log(f"FAIL  | {E:.1f} keV: {e}")
            log(f"GIVEUP| {E:.1f} keV")

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