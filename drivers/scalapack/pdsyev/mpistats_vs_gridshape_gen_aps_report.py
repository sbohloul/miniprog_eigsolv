import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
from io import StringIO
import argparse
import os
import json


def generate_report(args):
    
    datadir = args.datadir
    output_file = args.output + ".h5"
    dry_run = args.dry_run

    # Parse aps_results_* directories    
    aps_data = Path(datadir)
    aps_results = list(aps_data.glob("**/aps_result_*"))

    tags_file = os.path.join(datadir, "mpi_regions.txt")
    with open(tags_file, 'r') as f:
        json_data = f.read()
        tag_to_blacs_parameters = json.loads(json_data)

    # =======
    # dry-run
    # =======
    if dry_run:
        print("DRY-RUN ----------")
        print(
            f"datadir: {datadir}\n"
            f"output_file: {output_file}\n"
        )
        print("aps_results:")
        for d in aps_results:
            print(d)

        return
    
    # ====================
    # Available benchmarks
    # ====================
    print("Available benchmarks:")
    for key, val in tag_to_blacs_parameters.items():
        print(f"tag: {key}, parameters: {val}")

    # ===================================
    # MPI Functions summary for all ranks
    # ===================================
    df_name = "functions"
    df = pd.DataFrame()
    aps_command = "aps-report {} --format csv --functions"
    for apsdir in aps_results:
        apsdir_name = apsdir.name
        tag = apsdir_name.split("_")[-1]
        tag = tuple(tag_to_blacs_parameters[tag])
        # 
        print(f"df_name: {df_name}, apsdir_name: {apsdir_name}, tag: {tag}")
        # 
        aps_output = subprocess.run(
            aps_command.format(os.path.join(datadir, apsdir_name)),
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        aps_report = aps_output.stdout.strip().replace("|","")
        # 
        df_tmp = pd.read_csv(
            StringIO(aps_report),
            delimiter=";",
            header=2
        )
        new_index = pd.MultiIndex.from_tuples(
            [(tag, idx) for idx in df_tmp.index]
        )
        df_tmp.index = new_index
        df = pd.concat([df, df_tmp])

    df.to_hdf(output_file, df_name, mode='a')

    # ===========================
    # MPI functions Time per Rank
    # ===========================
    df_name = "mpifunc_per_rank"
    df = pd.DataFrame()
    aps_command = "aps-report {} --format csv --mpi-time-per-rank"
    for apsdir in aps_results:
        apsdir_name = apsdir.name
        tag = apsdir_name.split("_")[-1]
        tag = tuple(tag_to_blacs_parameters[tag])
        # 
        print(f"df_name: {df_name}, apsdir_name: {apsdir_name}, tag: {tag}")
        # 
        aps_output = subprocess.run(
            aps_command.format(os.path.join(datadir, apsdir_name)),
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        aps_report = aps_output.stdout.strip().replace("|","")
        aps_report = aps_report.replace("MPI;Time(sec)", "MPITime(sec)")
        aps_report= aps_report.replace("MPI;Time(%)", " MPITime(%)")
        # 
        df_tmp = pd.read_csv(
            StringIO(aps_report),
            delimiter=";",
            header=1
        )
        new_index = pd.MultiIndex.from_tuples(
            [(tag, idx) for idx in df_tmp.index]
        )
        df_tmp.index = new_index
        df = pd.concat([df, df_tmp])

    df.to_hdf(output_file, df_name, mode='a')

    # ================
    # MPI message size
    # ================
    df_name = "mpi_msgsize"
    df = pd.DataFrame()
    aps_command = "aps-report {} --format csv --message-sizes"
    for apsdir in aps_results:
        apsdir_name = apsdir.name
        tag = apsdir_name.split("_")[-1]
        tag = tuple(tag_to_blacs_parameters[tag])
        # 
        print(f"df_name: {df_name}, apsdir_name: {apsdir_name}, tag: {tag}")
        # 
        aps_output = subprocess.run(
            aps_command.format(os.path.join(datadir, apsdir_name)),
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        aps_report = aps_output.stdout.strip().replace("|","")
        # 
        df_tmp = pd.read_csv(
            StringIO(aps_report),
            delimiter=";",
            header=1
        )
        new_index = pd.MultiIndex.from_tuples(
            [(tag, idx) for idx in df_tmp.index]
        )
        df_tmp.index = new_index
        df = pd.concat([df, df_tmp])

    df.to_hdf(output_file, df_name, mode='a')

    # ===============================
    # Data transfer per communication
    # Rank -> Rank
    # ===============================
    df_name = "transfer_per_communication"
    df = pd.DataFrame()
    aps_command = "aps-report {} --format csv --transfers-per-communication"
    for apsdir in aps_results:
        apsdir_name = apsdir.name
        tag = apsdir_name.split("_")[-1]
        tag = tuple(tag_to_blacs_parameters[tag])
        # 
        print(f"df_name: {df_name}, apsdir_name: {apsdir_name}, tag: {tag}")
        # 
        aps_output = subprocess.run(
            aps_command.format(os.path.join(datadir, apsdir_name)),
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        aps_report = aps_output.stdout.strip().replace("|","")
        aps_report = aps_report.replace("-->;", "")
        aps_report = aps_report.replace("Rank;Rank", "FromRank;ToRank")
        #
        df_tmp = pd.read_csv(
            StringIO(aps_report),
            delimiter=";",
            header=1
        )
        new_index = pd.MultiIndex.from_tuples(
            [(tag, idx) for idx in df_tmp.index]
        )
        df_tmp.index = new_index
        df = pd.concat([
            df, df_tmp
        ])

    df.to_hdf(output_file, df_name, mode='a')

    # ======================
    # Data transfer per rank
    # ======================
    df_name = "transfer_per_rank"
    df = pd.DataFrame()
    aps_command = "aps-report {} --format csv --transfers-per-rank"
    for apsdir in aps_results:
        apsdir_name = apsdir.name
        tag = apsdir_name.split("_")[-1]
        tag = tuple(tag_to_blacs_parameters[tag])
        # 
        print(f"df_name: {df_name}, apsdir_name: {apsdir_name}, tag: {tag}")
        # 
        aps_output = subprocess.run(
            aps_command.format(os.path.join(datadir, apsdir_name)),
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        aps_report = aps_output.stdout.strip().replace("|","")
        #
        df_tmp = pd.read_csv(
            StringIO(aps_report),
            delimiter=";",
            header=1
        )
        new_index = pd.MultiIndex.from_tuples(
            [(tag, idx) for idx in df_tmp.index]
        )
        df_tmp.index = new_index
        df = pd.concat([
            df, df_tmp
        ])

    df.to_hdf(output_file, df_name, mode='a')

    # ==========================
    # Data transfer per function
    # ==========================
    df_name = "transfer_per_func"
    df = pd.DataFrame()
    aps_command = "aps-report {} --format csv --transfers-per-function"
    for apsdir in aps_results:
        apsdir_name = apsdir.name
        tag = apsdir_name.split("_")[-1]
        tag = tuple(tag_to_blacs_parameters[tag])
        # 
        print(f"df_name: {df_name}, apsdir_name: {apsdir_name}, tag: {tag}")
        # 
        aps_output = subprocess.run(
            aps_command.format(os.path.join(datadir, apsdir_name)),
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        aps_report = aps_output.stdout.strip().replace("|","")
        #
        df_tmp = pd.read_csv(
            StringIO(aps_report),
            delimiter=";",
            header=1
        )
        new_index = pd.MultiIndex.from_tuples(
            [(tag, idx) for idx in df_tmp.index]
        )
        df_tmp.index = new_index
        df = pd.concat([
            df, df_tmp
        ])

    df.to_hdf(output_file, df_name, mode='a')

    # ==================
    # Used communicators
    # ==================
    df_name = "communicators"
    df = pd.DataFrame()
    aps_command = "aps-report {} --format csv --communicators-list"
    for apsdir in aps_results:
        apsdir_name = apsdir.name
        tag = apsdir_name.split("_")[-1]
        tag = tuple(tag_to_blacs_parameters[tag])
        # 
        print(f"df_name: {df_name}, apsdir_name: {apsdir_name}, tag: {tag}")
        # 
        aps_output = subprocess.run(
            aps_command.format(os.path.join(datadir, apsdir_name)),
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        aps_report = aps_output.stdout.strip().replace("|","")
        #
        df_tmp = pd.read_csv(
            StringIO(aps_report),
            delimiter=";",
            header=1
        )
        new_index = pd.MultiIndex.from_tuples(
            [(tag, idx) for idx in df_tmp.index]
        )
        df_tmp.index = new_index
        df = pd.concat([
            df, df_tmp
        ])

    df.to_hdf(output_file, df_name, mode='a')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Runs aps-report on the result directories created by aps which are named as "aps_results_tag" 
        corresponding to benchmarks with various process grid and block size parameters. Different generted
        reports are converted to pandas data frames and stored in a h5 file as "output_filename.h5".
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("datadir",
                        type=str,
                        help='Directory containing "aps_results_* folders"'
                    )
    parser.add_argument("-o", "--output",
                        type=str,
                        default="mpistats_vs_gridshape",
                        help="Output file name"
                    )
    parser.add_argument('--dry-run',
                        action='store_true',
                        help="Simulate the process without making actual changes"
                        )

    args = parser.parse_args()
    generate_report(args)