"""
Ths script loads "fully ordered" parquet files as Pandas DataFrames and changes the common prefix
of all entries of the 'file_path' column to a new one.
"""

import argparse
from pathlib import Path

import pandas as pd


def main(args: argparse.Namespace):
    # checks
    condition = (args.previous_dataset_prefix.endswith("/") and args.new_dataset_prefix.endswith("/")) or (
        not args.previous_dataset_prefix.endswith("/") and not args.new_dataset_prefix.endswith("/")
    )
    if not condition:
        raise ValueError(
            f"Either both previous and new dataset prefixes should end with '/', or none of them should; got: previous_dataset_prefix={args.previous_dataset_prefix} and new_dataset_prefix={args.new_dataset_prefix}"
        )
    if "/" not in args.substring_to_remove:
        print("WARNING: substring_to_remove has no '/' in it!")
    # misc
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", None)
    # run
    print(f"Processing files:\n{'\n'.join(args.dataset_files_list)}")
    for ds_files in args.dataset_files_list:
        # load and check
        df = pd.read_parquet(ds_files)
        if not df["file_path"].str.startswith(args.previous_dataset_prefix).all():
            raise ValueError(f"Not all file paths start with '{args.previous_dataset_prefix}' in {ds_files}")
        # store new paths
        new_file_paths = df["file_path"].str.removeprefix(args.previous_dataset_prefix)
        new_file_paths = new_file_paths.str.replace(args.substring_to_remove, "", regex=False)
        new_file_paths = args.new_dataset_prefix + new_file_paths
        # checks
        if args.debug:
            print("\nWould change:")
            print(df["file_path"])
            print("to:")
            print(new_file_paths)
        print(f"\nChecking that {len(new_file_paths)} files exist...", end="", flush=True)
        if not new_file_paths.apply(lambda p: Path(p).exists()).all():
            print("")
            missing = new_file_paths[~new_file_paths.apply(lambda p: Path(p).exists())]
            if args.debug:
                print(f"ERROR: some new file paths do not exist: {missing}")
            else:
                raise ValueError(f"Some new file paths do not exist: {missing}")
        print(" All good!")
        # save modified parquet
        if not args.debug:
            df["file_path"] = new_file_paths
            df.to_parquet(ds_files)
            print(f"Saved modifed parquet to {ds_files}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_files_list", nargs="+", type=str, required=True)
    parser.add_argument("--previous_dataset_prefix", type=str, required=True)
    parser.add_argument("--new_dataset_prefix", type=str, required=True)
    parser.add_argument("--substring_to_remove", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    parsed_args = parser.parse_args()
    main(parsed_args)
