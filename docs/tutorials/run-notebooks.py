# ! python

import argparse
import glob
import time

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError

start = time.time()
# Parse args
parser = argparse.ArgumentParser(
    description="Runs a set of Jupyter \
                                              notebooks."
)
file_text = """ Notebook file(s) to be run, e.g. '*.ipynb' (default),
'my_nb1.ipynb', 'my_nb1.ipynb my_nb2.ipynb', 'my_dir/*.ipynb'
"""
parser.add_argument("file_list", metavar="F", type=str, nargs="*", help=file_text)
parser.add_argument(
    "-t",
    "--timeout",
    help="Length of time (in secs) a cell \
    can run before raising TimeoutError (default 600).",
    default=600,
    required=False,
)
parser.add_argument(
    "-p",
    "--run-path",
    help="The path the notebook will be \
    run from (default pwd).",
    default=".",
    required=False,
)
parser.add_argument(
    "-o", "--overwrite", help="Overwrite notebooks", action="store_true"
)
args = parser.parse_args()
# print('Args:', args)
if not args.file_list:  # Default file_list
    args.file_list = glob.glob("*.ipynb")

# Check list of notebooks
notebooks = []
print("Notebooks to run:")
for file_name in args.file_list:
    # Find notebooks but not notebooks previously output from this script
    if file_name.endswith(".ipynb") and not file_name.endswith("_out.ipynb"):
        print(file_name[:-6])
        notebooks.append(file_name[:-6])  # Want the filename without '.ipynb'

# Execute notebooks and output
num_notebooks = len(notebooks)
print("\n*****\n")
for i, note_name in enumerate(notebooks):
    if args.overwrite:
        n_out = note_name
    else:
        n_out = note_name + "_out"
    with open(note_name + ".ipynb") as note_file:
        nb = nbformat.read(note_file, as_version=4)
        ep = ExecutePreprocessor(timeout=int(args.timeout), kernel_name="python3")
        try:
            print("Running", note_name, ":", i, "/", num_notebooks)
            out = ep.preprocess(nb, {"metadata": {"path": args.run_path}})
        except CellExecutionError:
            out = None
            msg = 'Error executing the notebook "%s".\n' % note_name
            msg += 'See notebook "%s" for the traceback.' % n_out
            print(msg)
        except TimeoutError:
            msg = 'Timeout executing the notebook "%s".\n' % note_name
            print(msg)
        finally:
            # Write output file
            with open(n_out + ".ipynb", mode="w") as out_note_file:
                nbformat.write(nb, out_note_file)

stop = time.time()
print(f"Running notebooks took {stop - start:.2f} s")
