#!/usr/bin/env python3
# Script for making predictions using both the X module and the Y module

# Author: Jie Li
# Date created: Sep 20, 2019

import sys
if sys.version_info.major < 3 or sys.version_info.major == 3 and sys.version_info.minor < 5:
    raise ValueError("Python >= 3.5 required")

# Version checks for critical dependencies
import sklearn
import Bio
_sklearn_version = tuple(map(int, sklearn.__version__.split('.')[:2]))
_bio_version = tuple(map(int, Bio.__version__.split('.')[:2]))
if _sklearn_version != (0, 22):
    import warnings
    warnings.warn(
        f"scikit-learn {sklearn.__version__} detected. Models were trained with 0.22. "
        "Predictions may fail or be inaccurate. Install with: pip install scikit-learn==0.22",
        UserWarning
    )
if _bio_version != (1, 74):
    import warnings
    warnings.warn(
        f"Biopython {Bio.__version__} detected. This code requires 1.74 for BLOSUM62 compatibility. "
        "Install with: pip install biopython==1.74",
        UserWarning
    )

from spartap_features import PDB_SPARTAp_DataReader
from data_prep_functions import *
import ucbshifty
import joblib
import toolbox
import os
import pandas as pd
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed


# Suppress Setting With Copy warnings
pd.options.mode.chained_assignment = None

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
ML_MODEL_PATH = SCRIPT_PATH + "/models/"

# Model cache to avoid repeated loading (significant speedup in batch mode)
_MODEL_CACHE = {}

# Check for ONNX Runtime GPU support
_ONNX_AVAILABLE = False
_ONNX_GPU = False
try:
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
    _ONNX_GPU = 'CUDAExecutionProvider' in ort.get_available_providers()
    if _ONNX_GPU:
        print("ONNX Runtime GPU acceleration available")
except ImportError:
    pass


def _load_model(model_path):
    """Load a model with caching, preferring ONNX GPU if available."""
    if model_path not in _MODEL_CACHE:
        onnx_path = model_path.replace('.sav', '.onnx')

        # Try ONNX first (faster, especially on GPU)
        if _ONNX_AVAILABLE and os.path.exists(onnx_path):
            try:
                if _ONNX_GPU:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                else:
                    providers = ['CPUExecutionProvider']
                session = ort.InferenceSession(onnx_path, providers=providers)
                _MODEL_CACHE[model_path] = ('onnx', session)
                return _MODEL_CACHE[model_path]
            except Exception as e:
                print(f"Warning: Failed to load ONNX model {onnx_path}: {e}")

        # Fall back to sklearn
        _MODEL_CACHE[model_path] = ('sklearn', joblib.load(model_path))

    return _MODEL_CACHE[model_path]


def _predict(model_tuple, features):
    """Run prediction with either ONNX or sklearn model."""
    model_type, model = model_tuple

    if model_type == 'onnx':
        input_name = model.get_inputs()[0].name
        return model.run(None, {input_name: features.astype(np.float32)})[0].ravel()
    else:
        return model.predict(features).ravel()


def preload_models():
    """Preload all models into cache. Call before batch processing for best performance."""
    if not os.path.isdir(ML_MODEL_PATH):
        raise ValueError("models not installed in {}".format(ML_MODEL_PATH))

    onnx_count = 0
    sklearn_count = 0

    for atom in toolbox.ATOMS:
        for level in ["R0", "R1", "R2"]:
            model_path = ML_MODEL_PATH + "%s_%s.sav" % (atom, level)
            if os.path.exists(model_path):
                model_tuple = _load_model(model_path)
                if model_tuple[0] == 'onnx':
                    onnx_count += 1
                else:
                    sklearn_count += 1

    if onnx_count > 0:
        gpu_str = " (GPU)" if _ONNX_GPU else " (CPU)"
        print(f"Loaded {onnx_count} ONNX{gpu_str} + {sklearn_count} sklearn models")


def _batch_worker_init(models_path):
    """Initialize worker process with models (called once per worker)."""
    global ML_MODEL_PATH, _MODEL_CACHE
    ML_MODEL_PATH = models_path
    _MODEL_CACHE = {}
    preload_models()


def _batch_worker(args):
    """Worker function for parallel batch processing."""
    pdb_file, pH, use_tp, use_ml, test_mode, save_prefix = args
    try:
        preds = calc_sing_pdb(pdb_file, pH, TP=use_tp, ML=use_ml, test=test_mode)
        output_file = save_prefix + os.path.basename(pdb_file).replace(".pdb", ".csv")
        preds.to_csv(output_file, index=None)
        return (pdb_file, True, None)
    except Exception as e:
        return (pdb_file, False, str(e))


def build_input(pdb_file_name, pH=5, rcfeats=True, hse=True, hbrad=[5.0] * 3):
    '''
    Function for building dataframe for the specified pdb file.
    Returns a pandas dataframe for a specific single-chain pdb file.

    args:
        pdb_file_name = The path to the PDB file for prediction (str)
        pH = pH value to be considered
        rcfeats = Include a column for random coil chemical shifts (Bool)
        hse = Include a feature column for the half-sphere exposure (Bool)
        hbrad = Max length of hydrogen bonds for HA,HN and O (list of float)
    '''
    # Build feature reader and read all features
    feature_reader = PDB_SPARTAp_DataReader()
    pdb_data = feature_reader.df_from_file_3res(pdb_file_name,rcshifts=rcfeats, hse=hse, first_chain_only=False, sequence_columns=0,hbrad=hbrad)
    pdb_data["pH"]=pH
    return pdb_data

def data_preprocessing(data):
    '''
    Function for executing all the preprocessing steps based on the original extracted features, including fixing HA2/HA3 ring current ambiguity, adding hydrophobicity, powering features, drop unnecessary columns, etc.
    '''
    data = data.copy()
    data = data[sorted(data.columns)]
    data = ha23ambigfix(data, mode=0)
    Add_res_spec_feats(data, include_onehot=False)
    data = feat_pwr(data, hbondd_cols + cos_cols, [2])
    data = feat_pwr(data, hbondd_cols, [-1,-2,-3])
    dropped_cols = dssp_pp_cols + dssp_energy_cols + ['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM',"RES", 'CHAIN', 'RESNAME_ip1', 'RESNAME_im1', 'BMRB_RES_NUM', 'CG', 'RCI_S2', 'MATCHED_BMRB',"identifier"]+ rcoil_cols
    data = data.drop(set(dropped_cols) & set(data.columns), axis=1)
    return data

def prepare_data_for_atom(data,atom):
    '''
    Function to generate features data for a given atom type: meaning that the irrelevant ring current values are removed from features

    args:
        data - the dataset that contains all the features (pandas.DataFrame)
        atom - the atom to keep ring currents

    returns:
        pandas.DataFrame containing the cleaned feature set
    '''
    dat = data.copy()
    ring_col = atom + '_RC'
    rem1 = ring_cols.copy()
    rem1.remove(ring_col)
    dat = dat.drop(rem1, axis=1)
    dat[ring_col] = dat[ring_col].fillna(value=0)
    return dat

def calc_sing_pdb(pdb_file_name,pH=5,TP=True,TP_pred=None,ML=True,test=False):
    '''
    Function for calculating chemical shifts for a single PDB file using X module / Y module / both

    args:
        pdb_file_name = The path to the PDB file for prediction (str)
        pH = pH value to be considered
        TP = Whether or not use TP module (Bool)
        TP_pred = predicted shifts dataframe from Y module. If None, generate this prediction within this function (pandas.DataFrame / None)
        ML = Whether or not use ML module (Bool)
        test = Whether or not use test mode (Exclude mode for SHIFTY++, Bool)
    '''
    if not os.path.isdir(ML_MODEL_PATH):
        raise ValueError("models not installed in {}".format(ML_MODEL_PATH))
    if pH < 2 or pH > 12:
        print("Warning! Predictions for proteins in extreme pH conditions are likely to be erroneous. Take prediction results at your own risk!")
    preds = pd.DataFrame()
    if TP:
        if TP_pred is None:
            print("Calculating UCBShift-Y predictions ...")
            # generate hash string from pdb file name
            hashed_file_name = str(hash(pdb_file_name) % ((sys.maxsize + 1) * 2)) + '/'
            TP_pred = ucbshifty.main(pdb_file_name, 1, exclude=test, custom_working_dir=hashed_file_name)
        if not ML:
            # Prepare data when only doing TP prediction
            preds = TP_pred[["RESNUM","RESNAME"]]
            for atom in toolbox.ATOMS:
                if atom+"_RC" in TP_pred.columns:
                    rc = TP_pred[atom+"_RC"]
                else:
                    rc = 0
                preds[atom+"_Y"] = TP_pred[atom] + rc
    if ML:
        print("Generating features ...")
        feats = build_input(pdb_file_name, pH)
        feats.rename(index=str, columns=sparta_rename_map, inplace=True) # Rename columns so that random coil columns can be correctly recognized
        resnames = feats["RESNAME"]
        resnums = feats["RES_NUM"]
        rcoils = feats[rcoil_cols]
        feats = data_preprocessing(feats)

        result = {"RESNUM":resnums, "RESNAME":resnames}
        for atom in toolbox.ATOMS:
            print("Calculating UCBShift-X predictions for %s ..." % atom)
            # Predictions for each atom
            atom_feats = prepare_data_for_atom(feats, atom)
            r0 = _load_model(ML_MODEL_PATH + "%s_R0.sav" % atom)
            r0_pred = _predict(r0, atom_feats.values)

            feats_r1 = atom_feats.copy()
            feats_r1["R0_PRED"] = r0_pred
            r1 = _load_model(ML_MODEL_PATH + "%s_R1.sav" % atom)
            r1_pred = _predict(r1, feats_r1.values)
            # Write ML predictions
            result[atom+"_X"] = r1_pred + rcoils["RCOIL_"+atom]

            if TP:
                print("Calculating UCBShift predictions for %s ..." % atom)
                feats_r2 = atom_feats.copy()
                feats_r2["RESNAME"] = resnames
                feats_r2["RESNUM"] = resnums
                tp_atom = TP_pred[["RESNAME", "RESNUM", atom, atom+"_BEST_REF_SCORE", atom+"_BEST_REF_COV", atom+"_BEST_REF_MATCH"]]
                feats_r2 = pd.merge(feats_r2, tp_atom, on="RESNUM", suffixes=("","_TP"), how="left")
                # Write TP predictions
                result[atom+"_Y"] = feats_r2[atom].values
                result[atom+"_BEST_REF_SCORE"] = feats_r2[atom+"_BEST_REF_SCORE"].values
                result[atom+"_BEST_REF_COV"] = feats_r2[atom+"_BEST_REF_COV"].values
                result[atom+"_BEST_REF_MATCH"] = feats_r2[atom+"_BEST_REF_MATCH"].values
                valid = (feats_r2.RESNAME == feats_r2.RESNAME_TP) & (feats_r2[atom].notnull())
                # Subtract random coils to make secondary TP shifts
                feats_r2[atom] -= rcoils["RCOIL_"+atom].values
                feats_r2["R0_PRED"] = r0_pred
                valid_feats_r2 = feats_r2.drop(["RESNAME","RESNUM","RESNAME_TP"], axis=1)[valid]
                r2_pred = r1_pred.copy()
                if len(valid_feats_r2):
                    r2 = _load_model(ML_MODEL_PATH + "%s_R2.sav" % atom)
                    r2_pred_valid = _predict(r2, valid_feats_r2.values)
                    r2_pred[valid] = r2_pred_valid
                # Write final predictions
                result[atom+"_UCBShift"] = r2_pred + rcoils["RCOIL_"+atom]
        preds = pd.DataFrame(result)
    return preds

        

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='This program is an NMR chemical shift predictor for protein chemical shifts (including H, Hα, C\', Cα, Cβ and N) in aqueous solution. It has two sub-modules: one is the machine learning (X) module. It uses ensemble tree-based methods to predict chemical shifts from the features extracted from the PDB files. The second sub-module is the transfer prediction (Y) module that predicts chemical shifts by "transfering" shifts from similar proteins in the database to the query protein through structure and sequence alignments. Finally, the two parts are combined to give the UCBShift predictions.')
    args.add_argument("input", help="The query PDB file or list of PDB files for which the shifts are calculated")
    args.add_argument("--batch", "-b", action="store_true", help="If toggled, input accepts a text file specifying all the PDB files need to be calculated (Each line is a PDB file name. If pH values are specified, followed with a space)")
    args.add_argument("--output", "-o", help="Filename of generated output file. A file [shifts.csv] is generated by default. If in batch mode, you should specify the path for storing all the output files. Each output file has the same name as the input PDB file name.", default="shifts.csv")
    args.add_argument("--worker", "-w", type=int, help="Number of CPU cores to use for parallel prediction in batch mode.", default=4)
    args.add_argument("--shifty_only", "-y", "-Y", action="store_true", help="Only use UCBShift-Y (transfer prediction) module. Equivalent to executing UCBShift-Y directly with default settings")
    args.add_argument("--shiftx_only", "-x", "-X", action="store_true", help="Only use UCBShift-X (machine learning) module. No alignment results will be utilized or calculated")
    args.add_argument("--pH", "-pH", "-ph", type=float, help="pH value to be considered. Default is 5", default=5)
    args.add_argument("--test", "-t", action="store_true", help="If toggled, use test mode for UCBShift-Y prediction")
    args.add_argument("--models", help="Alternate location for models directory")
    args = args.parse_args()
    if args.models:
        if not os.path.isdir(args.models):
            raise ValueError("Directory {} specified by models does not exists".format(args.models))
        ML_MODEL_PATH = args.models
 
    # Preload models for faster processing (except parallel batch mode - workers load their own)
    if not args.shifty_only and not (args.batch and args.worker > 1):
        print("Loading models...")
        preload_models()

    if not args.batch:
        preds = calc_sing_pdb(args.input, args.pH, TP=not args.shiftx_only, ML=not args.shifty_only, test=args.test)
        preds.to_csv(args.output, index=None)
    else:
        inputs = []
        with open(args.input) as f:
            for line in f:
                line_content = line.split()
                if len(line_content) == 1:
                    # No pH values explicitly specified. Use the global pH values
                    line_content.append(args.pH)
                else:
                    line_content[-1] = float(line_content[-1])
                inputs.append(line_content)
        # Decide saving folder
        if args.output == "shifts.csv":
            # No specific output path specified. Store all files in the current folder
            SAVE_PREFIX = ""
        else:
            SAVE_PREFIX = args.output
            if SAVE_PREFIX[-1] != "/":
                SAVE_PREFIX = SAVE_PREFIX + "/"

        # Process in batch with multiprocessing
        num_workers = min(args.worker, len(inputs))
        use_tp = not args.shiftx_only
        use_ml = not args.shifty_only

        if num_workers > 1 and len(inputs) > 1:
            # Parallel processing with worker pool
            print(f"Processing {len(inputs)} files with {num_workers} workers...")
            work_items = [
                (item[0], item[1], use_tp, use_ml, args.test, SAVE_PREFIX)
                for item in inputs
            ]

            # Use spawn context to avoid fork issues with sklearn models
            ctx = multiprocessing.get_context('spawn')
            with ctx.Pool(num_workers, initializer=_batch_worker_init, initargs=(ML_MODEL_PATH,)) as pool:
                results = pool.map(_batch_worker, work_items)

            # Report results
            success = sum(1 for _, ok, _ in results if ok)
            failed = [(f, e) for f, ok, e in results if not ok]
            print(f"Completed: {success}/{len(inputs)} files")
            if failed:
                print("Failed files:")
                for f, e in failed:
                    print(f"  {f}: {e}")
        else:
            # Sequential processing (single file or single worker)
            for idx, item in enumerate(inputs):
                preds = calc_sing_pdb(item[0], item[1], TP=use_tp, ML=use_ml, test=args.test)
                preds.to_csv(SAVE_PREFIX + os.path.basename(item[0]).replace(".pdb", ".csv"), index=None)
                print("Finished prediction for %s (%d/%d)" % (item[0], idx + 1, len(inputs)))    
    
    print("Complete!")
   
