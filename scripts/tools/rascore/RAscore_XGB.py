import os
import numpy as np
import pickle

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.DataStructs import cDataStructs


class RAScorerXGB:
    """
    Prediction of machine learned retrosynthetic accessibility score
    The RAScore is calculated based on the predictions made on 200,000 compounds sampled from ChEMBL.
    The compounds were subjected to retrosynthetic analysis using a CASP tool (AiZynthfinder) and output used as labels to train a binary classifier.

    If the compounds are ChEMBL like use the RAscore models.
    Else if the compounds are more exotic, are small fragments, or closely resemble GDB. GBDscore may give a better result.

    This class facilitates predictions from the resulting model.
    """

    def __init__(self, model_path=None):
        """
        Loads the model.

        :param model_path: path to the XGBoost model (.pkl or .json) file.
                           If a .pkl path is given and a .json sibling exists,
                           the .json is preferred (avoids xgboost version conflicts).
        :type model_path: str
        """
        import xgboost as xgb

        HERE = os.path.abspath(os.path.dirname(__file__))
        MODEL = os.path.join(HERE, "models/XGB_chembl_ecfp_counts/model.pkl")
        path = MODEL if model_path is None else model_path

        # Prefer JSON native format over legacy pickle if available
        json_path = os.path.splitext(path)[0] + ".json"
        if os.path.exists(json_path):
            booster = xgb.Booster()
            booster.load_model(json_path)
            self.xgb_model = booster
            self._use_booster = True
        else:
            self.xgb_model = pickle.load(open(path, "rb"))
            self._use_booster = False

    def ecfp(self, smiles):
        """
        Converts SMILES into a counted ECFP6 vector with features.

        :param smiles: SMILES representation of the moelcule of interest
        :type smiles: str
        :return: ECFP6 counted vector with features
        :rtype: np.array
        """
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=False)
        size = 2048
        arr = np.zeros((size,), np.int32)
        for idx, v in fp.GetNonzeroElements().items():
            nidx = idx % size
            arr[nidx] += int(v)
        return arr

    def predict(self, smiles):
        """
        Predicts score from SMILES.

        :param smiles: SMILES representation of the moelcule of interest
        :type smiles: str
        :return: score
        :rtype: float
        """
        import xgboost as xgb

        arr = self.ecfp(smiles)
        if self._use_booster:
            dmat = xgb.DMatrix(arr.reshape(1, -1))
            return float(self.xgb_model.predict(dmat)[0])
        else:
            proba = self.xgb_model.predict_proba(arr.reshape(1, -1))
            return proba[0][1]

