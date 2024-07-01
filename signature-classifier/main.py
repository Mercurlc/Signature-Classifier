import gc
import json
import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_score, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SignatureVerification:
    """
    A class used to verify signatures using HOG features and KNN classifier.

    Attributes
    ----------
    dataset_path : str
        Path to the dataset containing real and forged signatures.
    storage_path : str
        Path to save models, parameters, and results.
    verbose : bool, optional
        If True, enables logging information (default is False).
    best_params : dict
        Stores the best hyperparameters found during optimization.
    signatures : dict
        Dictionary containing the paths to real and forged signatures.

    Methods
    -------
    log_info(message)
        Logs an info message if verbose is True.
    log_error(message)
        Logs an error message if verbose is True.
    _load_signatures()
        Loads signatures from the dataset path.
    _extract_hog_features(image_path, resize_shape, orientations, pixels_per_cell, cells_per_block)
        Extracts HOG features from an image.
    _prepare_data(resize_shape, orientations, pixels_per_cell, cells_per_block)
        Prepares the dataset by extracting HOG features.
    _objective(trial)
        Objective function for hyperparameter optimization using Optuna.
    optimize_hyperparameters(n_trials)
        Optimizes hyperparameters using Optuna.
    train_final_model(params_dict=None, save_plots=False)
        Trains the final model with given or optimized hyperparameters.
    evaluate_model(model, X_test, y_test, save_plots)
        Evaluates the trained model and saves classification report and confusion matrix.
    predict(image_path, top_n=5, return_forg_preds=True)
        Predicts the class of a given signature image.
    _forg_predict(features, class_name, params)
        Predicts the similarity score for forgery detection.
    _save_model(model)
        Saves the trained model.
    _load_model()
        Loads the trained model.
    _save_params(params)
        Saves the hyperparameters.
    _load_params()
        Loads the hyperparameters.
    """

    def __init__(self, dataset_path: str, storage_path: str, verbose: bool = False) -> None:
        """
        Initializes the SignatureVerification class with dataset path, storage path, and verbosity.

        Parameters:
        dataset_path (str): Path to the dataset containing real and forged signatures.
        storage_path (str): Path to save models, parameters, and results.
        verbose (bool): If True, enables logging information.
        """
        self.best_params = None
        self.dataset_path = dataset_path
        self.storage_path = storage_path
        self.verbose = verbose
        self.signatures = self._load_signatures()
        os.makedirs(storage_path, exist_ok=True)
        sns.set(style="whitegrid")

    def log_info(self, message: str):
        """
        Logs an info message if verbose is True.

        Parameters:
        message (str): The message to be logged.
        """
        if self.verbose:
            logging.info(message)

    def log_error(self, message: str):
        """
        Logs an error message if verbose is True.

        Parameters:
        message (str): The message to be logged.
        """
        if self.verbose:
            logging.error(message)

    def _load_signatures(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Loads signatures from the dataset path.

        Returns:
        Dict[str, Dict[str, List[str]]]: Dictionary containing the paths to real and forged signatures.
        """
        signatures = {}
        for folder in os.listdir(self.dataset_path):
            if '_' in folder:
                key = folder.split('_')[0]
                if key not in signatures:
                    signatures[key] = {'real': [], 'forg': []}
                signatures[key]['forg'].extend([os.path.join(self.dataset_path, folder, f) for f in
                                                os.listdir(os.path.join(self.dataset_path, folder))])
            else:
                key = folder
                if key not in signatures:
                    signatures[key] = {'real': [], 'forg': []}
                signatures[key]['real'].extend([os.path.join(self.dataset_path, folder, f) for f in
                                                os.listdir(os.path.join(self.dataset_path, folder))])
        self.log_info("Signatures loaded successfully.")
        return signatures

    def _extract_hog_features(self, image_path: str, resize_shape: Tuple[int, int], orientations: int,
                              pixels_per_cell: int, cells_per_block: int) -> np.ndarray:
        """
        Extracts HOG features from an image.

        Parameters:
        image_path (str): Path to the image.
        resize_shape (Tuple[int, int]): Size to resize the image.
        orientations (int): Number of orientation bins.
        pixels_per_cell (int): Size (in pixels) of a cell.
        cells_per_block (int): Number of cells in each block.

        Returns:
        np.ndarray: HOG features of the image.
        """
        try:
            image = imread(image_path, as_gray=True)
            image_resized = resize(image, resize_shape)
            features = hog(image_resized,
                           orientations=orientations,
                           pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                           cells_per_block=(cells_per_block, cells_per_block))
            return features
        except Exception as e:
            self.log_error(f"Error extracting HOG features from {image_path}: {e}")
            raise

    def _prepare_data(self, resize_shape: Tuple[int, int], orientations: int, pixels_per_cell: int,
                      cells_per_block: int) -> pd.DataFrame:
        """
        Prepares the dataset by extracting HOG features.

        Parameters:
        resize_shape (Tuple[int, int]): Size to resize the images.
        orientations (int): Number of orientation bins.
        pixels_per_cell (int): Size (in pixels) of a cell.
        cells_per_block (int): Number of cells in each block.

        Returns:
        pd.DataFrame: DataFrame containing HOG features and labels.
        """
        data = []
        for user, paths in self.signatures.items():
            for path in paths['real']:
                try:
                    features = self._extract_hog_features(path, resize_shape, orientations, pixels_per_cell,
                                                          cells_per_block)
                    data.append({
                        'Path': path,
                        'HOG': features,
                        'Real': 1,
                        'Class': user
                    })
                except Exception as e:
                    self.log_error(f"Error processing real signature {path}: {e}")
            for path in paths['forg']:
                try:
                    features = self._extract_hog_features(path, resize_shape, orientations, pixels_per_cell,
                                                          cells_per_block)
                    data.append({
                        'Path': path,
                        'HOG': features,
                        'Real': 0,
                        'Class': user
                    })
                except Exception as e:
                    self.log_error(f"Error processing forged signature {path}: {e}")
        self.log_info("Data prepared successfully.")
        return pd.DataFrame(data)

    def _objective(self, trial: optuna.trial.Trial) -> float:
        """
        Objective function for hyperparameter optimization using Optuna.

        Parameters:
        trial (optuna.trial.Trial): Optuna trial object.

        Returns:
        float: Mean F1 score from cross-validation.
        """
        resize_shape = (trial.suggest_categorical('resize_height', [64, 128, 256, 512]),
                        trial.suggest_categorical('resize_width', [64, 128, 256, 512]))
        orientations = trial.suggest_int('orientations', 4, 16)
        pixels_per_cell = trial.suggest_int('pixels_per_cell', 4, 16)
        cells_per_block = trial.suggest_int('cells_per_block', 1, 4)
        nns = trial.suggest_int('nns', 2, 10)

        df = self._prepare_data(resize_shape, orientations, pixels_per_cell, cells_per_block)
        X = np.array(df['HOG'].tolist())
        y = np.array(df['Class'].tolist())

        try:
            knn_model = KNeighborsClassifier(n_neighbors=nns, n_jobs=-1)
            skf = StratifiedKFold(n_splits=5)
            scores = cross_val_score(knn_model, X, y, cv=skf, scoring='f1_macro', n_jobs=-1)

            del df, X, y, knn_model
            gc.collect()
            return scores.mean()
        except Exception as e:
            self.log_error(f"Error during cross-validation: {e}")
            return float('nan')

    def optimize_hyperparameters(self, n_trials: int) -> None:
        """
        Optimizes hyperparameters using Optuna.

        Parameters:
        n_trials (int): Number of trials for the optimization.
        """
        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(self._objective, n_trials=n_trials)

            self.best_params = study.best_trial.params
            self._save_params(self.best_params)
            self.log_info("Hyperparameters optimized successfully.")
            print("Best trial:")
            print(f"  Value: {study.best_trial.value}")
            print("  Params: ")
            for key, value in study.best_trial.params.items():
                print(f"    {key}: {value}")
        except Exception as e:
            self.log_error(f"Error during hyperparameter optimization: {e}")
            raise

    def train_final_model(self, params_dict: Optional[Dict[str, Union[int, float]]] = None,
                          save_plots: bool = False) -> None:
        """
        Trains the final model with given or optimized hyperparameters.

        Parameters:
        params_dict (Optional[Dict[str, Union[int, float]]]): Dictionary containing custom hyperparameters.
        save_plots (bool): If True, saves the classification report and confusion matrix plots.
        """
        try:
            if params_dict:
                params = params_dict
                self._save_params(params)
                self.log_info("Custom params saved successfully.")
            else:
                params = self._load_params()

            resize_shape = params['resize_shape']
            orientations = params['orientations']
            pixels_per_cell = params['pixels_per_cell']
            cells_per_block = params['cells_per_block']
            nns = params['nns']

            df = self._prepare_data(resize_shape, orientations, pixels_per_cell, cells_per_block)
            X = np.array(df['HOG'].tolist())
            y = np.array(df['Class'].tolist())

            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
            for train_index, test_index in sss.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

            knn_model = KNeighborsClassifier(n_neighbors=nns, n_jobs=-1)
            knn_model.fit(X_train, y_train)

            self._save_model(knn_model)
            self.log_info("Final model trained and saved successfully.")

            self.evaluate_model(knn_model, X_test, y_test, save_plots)
        except FileNotFoundError as e:
            self.log_error(f"File not found: {e}")
            raise
        except Exception as e:
            self.log_error(f"Error during final model training: {e}")
            raise

    def evaluate_model(self, model: KNeighborsClassifier, X_test: np.ndarray, y_test: np.ndarray,
                       save_plots: bool) -> None:
        """
        Evaluates the trained model and saves classification report and confusion matrix.

        Parameters:
        model (KNeighborsClassifier): Trained KNN model.
        X_test (np.ndarray): Test set features.
        y_test (np.ndarray): Test set labels.
        save_plots (bool): If True, saves the classification report and confusion matrix plots.
        """
        try:
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)

            report_path = os.path.join(self.storage_path, 'classification_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
            self.log_info("Classification report saved successfully.")

            if save_plots:
                report_df = pd.DataFrame(report).transpose()
                report_df = report_df.drop(columns=['support'], errors='ignore')
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(report_df.iloc[:-1, :], annot=True, fmt=".2f", cmap="coolwarm", ax=ax, vmin=0, vmax=1)
                ax.set_title('Classification Report')
                ax.set_xlabel('Metrics')
                ax.set_ylabel('Classes')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.savefig(os.path.join(self.storage_path, 'classification_report_heatmap.png'))
                plt.close()
                self.log_info("Classification report heatmap saved successfully.")

                conf_matrix = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="coolwarm", ax=ax,
                            xticklabels=model.classes_, yticklabels=model.classes_)
                ax.set_xlabel("Predicted labels")
                ax.set_ylabel("True labels")
                ax.set_title('Confusion Matrix')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(self.storage_path, 'confusion_matrix_heatmap.png'))
                plt.close()
                self.log_info("Confusion matrix heatmap saved successfully.")
        except Exception as e:
            self.log_error(f"Error during model evaluation: {e}")
            raise

    def predict(self, image_path: str,
                top_n: int = 5,
                return_forg_preds: bool = True) -> List[Union[Tuple[str, float], Tuple[str, float, float]]]:
        """
        Predicts the class of a given signature image.

        Parameters:
        image_path (str): Path to the signature image.
        top_n (int): Number of top predictions to return.
        return_forg_preds (bool): If True, returns similarity score for forgery detection.

        Returns:
        List[Union[Tuple[str, float], Tuple[str, float, float]]]: List of top predictions with probabilities and optional forgery scores.
        """
        try:
            self.model = self._load_model()

            params = self._load_params()
            resize_shape = params['resize_shape']
            orientations = params['orientations']
            pixels_per_cell = params['pixels_per_cell']
            cells_per_block = params['cells_per_block']

            features = self._extract_hog_features(image_path, resize_shape, orientations, pixels_per_cell,
                                                  cells_per_block).reshape(1, -1)
            probs = self.model.predict_proba(features)[0]

            classes = self.model.classes_
            top_indices = np.argsort(probs)[-top_n:][::-1]
            top_classes = classes[top_indices]
            top_probs = probs[top_indices]

            predictions = [(cls, prob) for cls, prob in zip(top_classes, top_probs) if prob > 0]

            self.log_info(f"found {len(predictions)} with proba >0")

            if return_forg_preds:
                forg_probs = []
                for class_name, prob in predictions:
                    similarity = self._forg_predict(features, class_name, params)
                    forg_probs.append(similarity)
                predictions = [(cls, prob, forg_prob) for (cls, prob), forg_prob in zip(predictions, forg_probs)]

            self.log_info(f"Prediction for {image_path} completed successfully.")
            return predictions
        except FileNotFoundError as e:
            self.log_error(f"Model file not found: {e}")
            raise
        except Exception as e:
            self.log_error(f"Error during prediction: {e}")
            raise

    def _forg_predict(self, features: np.ndarray, class_name: str,
                      params: Optional[Dict[str, Union[int, Tuple[int, int]]]]) -> float:
        """
        Predicts the similarity score for forgery detection.

        Parameters:
        features (np.ndarray): HOG features of the signature image.
        class_name (str): The class (user) to check forgery against.
        params (Optional[Dict[str, Union[int, Tuple[int, int]]]]): Hyperparameters for HOG feature extraction.

        Returns:
        float: Similarity score for forgery detection.
        """
        try:
            resize_shape = params['resize_shape']
            orientations = params['orientations']
            pixels_per_cell = params['pixels_per_cell']
            cells_per_block = params['cells_per_block']

            real_paths = self.signatures[class_name]['real']
            forg_paths = self.signatures[class_name]['forg']

            real_features = np.array([
                self._extract_hog_features(path, resize_shape, orientations, pixels_per_cell, cells_per_block)
                for path in real_paths
            ])
            forg_features = np.array([
                self._extract_hog_features(path, resize_shape, orientations, pixels_per_cell, cells_per_block)
                for path in forg_paths
            ])

            real_sim = cosine_similarity(features, real_features).mean()
            forg_sim = cosine_similarity(features, forg_features).mean()

            similarity = 1 / (1 + np.exp(-(real_sim - forg_sim)))
            return similarity
        except Exception as e:
            self.log_error(f"Error during forgery prediction for class {class_name}: {e}")
            raise

    def _save_model(self, model: KNeighborsClassifier) -> None:
        """
        Saves the trained model.

        Parameters:
        model (KNeighborsClassifier): Trained KNN model.
        """
        try:
            with open(os.path.join(self.storage_path, 'knn_model.pkl'), 'wb') as f:
                pickle.dump(model, f)
            self.log_info("Model saved successfully.")
        except Exception as e:
            self.log_error(f"Error saving model: {e}")
            raise

    def _load_model(self) -> KNeighborsClassifier:
        """
        Loads the trained model.

        Returns:
        KNeighborsClassifier: Trained KNN model.
        """
        try:
            with open(os.path.join(self.storage_path, 'knn_model.pkl'), 'rb') as f:
                model = pickle.load(f)
            self.log_info("Model loaded successfully.")
            return model
        except FileNotFoundError as e:
            self.log_error(f"Model file not found: {e}")
            raise
        except Exception as e:
            self.log_error(f"Error loading model: {e}")
            raise

    def _save_params(self, params: Dict[str, Union[int, float]]) -> None:
        """
        Saves the hyperparameters.

        Parameters:
        params (Dict[str, Union[int, float]]): Dictionary containing hyperparameters.
        """
        try:
            with open(os.path.join(self.storage_path, 'params.pkl'), 'wb') as f:
                pickle.dump(params, f)
            self.log_info("Parameters saved successfully.")
        except Exception as e:
            self.log_error(f"Error saving parameters: {e}")
            raise

    def _load_params(self) -> Dict[str, Union[int, float]]:
        """
        Loads the hyperparameters.

        Returns:
        Dict[str, Union[int, float]]: Dictionary containing hyperparameters.
        """
        try:
            with open(os.path.join(self.storage_path, 'params.pkl'), 'rb') as f:
                params = pickle.load(f)
            self.log_info("Parameters loaded successfully.")
            return params
        except FileNotFoundError as e:
            self.log_error(f"Parameters file not found: {e}")
            raise
        except Exception as e:
            self.log_error(f"Error loading parameters: {e}")
            raise
