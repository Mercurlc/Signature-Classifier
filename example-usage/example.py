import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../signature-classifier')))

from signature_classifier import SignatureVerification

#First case, using custom params(very faster):

# Step 1: initializing a class
verification = SignatureVerification('path_to_dataset', 'path_to_storage', verbose=True)

# Step 2: Train final model with custom parameters
custom_params = {
    'resize_shape': (128, 256),
    'nns': 3,
    'orientations': 12,
    'cells_per_block': 3,
    'pixels_per_cell': 16,
}
verification.train_final_model(params_dict=custom_params, save_plots=True)

# Step 3: Predict using the trained model
predictions = verification.predict('path_to_test.png')
print(predictions)

#After training, the parameters and models are saved, and you don't have to train the model again


# #Second case, using optuna for training params:
#
# # Step 1: Optimize hyperparameters and train final model
# erification = SignatureVerification('path_to_dataset', 'path_to_storage', verbose=True)
# verification.optimize_hyperparameters(n_trials=50)
# verification.train_final_model()
#
# # Step 2: Predict using the trained mode
# predictions = verification.predict('path_to_test.png')
# print(predictions)
#
# #After training, the parameters and models are saved, and you don't have to train the model again
