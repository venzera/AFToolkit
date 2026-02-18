import os

# When installed via pip (e.g. in Colab), __file__ points into site-packages,
# so weights are stored relative to the current working directory instead.
MODEL_WEIGHTS_DIR = os.path.join(os.getcwd(), 'weights')
OF_WEIGHTS = "params_model_2_ptm.npz"
SOURCE_OF_WEIGHTS_URL = "https://storage.googleapis.com/alphafold/alphafold_params_colab_2022-12-06.tar"