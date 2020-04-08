# import necessary packages
from keras.models import load_model
import coremltools
import argparse
import pickle
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model")
ap.add_argument("-o", "--output", required=True,
	help="name of output the coreml model")
args = vars(ap.parse_args())
 
# load the trained convolutional neural network
print("[INFO] loading model...")
model = load_model(args["model"])

# convert the model to coreml format
# input_names="image_array",
print("[INFO] converting model")

coreml_model = coremltools.converters.keras.convert(model,
	input_names="image",
	image_input_names="image",
	image_scale=1/255.0,
	is_bgr=True,
	output_names="predict_val"
)

# save the model to disk
output = args["output"] + ".mlmodel"
print("[INFO] saving model as {}".format(output))
coreml_model.save(output)