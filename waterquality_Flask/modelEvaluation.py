import coremltools
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path of image")
args = vars(ap.parse_args())

# Load the model
model = coremltools.models.MLModel('waterpredict.mlmodel')

# Make predictions
predictions = model.predict(args["image"])

print(predictions)