from NN1 import NN1

SAVEPATH = "../SavedNNs/NN1/"
DATAPATH = "../FER2013/icml_face_data.csv"
WEIGHTSPATH = "../FER2013/fer2013_weights.csv"
NAME = "NN1"

model = NN1(SAVEPATH, DATAPATH, WEIGHTSPATH, name=NAME)
#model.train()
model.loadModel()
#model.evaluateOnValid()
#model.evaluateOnTest()
#model.generatePlots()
model.generateConfusionMatrices()
model.generateConfusionMatrices(percentage=False)
model.generateConfusionMatrices(normalize=True)