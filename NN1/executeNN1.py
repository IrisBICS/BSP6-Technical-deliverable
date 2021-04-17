from NN1 import NN1

SAVEPATH = "../SavedNNs/NN1/"
DATAPATH = "../FER2013/icml_face_data.csv"
NAME = "NN1"

model = NN1(SAVEPATH, DATAPATH, name=NAME)
#model.train()
model.loadModel()
#model.evaluateOnValid()
#model.evaluateOnTest()
#model.generatePlots()
model.generateConfusionMatrices()