from NN2 import NN2

SAVEPATH = "../SavedNNs/NN2/"
DATAPATH = "../FER2013/fer2013_landmarks.csv"
NAME = "NN2"

model = NN2(SAVEPATH, DATAPATH, name=NAME)
#model.train()
model.loadModel()
#model.evaluateOnValid()
#model.evaluateOnTest()
#model.generatePlots()
model.generateConfusionMatrices()