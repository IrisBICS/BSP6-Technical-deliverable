from model2 import Model2

SAVEPATH = "../SavedModels/Model2/"
DATAPATH = "../FER2013/fer2013_landmarks.csv"
NAME = "model2"

model = Model2(SAVEPATH, DATAPATH, name=NAME)
model.train()
#model.loadModel()
model.evaluateOnValid()
model.evaluateOnTest()