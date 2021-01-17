# In[1]:


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
pd.options.mode.chained_assignment = None  # default='warn'
print("Imported")

cred = credentials.Certificate("halcyon-firebase-admins.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://halcyon-c8ffd-default-rtdb.firebaseio.com/Client_Data/Steps'
})
print('initialized')

steps = db.reference('Client_Data/Steps').get()
sleep = db.reference('Client_Data/Sleep (seconds)').get()
cal = db.reference('Client_Data/Active_Calories (Cal)').get()
heart = db.reference('Client_Data/Heart_Rate_(bpm)').get()

temp = [*steps.values()]
steps = []
timeSteps = []
for i in temp:
    steps.append(i['Steps'])
    timeSteps.append(i['TimeStamp'].split(" ")[0])
print(steps)
print(timeSteps)

temp = [*sleep.values()]
sleep = []
timeSleep = []
for j in temp:
    sleep.append(j['Sleep (seconds)'])
    timeSleep.append(j['TimeStamp'].split(" ")[0])
print(sleep)
print(timeSleep)

temp = [*cal.values()]
cal = []
timeCal = []
for k in temp:
    cal.append(k['Active_Calories (Cal)'])
    timeCal.append(k['TimeStamp'].split(" ")[0])
print(cal)
print(timeCal)

temp = [*heart.values()]
heart = []
timeHeart = []
for l in temp:
    heart.append(l['Heart_Rate_(bpm)'])
    timeHeart.append(k['TimeStamp'].split(" ")[0])
print(heart)
print(timeHeart)

values = [steps, sleep, cal, heart]
timeStamps = [timeSteps, timeSleep, timeCal, timeHeart]
valueLabels = ['Steps', 'Sleep', 'Active Calories', 'Heart Rate']

manInput = db.reference('Client_Data/Manual_Input').get()
print(manInput)

temp = [*manInput.values()]
inputLabels = ['Energy', 'Creativity', 'Motivation', 'Happiness']
manInput = [[], [], [], []]
timeInput = []
for i in range(len(temp)):
    for key, value in temp[i].items():
        if key == 'TimeStamp':
            timeInput.append(value.split(" ")[0])
        else:
            manInput[inputLabels.index(key)].append(value)
print(manInput)
print(inputLabels)
print(timeInput)

print(timeInput)
print(timeStamps[0])
print(timeStamps[1])
print(timeStamps[2])
print(timeStamps[3])

# values | timeStamps | valueLabels | manInput | inputLabels | timeInput
# steps | sleep | cal | heart

validTimes = []
for t in timeInput:
    if t in timeStamps[valueLabels.index('Steps')] and t in timeStamps[valueLabels.index('Sleep')] and t in timeStamps[
        valueLabels.index('Active Calories')]:
        validTimes.append(t)
validTimes = list(dict.fromkeys(validTimes))
print(validTimes)

def count(valid, time, userIn, average):
    lst = []
    for v in range(len(valid)):
        steppin = 0
        count = 0
        for day in range(len(time)):
            if time[day] == valid[v]:
                steppin += userIn[day]
                count += 1
        if average == 0:
            lst.append(steppin)
        else:
            try:
                lst.append(steppin // count)
            except ZeroDivisionError:
                lst.append(0)
    return lst

values[valueLabels.index('Steps')] = count(validTimes, timeSteps, steps, 0)
values[valueLabels.index('Sleep')] = count(validTimes, timeSleep, sleep, 0)
values[valueLabels.index('Active Calories')] = count(validTimes, timeCal, cal, 0)
values[valueLabels.index('Heart Rate')] = count(validTimes, timeHeart, heart, 1)
for i in range(len(manInput)):
    manInput[i] = count(validTimes, timeInput, manInput[i], 1)
print(values)
print(manInput)

dfh = pd.DataFrame(values[0], columns=[valueLabels[0]])
dfe = pd.DataFrame(values[0], columns=[valueLabels[0]])
dfc = pd.DataFrame(values[0], columns=[valueLabels[0]])
dfm = pd.DataFrame(values[0], columns=[valueLabels[0]])

for i in range(1, len(values)):
    dfh[valueLabels[i]] = values[i]
    dfe[valueLabels[i]] = values[i]
    dfc[valueLabels[i]] = values[i]
    dfm[valueLabels[i]] = values[i]
dfe['Energy'] = manInput[inputLabels.index('Energy')]
dfc['Creativity'] = manInput[inputLabels.index('Creativity')]
dfm['Motivation'] = manInput[inputLabels.index('Motivation')]
dfh['Happiness'] = manInput[inputLabels.index('Happiness')]

def trainModel(df, model, label):
    X = df.drop(label, axis=1)
    y = df[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.12, shuffle=False)
    model.fit(X_train, y_train)
    mae_train = mean_absolute_error(y_train, model.predict(X_train))
    print(label)
    print("Training Set Mean Absolute Error: %2f" % mae_train)
    mae_test = mean_absolute_error(y_test, model.predict(X_test))
    print("Test Set Mean Absolute Error: %2f" % mae_test)


happiness_model = ensemble.GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=7,
    min_samples_split=4,
    min_samples_leaf=6,
    max_features=0.8,
    loss='huber'
)

motivation_model = ensemble.GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=7,
    min_samples_split=4,
    min_samples_leaf=6,
    max_features=0.8,
    loss='huber'
)

energy_model = ensemble.GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=7,
    min_samples_split=4,
    min_samples_leaf=6,
    max_features=0.8,
    loss='huber'
)

creativity_model = ensemble.GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=7,
    min_samples_split=4,
    min_samples_leaf=6,
    max_features=0.8,
    loss='huber'
)

trainModel(dfh, happiness_model, 'Happiness')
trainModel(dfm, motivation_model, 'Motivation')
trainModel(dfe, energy_model, 'Energy')
trainModel(dfc, creativity_model, 'Creativity')

# Pass it a row (not a whole dataframe)
def findFactors(model, df):
    scale = 1.1
    predictions = []
    for column in df:
        print(column)
        df[column] *= scale
        inX = df
        print(model.predict(inX))
        predictions.append(model.predict(inX)[0])
        df[column] /= scale
    return predictions

def mostRecent(dates):
    highest = [0, 0, 0]
    for i in range(len(dates)):
        current = dates[i].split("/")
        if int(current[0]) > int(highest[0]):
            highest = current
        if int(current[0]) == int(highest[0]):
            if int(current[1]) > int(highest[1]):
                highest = current
            if int(current[1]) == int(current[1]):
                if int(current[2]) > int(highest[2]):
                    highest = current

    return "%s/%s/%s" % (highest[0], highest[1], highest[2])

def sortPrediction(prediction):
    categories = ['Steps', 'Sleep', 'Active Calories', 'Heart Rate']
    sortPred = []
    sortCat = []
    m = 0
    for i in range(len(prediction)):
        m = max(prediction)
        sortCat.append(categories[prediction.index(m)])
        sortPred.append(m)
        categories.pop(prediction.index(m))
        prediction.pop(prediction.index(m))
    return (sortPred, sortCat)


mostRecentDate = mostRecent(validTimes)
dayIndex = validTimes.index(mostRecentDate)

df = dfh.iloc[dayIndex:dayIndex + 1].drop('Happiness', axis=1)
hPredict = findFactors(happiness_model, df)
mPredict = findFactors(motivation_model, df)
ePredict = findFactors(energy_model, df)
cPredict = findFactors(creativity_model, df)

hPredict = sortPrediction(hPredict)
ePredict = sortPrediction(ePredict)
mPredict = sortPrediction(mPredict)
cPredict = sortPrediction(cPredict)

ref = db.reference()
update_posts_ref_pos1 = ref.child('Analized_Data').child('Happy')
update_posts_ref_pos1.set({
    'position1' : hPredict[1][0], 'pos1Ammount' : hPredict[0][0],
    'position2' : hPredict[1][1], 'pos2Ammount' : hPredict[0][1],
    'position3' : hPredict[1][2], 'pos3Ammount' : hPredict[0][2],
    'position4' : hPredict[1][3], 'pos4Ammount' : hPredict[0][3],
})
update_posts_ref_pos1 = ref.child('Analized_Data').child('Energy')
update_posts_ref_pos1.set({
    'position1' : ePredict[1][0], 'pos1Ammount' : ePredict[0][0],
    'position2' : ePredict[1][1], 'pos2Ammount' : ePredict[0][1],
    'position3' : ePredict[1][2], 'pos3Ammount' : ePredict[0][2],
    'position4' : ePredict[1][3], 'pos4Ammount' : ePredict[0][3],
})
update_posts_ref_pos1 = ref.child('Analized_Data').child('Creative')
update_posts_ref_pos1.set({
    'position1' : cPredict[1][0], 'pos1Ammount' : cPredict[0][0],
    'position2' : cPredict[1][1], 'pos2Ammount' : cPredict[0][1],
    'position3' : cPredict[1][2], 'pos3Ammount' : cPredict[0][2],
    'position4' : cPredict[1][3], 'pos4Ammount' : cPredict[0][3],
})
update_posts_ref_pos1 = ref.child('Analized_Data').child('Motivation')
update_posts_ref_pos1.set({
    'position1' : mPredict[1][0], 'pos1Ammount' : mPredict[0][0],
    'position2' : mPredict[1][1], 'pos2Ammount' : mPredict[0][1],
    'position3' : mPredict[1][2], 'pos3Ammount' : mPredict[0][2],
    'position4' : mPredict[1][3], 'pos4Ammount' : mPredict[0][3],
})