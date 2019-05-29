from pmdarima.arima import auto_arima

# train = [0,1,2,0,1,2,0,1,2]
# model = auto_arima(train, error_action='ignore', suppress_warnings=True)
# model.fit(train)
# forcest = model.predict(n_periods=1)
# print(forcest)
import pandas as pd


pricesfile = ".\\stock_data\\DowJones.csv"
prices = pd.read_csv(pricesfile)
prices = prices.drop(['High', 'Low', 'Close', 'Volume', 'Adj Close'], 1)
prices = prices[::-1].reset_index(drop=True)



prices = prices.drop(['Date'],1)
# print(prices)
arr = prices.iloc[:,0].values
# print(arr)


def trend_direction(num1,num2):
    cur = num2 - num1
    res = cur / num1
    if res< -0.0041:
        return 0
    elif -0.0041 <= res <= 0.0087:
        return 1
    else:
        return 2
print(len(arr))

arr = arr[:800]

total = 0
right = 0
T = 15
for i in range(len(arr)- T):
    print(i)
    train = arr[i:i+T]
    target = arr[i+T]

    model = auto_arima(train, error_action='ignore', suppress_warnings=True)

    model.fit(train)

    forcest = model.predict(n_periods=1)

    truth = trend_direction(train[-1],target)
    predict = trend_direction(train[-1],forcest[0])

    total += 1
    if truth == predict:
        right += 1
acc = right/total

print(acc)


