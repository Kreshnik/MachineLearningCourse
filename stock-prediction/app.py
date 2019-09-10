from StockPredictor import StockPredictor


def main():
    predictor = StockPredictor()

    predictor.train(period="Year")
    print("Year prediction:", predictor.predict(2020))
    predictor.plot_prediction(period="Year", numeric_value=2020)

    predictor.train(period="Month")
    print("Month prediction:", predictor.predict(9))
    predictor.plot_prediction(period="Month", numeric_value=9)

    predictor.train(period="Day")
    print("Day prediction:", predictor.predict(20))
    predictor.plot_prediction(period="Day", numeric_value=20)


if __name__ == "__main__": main()