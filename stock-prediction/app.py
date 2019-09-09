from StockPredictor import StockPredictor


def main():
    predictor = StockPredictor()

    predictor.train(period="Year")
    print("Year prediction:", predictor.predict(2019))

    predictor.train(period="Month")
    print("Month prediction:", predictor.predict(5))

    predictor.train(period="Day")
    print("Day prediction:", predictor.predict(7))


if __name__ == "__main__": main()