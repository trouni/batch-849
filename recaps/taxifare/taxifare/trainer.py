from taxifare.data import get_data, clean_df, holdout
from taxifare.model import get_model
from taxifare.pipeline import get_pipeline


class Trainer:
    def train(self):
        # Load data
        df = get_data(1000)

        # Clean data
        df = clean_df(df)

        # Holdout
        X_train, X_test, y_train, y_test = holdout(df)

        # Model
        model = get_model()

        # Pipeline
        pipeline = get_pipeline(model)

        # Train
        pipeline.fit(X_train, y_train)

        return pipeline
