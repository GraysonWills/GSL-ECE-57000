import mlflow

def track_experiment(model, hyperparameters, metrics, run_name="Experiment"):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(hyperparameters)
        mlflow.log_metrics(metrics)
        mlflow.pytorch.log_model(model, "model")

# Example usage:
# track_experiment(corrnet, hyperparameters={"lr": 0.001, "epochs": 20}, metrics={"WER": 0.15})
