from src.data.make_demo import generate
from src.models.train import train


def test_training_pipeline_writes_metrics(tmp_path):
    generate(days=30, output_dir=tmp_path)
    result = train(
        data_path=tmp_path / "hourly_features.parquet",
        output_dir=tmp_path,
        models_dir=tmp_path,
    )
    assert result["train_rows"] > result["test_rows"] > 0
    assert result["models"][result["winner"]]["MAE"] >= 0
