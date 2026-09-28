import os

os.environ["GOV_SKIP_APP"] = "1"  # tests build their own app with create_app()
os.environ.pop("GOV_LLM_PROVIDER", None)

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from governance.pipeline import GovernancePipeline  # noqa: E402
from governance.retrieval import BM25Index  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def index():
    return BM25Index.from_directory(ROOT / "data" / "corpus")


@pytest.fixture
def pipeline(index):
    return GovernancePipeline(index)
