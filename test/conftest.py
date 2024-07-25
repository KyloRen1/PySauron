import pytest

from pysauron.datasets import XDViolence, UCFCrime, Avenue, ShanghaiTech
from pysauron.transforms import RandomAnomalyInject, ChangeDirection, FrameDrop


def pytest_addoption(parser):
    parser.addoption(
        "--dataset-root", action="store", default="data", help="Dataset location"
    )
    parser.addoption(
        "--frame-drop-k", action="store", default=3, help="Number of frames to drop" 
    )
    parser.addoption(
        "--anomaly-folder", action="store", 
        default="pysauron/transforms/assets/anomalies/animals", 
        help='folder with visual anomalies'
    )


@pytest.fixture(scope="module")
def xdviolence_dataset(request):
    root = request.config.getoption("--dataset-root")
    root += "/XD-Violence"
    dataset = XDViolence(
        root=root, 
        test_mode=True,
        _debug=True
    )
    return dataset


@pytest.fixture(scope="module")
def ucfcrime_dataset(request):
    root = request.config.getoption("--dataset-root")
    root += "/UCF-Crime"
    dataset = UCFCrime(
        root=root, 
        test_mode=True,
        _debug=True
    )
    return dataset


@pytest.fixture(scope="module")
def avenue_dataset(request):
    root = request.config.getoption("--dataset-root")
    root += "/AvenueDataset"
    dataset = Avenue(
        root=root, 
        test_mode=True,
        _debug=True
    )
    return dataset


@pytest.fixture(scope="module")
def shanghaitech_dataset(request):
    root = request.config.getoption("--dataset-root")
    root += "/ShanghaiTech"
    dataset = ShanghaiTech(
        root=root, 
        test_mode=True,
        _debug=True
    )
    return dataset


@pytest.fixture(scope="module")
def frame_drop_transform(request):
    k = request.config.getoption("--frame-drop-k")
    transform = FrameDrop(k=k, always_apply=True)
    return transform


@pytest.fixture(scope="module")
def change_direction_transform(request):
    transform = ChangeDirection(always_apply=True)
    return transform


@pytest.fixture(scope="module")
def random_anomaly_inject_transform(request):
    anomaly_folder = request.config.getoption("--anomaly-folder")
    transform = RandomAnomalyInject(anomaly_folder=anomaly_folder, always_apply=True)
    return transform
