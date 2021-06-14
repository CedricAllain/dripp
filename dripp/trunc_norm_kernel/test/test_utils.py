import numpy as np

from dripp.trunc_norm_kernel.utils import get_last_timestamps


def test_get_last_timestamps():

    # ===== NORMAL CASES =====

    # 1d

    timestamps = [1, 5, 8, 10]  # already sorted
    t = [5.5, 9]
    last_timestamps = get_last_timestamps(timestamps, t)
    res = np.array([[5.0, 8.0]])  # , dtype=object)
    np.testing.assert_allclose(last_timestamps, res)

    timestamps = [8, 10]  # start "late"
    t = [5.5, 9]
    last_timestamps = get_last_timestamps(timestamps, t)
    res = np.array([[np.nan, 8.0]])  # , dtype=object)
    # np.testing.assert_array_equal(last_timestamps, res)
    np.testing.assert_allclose(last_timestamps, res, equal_nan=True)

    timestamps = [8, 10, 1, 5]  # non sorted
    t = [5.5, 9]
    last_timestamps = get_last_timestamps(timestamps, t)
    res = np.array([[5.0, 8.0]])  # , dtype=object)
    np.testing.assert_allclose(last_timestamps, res)

    # 2d

    timestamps = [[1, 5, 8, 10],
                  [3, 7, 9, 18, 29]]  # already sorted
    t = [5.5, 9]
    last_timestamps = get_last_timestamps(timestamps, t)
    res = np.array([[5.0, 8.0],
                    [3.0, 9.0]])  # , dtype=object)
    np.testing.assert_allclose(last_timestamps, res)

    # ===== PATHOLOGICAL CASES =====

    timestamps = []  # empty timestamps
    t = [5.5, 9]
    last_timestamps = get_last_timestamps(timestamps, t)
    res = np.array([[]])  # , dtype = object)
    np.testing.assert_allclose(last_timestamps, res)

    timestamps = [[1, 5, 8, 10],
                  []]  # one empty
    t = [5.5, 9]
    last_timestamps = get_last_timestamps(timestamps, t)
    res = np.array([np.array([5.0, 8.0]),
                    np.array([np.nan, np.nan])])  # , dtype=object)
    np.testing.assert_allclose(last_timestamps, res)


if __name__ == '__main__':
    test_get_last_timestamps()
