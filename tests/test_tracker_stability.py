from tracker import _stabilize_track_ids


def test_reuses_previous_person_id_after_short_detection_gap():
    rows = [
        [0, 0.0, 5, 100, 100, 160, 220, 0.9, 0],
        [1, 0.033, 5, 102, 102, 162, 222, 0.9, 0],
        [2, 0.067, 7, 100, 100, 160, 220, 0.9, 0],
        [3, 0.1, 7, 104, 104, 164, 224, 0.9, 0],
    ]

    stabilized = _stabilize_track_ids(rows, max_gap_s=1.0, max_distance_px=80)

    assert [row[2] for row in stabilized] == [5, 5, 5, 5]


def test_keeps_distinct_people_as_distinct_ids():
    rows = [
        [0, 0.0, 1, 100, 100, 160, 220, 0.9, 0],
        [1, 0.033, 2, 500, 100, 560, 220, 0.9, 0],
        [2, 0.067, 3, 1000, 100, 1060, 220, 0.9, 0],
    ]

    stabilized = _stabilize_track_ids(rows, max_gap_s=1.0, max_distance_px=80)

    assert {row[2] for row in stabilized} == {1, 2, 3}
