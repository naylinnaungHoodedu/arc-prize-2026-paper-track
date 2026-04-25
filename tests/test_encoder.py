import numpy as np

from htba.encoder import encode_frame, object_distance


def test_encoder_extracts_connected_components():
    frame = np.zeros((6, 6), dtype=int)
    frame[1:3, 1:3] = 2
    frame[4, 4] = 3

    objects = encode_frame(frame)

    assert objects.background == 0
    assert objects.object_count == 2
    assert objects.colors == (2, 3)
    assert sorted(obj.size for obj in objects.objects) == [1, 4]


def test_motion_delta_is_attached_between_frames():
    first = np.zeros((6, 6), dtype=int)
    first[1:3, 1:3] = 2
    second = np.zeros((6, 6), dtype=int)
    second[2:4, 1:3] = 2

    encoded_first = encode_frame(first)
    encoded_second = encode_frame(second, previous=encoded_first)

    assert encoded_second.objects[0].motion_delta == (1.0, 0.0)


def test_object_distance_is_zero_for_identical_encodings():
    frame = np.zeros((6, 6), dtype=int)
    frame[2, 2] = 4

    encoded = encode_frame(frame)

    assert object_distance(encoded, encoded) == 0.0
