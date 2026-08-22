from __future__ import annotations


def make_fixture_rows():
    return [
        {
            "id": 1,
            "img_type": "single_view",
            "format_type": "fill",
            "task": "distance_prediction_oc",
            "source": "synthetic",
            "image": ["rgb-1.jpg"],
            "depth": ["depth-1.npy"],
            "pose": [[1, 0, 0, 0]],
            "intrinsic_color": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "intrinsic_depth": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "question": "How far?",
            "answer": "3.5",
        },
        {
            "id": "2",
            "img_type": "single_view",
            "format_type": "select",
            "task": "distance_prediction_oc",
            "source": "synthetic",
            "image": ["rgb-2.jpg"],
            "depth": [],
            "pose": [[1, 0, 0, 0]],
            "intrinsic_color": [],
            "intrinsic_depth": [],
            "question": "Which?",
            "answer": "A",
        },
        {
            "id": 3,
            "img_type": "multi_view",
            "format_type": "fill",
            "task": "spatial_relation_oo",
            "source": "synthetic",
            "image": ["rgb-3-a.jpg", "rgb-3-b.jpg"],
            "depth": ["depth-3-a.npy", "depth-3-b.npy"],
            "pose": [],
            "intrinsic_color": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "intrinsic_depth": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "question": "Where?",
            "answer": "left",
        },
    ]
