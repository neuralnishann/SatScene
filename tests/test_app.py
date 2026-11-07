from satscene.app import _analyze_image, _analyze_models, _clear_results, create_interface


class FakeAgent:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def analyze_structured(self, image_path):
        if self.error:
            raise self.error
        return self.result


def test_analyze_image_returns_ranked_predictions_and_follow_up():
    agent = FakeAgent({
        "class": "Forest",
        "confidence": 0.8,
        "top_k": [("Forest", 0.8), ("Grassland", 0.15), ("Mountain", 0.05)],
        "title": "Forest",
        "suggested_follow_up": "Check canopy cover.",
        "low_confidence": False,
    })

    status, predictions, follow_up = _analyze_image(agent, "scene.jpg")

    assert "Analysis complete" in status
    assert predictions == {"Forest": 0.8, "Grassland": 0.15, "Mountain": 0.05}
    assert "Check canopy cover." in follow_up


def test_analyze_image_marks_low_confidence_and_handles_missing_image():
    result = {
        "top_k": [("River", 0.3), ("Lake", 0.25)],
        "title": "River",
        "suggested_follow_up": "Review water extent.",
        "low_confidence": True,
    }

    status, predictions, _ = _analyze_image(FakeAgent(result), "scene.jpg")
    assert "uncertain" in status
    assert predictions == {"River": 0.3, "Lake": 0.25}
    assert "Choose an image" in _analyze_image(FakeAgent(), None)[0]


def test_analyze_image_surfaces_expected_errors():
    status, predictions, follow_up = _analyze_image(
        FakeAgent(error=OSError("unreadable image")),
        "scene.jpg",
    )

    assert "unreadable image" in status
    assert predictions == {}
    assert follow_up == ""


def test_clear_results_resets_image_and_all_outputs():
    image, status, predictions, follow_up, analyze_button = _clear_results()

    assert image is None
    assert "Ready to analyze" in status
    assert predictions is None
    assert follow_up == ""
    assert analyze_button["interactive"] is False


def test_analyze_models_returns_each_models_ranked_predictions():
    result = {
        "top_k": [("Forest", 0.8), ("Grassland", 0.2)],
    }
    agents = [
        ("Model A", FakeAgent(result)),
        ("Model B", FakeAgent(result)),
        ("Model C", FakeAgent(result)),
    ]

    status, *predictions = _analyze_models(agents, "scene.jpg")

    assert "All 3 models" in status
    assert predictions == [{"Forest": 0.8, "Grassland": 0.2}] * 3


def test_analyze_models_surfaces_individual_failures_without_stopping():
    result = {"top_k": [("Forest", 0.8)]}
    agents = [
        ("Model A", FakeAgent(result)),
        ("Model B", FakeAgent(error=OSError("unreadable image"))),
        ("Model C", FakeAgent(result)),
    ]

    status, *predictions = _analyze_models(agents, "scene.jpg")

    assert "Model B: unreadable image" in status
    assert predictions == [{"Forest": 0.8}, {}, {"Forest": 0.8}]


def test_interface_builds_with_clear_all_button():
    interface = create_interface(FakeAgent())

    assert interface.title == "SatScene · Satellite Scene Classifier"


def test_interface_builds_with_three_model_comparison():
    agents = [(f"Model {name}", FakeAgent()) for name in "ABC"]

    interface = create_interface(agents[0][1], comparison_agents=agents)

    assert interface.title == "SatScene · Satellite Scene Classifier"
