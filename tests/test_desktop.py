from satscene import desktop
from satscene.desktop import DesktopController, analyze_comparison


class FakeAgent:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def analyze_structured(self, image_path):
        if self.error:
            raise self.error
        return self.result


def test_desktop_comparison_returns_each_model_result_and_surfaces_errors():
    agents = [
        ("Model A", FakeAgent({"class": "Forest"})),
        ("Model B", FakeAgent(error=OSError("image could not be read"))),
        ("Model C", FakeAgent({"class": "River"})),
    ]

    results = analyze_comparison(agents, "scene.jpg")

    assert results == [
        ("Model A", {"class": "Forest"}, None),
        ("Model B", None, "image could not be read"),
        ("Model C", {"class": "River"}, None),
    ]


class FakeLabel:
    def __init__(self):
        self.value = None

    def setStringValue_(self, value):
        self.value = value

    def setString_(self, value):
        self.value = value


class FakeProgress:
    def stopAnimation_(self, sender):
        self.stopped = True


class FakeButton:
    def setEnabled_(self, enabled):
        self.enabled = enabled


def test_desktop_comparison_renders_all_model_outputs():
    columns = []
    for _ in range(3):
        columns.append({
            "prediction": FakeLabel(),
            "confidence": FakeLabel(),
            "alternatives": [(None, FakeLabel(), FakeLabel()) for _ in range(3)],
        })
    controller = type("FakeController", (), {
        "comparison_columns": columns,
        "status": FakeLabel(),
        "status_detail": FakeLabel(),
        "spinner": FakeProgress(),
        "is_processing": True,
        "clear_button": FakeButton(),
        "analyze_button": FakeButton(),
    })()
    results = [
        (f"Model {idx}", {
            "class": label,
            "confidence": 0.8,
            "top_k": [(label, 0.8), ("Other", 0.2)],
        }, None)
        for idx, label in enumerate(("Forest", "River", "Urban"), start=1)
    ]

    DesktopController.showComparisonResult_(controller, results)

    assert [column["prediction"].value for column in columns] == ["Forest", "River", "Urban"]
    assert [column["confidence"].value for column in columns] == [
        "Confidence · 80.0%",
    ] * 3
    assert [column["alternatives"][0][1].value for column in columns] == [
        "Forest", "River", "Urban",
    ]
    assert controller.status.value == "Comparison complete"
    assert controller.is_processing is False


class FakeView:
    def setImage_(self, image):
        self.image = image

    def setHidden_(self, hidden):
        self.hidden = hidden

    def setStringValue_(self, value):
        self.value = value

    def setString_(self, value):
        self.value = value

    def setDoubleValue_(self, value):
        self.value = value

    def setEnabled_(self, enabled):
        self.enabled = enabled

    def stopAnimation_(self, sender):
        self.stopped = True


class FakeImage:
    @classmethod
    def alloc(cls):
        return cls()

    def initWithContentsOfFile_(self, path):
        return object()


def test_loading_image_starts_comparison_automatically(tmp_path, monkeypatch):
    image_path = tmp_path / "scene.jpg"
    image_path.touch()
    monkeypatch.setattr(desktop, "NSImage", FakeImage)
    controller = type("FakeController", (), {})()
    controller.comparison_agents = [("A", object()), ("B", object()), ("C", object())]
    controller.preview = FakeView()
    controller.drop_hint = FakeView()
    controller.file_types = FakeView()
    controller.clear_button = FakeView()
    controller.analyze_button = FakeView()
    controller.status = FakeView()
    controller.status_detail = FakeView()
    controller.prediction = FakeView()
    controller.confidence = FakeView()
    controller.confidence_bar = FakeView()
    controller.warning = FakeView()
    controller.alt_labels = []
    controller.alt_scores = []
    controller.action_title = FakeView()
    controller.action_text = FakeView()
    controller.spinner = FakeView()
    controller.resetComparisonResults = lambda: None
    def submit_image(sender):
        controller.analysis_started = True
        controller.status_detail.setStringValue_("Running all three models…")

    controller.submitImage_ = submit_image

    assert DesktopController.loadImageFromPath(controller, str(image_path))
    assert controller.analysis_started is True
    assert controller.status_detail.value == "Running all three models…"
