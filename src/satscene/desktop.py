"""Native macOS Cocoa GUI; no browser or local web server is started."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import queue
import sys
import threading

import torch
import yaml
import objc

try:
    from AppKit import (
        NSApplication, NSApplicationActivationPolicyRegular, NSBackingStoreBuffered,
        NSBox, NSBoxCustom, NSButton, NSColor, NSFont, NSImage,
        NSImageFrameGroove, NSImageScaleProportionallyUpOrDown,
        NSImageView, NSLineBorder, NSMakeRect, NSNoTitle, NSOpenPanel, NSProgressIndicator,
        NSProgressIndicatorStyleBar, NSProgressIndicatorStyleSpinning, NSScrollView,
        NSTextField, NSTextView, NSView,
        NSWindow, NSWindowStyleMaskClosable, NSWindowStyleMaskMiniaturizable,
        NSWindowStyleMaskResizable, NSWindowStyleMaskTitled, NSDragOperationCopy,
        NSFilenamesPboardType, NSTextAlignmentCenter,
    )
    from Foundation import NSObject, NSTimer, NSURL
except ImportError as exc:
    raise SystemExit(
        "The native Mac GUI needs PyObjC. Install project dependencies with "
        "`.venv/bin/python -m pip install -r requirements.txt`."
    ) from exc

from .agent import SatelliteAgent
from .labels import read_class_names
from .model import build_model, checkpoint_metadata, load_checkpoint
from .runtime import best_device


def resolve_classes(cfg: dict, checkpoint: str) -> list[str]:
    metadata = checkpoint_metadata(checkpoint)
    names = metadata.get("class_names") or cfg.get("class_names")
    if not names:
        yaml_path = Path(cfg["data"]["root"]) / "data.yaml"
        if yaml_path.is_file():
            names = read_class_names(str(yaml_path))
    if not names:
        raise RuntimeError("Class names are unavailable; provide data.yaml or a metadata checkpoint.")
    return list(names)


def color(rgb):
    return NSColor.colorWithCalibratedRed_green_blue_alpha_(rgb[0], rgb[1], rgb[2], 1.0)


def panel(frame, background=(1, 1, 1), border=None, radius=15):
    view = NSBox.alloc().initWithFrame_(frame)
    view.setBoxType_(NSBoxCustom)
    view.setTitle_("")
    view.setTitlePosition_(NSNoTitle)
    view.setBorderType_(NSLineBorder if border else 0)
    view.setFillColor_(color(background))
    view.setCornerRadius_(radius)
    if border:
        view.setBorderColor_(color(border))
    return view


def label(text: str, frame, size: float = 14, foreground=(0.12, 0.17, 0.25), bold=False):
    view = NSTextField.alloc().initWithFrame_(frame)
    view.setStringValue_(text)
    view.setEditable_(False)
    view.setSelectable_(False)
    view.setBezeled_(False)
    view.setDrawsBackground_(False)
    view.setTextColor_(color(foreground))
    view.setFont_(NSFont.boldSystemFontOfSize_(size) if bold else NSFont.systemFontOfSize_(size))
    return view


def button(title: str, frame, target, action: str):
    view = NSButton.alloc().initWithFrame_(frame)
    view.setTitle_(title)
    view.setTarget_(target)
    view.setAction_(action)
    view.setBezelStyle_(1)
    return view


class ImageDropView(NSImageView):
    """Image preview that also accepts image files dragged from Finder."""
    def initWithFrame_controller_(self, frame, controller):
        self = objc.super(ImageDropView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.controller = controller
        # Finder uses file-list pasteboards; some apps expose file URLs instead.
        self.registerForDraggedTypes_([NSFilenamesPboardType, "public.file-url"])
        self.setImageScaling_(NSImageScaleProportionallyUpOrDown)
        self.setImageFrameStyle_(NSImageFrameGroove)
        return self

    def draggingEntered_(self, sender):
        return NSDragOperationCopy if self._first_path(sender.draggingPasteboard()) else 0

    def draggingUpdated_(self, sender):
        return self.draggingEntered_(sender)

    def prepareForDragOperation_(self, sender):
        return bool(self._first_path(sender.draggingPasteboard()))

    def performDragOperation_(self, sender):
        path = self._first_path(sender.draggingPasteboard())
        if not path:
            return False
        return self.controller.loadImageFromPath(str(path))

    def mouseDown_(self, event):
        # Make the full drop target a convenient file-picker target as well.
        self.controller.chooseImage_(None)

    @objc.python_method
    def _first_path(self, pasteboard):
        paths = pasteboard.propertyListForType_(NSFilenamesPboardType)
        if paths:
            return str(paths[0])
        file_url = pasteboard.stringForType_("public.file-url")
        if file_url:
            url = NSURL.URLWithString_(str(file_url))
            return str(url.path()) if url else None
        return None


class ResponsiveContentView(NSView):
    """Notify the controller whenever AppKit changes the content size."""
    def initWithFrame_controller_(self, frame, controller):
        self = objc.super(ResponsiveContentView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.controller = controller
        self.setAutoresizesSubviews_(False)
        return self

    def resizeSubviewsWithOldSize_(self, old_size):
        objc.super(ResponsiveContentView, self).resizeSubviewsWithOldSize_(old_size)
        self.controller.layoutWindow()

    def setFrameSize_(self, new_size):
        objc.super(ResponsiveContentView, self).setFrameSize_(new_size)
        self.controller.layoutWindow()


class DesktopController(NSObject):
    def initWithAgent_device_(self, agent, device):
        self = objc.super(DesktopController, self).init()
        if self is None:
            return None
        self.agent = agent
        self.device = device
        self.image_path = None
        self.window = None
        self.preview = None
        self.report = None
        self.status = None
        self.agent_status = None
        self.prediction = None
        self.spinner = None
        self.confidence = None
        self.confidence_bar = None
        self.alt_numbers = []
        self.alt_labels = []
        self.alt_scores = []
        self.action_title = None
        self.action_text = None
        self.warning = None
        self.drop_hint = None
        self.header = None
        self.title_label = None
        self.subtitle = None
        self.left_card = None
        self.input_title = None
        self.file_types = None
        self.right_card = None
        self.report_title = None
        self.alternatives_title = None
        self.action_card = None
        self.action_heading = None
        self.upload_button = None
        self.submit_button = None
        self.is_processing = False
        self.result_queue = queue.Queue()
        self.result_timer = None
        return self

    def applicationShouldTerminateAfterLastWindowClosed_(self, app):
        return True

    def applicationDidFinishLaunching_(self, notification):
        width, height = 1160, 790
        style = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                 NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable)
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, width, height), style, NSBackingStoreBuffered, False
        )
        self.window.setTitle_("SatScene — Satellite Scene Classifier")
        self.window.center()
        self.window.setMinSize_((900, 650))
        self.window.setDelegate_(self)
        old_content = self.window.contentView()
        content = ResponsiveContentView.alloc().initWithFrame_controller_(old_content.frame(), self)
        self.window.setContentView_(content)
        self.window.setBackgroundColor_(color((0.95, 0.97, 0.99)))

        self.header = panel(NSMakeRect(0, height - 112, width, 112), (0.075, 0.13, 0.24), radius=0)
        content.addSubview_(self.header)
        self.title_label = label("SatScene", NSMakeRect(0, 60, width, 32), 25,
                                 foreground=(1, 1, 1), bold=True)
        self.title_label.setAlignment_(NSTextAlignmentCenter)
        self.header.addSubview_(self.title_label)
        self.subtitle = label("PERCEIVE → REASON → REPORT",
                              NSMakeRect(0, 32, width, 19), 10,
                              foreground=(0.76, 0.86, 0.94), bold=True)
        self.subtitle.setAlignment_(NSTextAlignmentCenter)
        self.header.addSubview_(self.subtitle)
        self.agent_status = label(f" READY  ·  {str(self.device).upper()}",
                                  NSMakeRect(width - 245, 44, 215, 25), 11,
                                  foreground=(0.72, 0.96, 0.86), bold=True)
        self.header.addSubview_(self.agent_status)

        self.left_card = panel(NSMakeRect(25, 70, 520, height - 220), (1, 1, 1), (0.87, 0.90, 0.94))
        content.addSubview_(self.left_card)
        self.input_title = label("UPLOAD IMAGE", NSMakeRect(48, height - 180, 300, 20), 11,
                                 foreground=(0.25, 0.43, 0.56), bold=True)
        content.addSubview_(self.input_title)
        self.preview = ImageDropView.alloc().initWithFrame_controller_(
            NSMakeRect(48, 155, 474, height - 335), self
        )
        self.preview.setImageFrameStyle_(NSImageFrameGroove)
        content.addSubview_(self.preview)
        self.drop_hint = label("DROP AN IMAGE\n\nDrag a satellite image here\nor use the Upload Image button",
                               NSMakeRect(90, 320, 390, 90), 15,
                               foreground=(0.20, 0.45, 0.56), bold=True)
        self.drop_hint.setAlignment_(__import__("AppKit").NSTextAlignmentCenter)
        self.preview.addSubview_(self.drop_hint)
        self.file_types = label("JPG · PNG · TIFF · BMP", NSMakeRect(50, 125, 350, 18), 10,
                                foreground=(0.49, 0.55, 0.62))
        content.addSubview_(self.file_types)
        self.upload_button = button("Upload Image", NSMakeRect(48, 82, 190, 38), self, "chooseImage:")
        self.upload_button.setContentTintColor_(color((0.07, 0.48, 0.63)))
        self.upload_button.setToolTip_("Browse your Mac for a satellite or aerial image")
        content.addSubview_(self.upload_button)
        self.submit_button = button("Submit Image to Model", NSMakeRect(300, 82, 222, 38), self, "submitImage:")
        self.submit_button.setEnabled_(False)
        self.submit_button.setContentTintColor_(color((0.07, 0.48, 0.63)))
        content.addSubview_(self.submit_button)

        self.right_card = panel(NSMakeRect(565, 70, 570, height - 220), (1, 1, 1), (0.87, 0.90, 0.94))
        content.addSubview_(self.right_card)
        self.report_title = label("AGENT RESULT", NSMakeRect(590, height - 180, 280, 20), 11,
                                  foreground=(0.25, 0.43, 0.56), bold=True)
        content.addSubview_(self.report_title)
        self.prediction = label("Waiting for an image", NSMakeRect(590, height - 235, 500, 42), 25,
                                foreground=(0.075, 0.13, 0.24), bold=True)
        content.addSubview_(self.prediction)
        self.spinner = NSProgressIndicator.alloc().initWithFrame_(NSMakeRect(1080, height - 228, 22, 22))
        self.spinner.setStyle_(NSProgressIndicatorStyleSpinning)
        self.spinner.setDisplayedWhenStopped_(False)
        content.addSubview_(self.spinner)
        self.confidence = label("Confidence will appear after analysis", NSMakeRect(590, height - 270, 490, 22), 12,
                                foreground=(0.38, 0.44, 0.52))
        content.addSubview_(self.confidence)
        self.confidence_bar = NSProgressIndicator.alloc().initWithFrame_(NSMakeRect(590, height - 294, 500, 9))
        self.confidence_bar.setStyle_(NSProgressIndicatorStyleBar)
        self.confidence_bar.setMinValue_(0.0)
        self.confidence_bar.setMaxValue_(100.0)
        self.confidence_bar.setDoubleValue_(0.0)
        content.addSubview_(self.confidence_bar)

        self.alternatives_title = label("TOP ALTERNATIVES", NSMakeRect(590, height - 335, 260, 18), 10,
                                        foreground=(0.38, 0.47, 0.56), bold=True)
        content.addSubview_(self.alternatives_title)
        self.alt_numbers, self.alt_labels, self.alt_scores = [], [], []
        for idx, y in enumerate((height - 377, height - 414, height - 451), start=1):
            number_view = label(f"0{idx}", NSMakeRect(590, y, 32, 22), 11,
                                foreground=(0.10, 0.52, 0.63), bold=True)
            content.addSubview_(number_view)
            name_view = label("—", NSMakeRect(630, y, 320, 22), 13)
            score_view = label("", NSMakeRect(950, y, 140, 22), 12,
                               foreground=(0.40, 0.47, 0.55))
            self.alt_numbers.append(number_view)
            content.addSubview_(name_view)
            content.addSubview_(score_view)
            self.alt_labels.append(name_view)
            self.alt_scores.append(score_view)

        self.action_card = panel(NSMakeRect(588, 142, 524, 138), (0.91, 0.96, 0.98),
                                 (0.82, 0.90, 0.94), radius=12)
        content.addSubview_(self.action_card)
        self.action_heading = label("SUGGESTED GEO FOLLOW-UP", NSMakeRect(608, 247, 470, 18), 10,
                                    foreground=(0.10, 0.48, 0.58), bold=True)
        content.addSubview_(self.action_heading)
        self.action_title = label("Your analysis suggestion will appear here", NSMakeRect(608, 216, 475, 24), 14,
                                  foreground=(0.075, 0.13, 0.24), bold=True)
        content.addSubview_(self.action_title)
        self.action_text = NSTextView.alloc().initWithFrame_(NSMakeRect(604, 153, 490, 55))
        self.action_text.setEditable_(False)
        self.action_text.setSelectable_(False)
        self.action_text.setDrawsBackground_(False)
        self.action_text.setFont_(NSFont.systemFontOfSize_(12))
        self.action_text.setTextColor_(color((0.26, 0.34, 0.42)))
        content.addSubview_(self.action_text)

        self.warning = label("", NSMakeRect(590, 103, 510, 25), 11,
                             foreground=(0.70, 0.34, 0.08), bold=True)
        content.addSubview_(self.warning)
        self.status = label("Model suggestions are decision support for analyst review.",
                            NSMakeRect(590, 82, 510, 18), 10,
                            foreground=(0.49, 0.55, 0.62))
        content.addSubview_(self.status)
        self.layoutWindow()
        self.result_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.1, self, "pollResults:", None, True
        )
        self.window.makeKeyAndOrderFront_(None)
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)

    def windowDidResize_(self, notification):
        self.layoutWindow()

    def pollResults_(self, timer):
        try:
            state, payload = self.result_queue.get_nowait()
        except queue.Empty:
            return
        if state == "ok":
            self.showResult_(payload)
        else:
            self.showError_(payload)

    @objc.python_method
    def layoutWindow(self):
        """Reflow the two cards and their contents to the current window size."""
        if self.window is None or self.preview is None:
            return
        bounds = self.window.contentView().bounds()
        width, height = bounds.size.width, bounds.size.height
        margin, gap, card_y, header_height = 25, 20, 24, 112
        card_height = max(450, height - header_height - 42)
        vertical_scale = min(1.0, max(0.75, card_height / 600))
        v = lambda value: value * vertical_scale
        left_width = (width - margin * 2 - gap) * 0.49
        right_x = margin + left_width + gap
        right_width = width - right_x - margin
        self.header.setFrame_(NSMakeRect(0, height - header_height, width, header_height))
        self.title_label.setFrame_(NSMakeRect(0, 60, width, 32))
        self.subtitle.setFrame_(NSMakeRect(0, 32, width, 19))
        self.agent_status.setFrame_(NSMakeRect(width - 245, 44, 215, 25))
        self.left_card.setFrame_(NSMakeRect(margin, card_y, left_width, card_height))
        self.right_card.setFrame_(NSMakeRect(right_x, card_y, right_width, card_height))

        inset = 22
        preview_x, preview_y = margin + inset, card_y + 120
        preview_width = left_width - inset * 2
        preview_height = max(220, card_height - 185)
        self.input_title.setFrame_(NSMakeRect(preview_x, card_y + card_height - 38, preview_width, 20))
        self.preview.setFrame_(NSMakeRect(preview_x, preview_y, preview_width, preview_height))
        self.drop_hint.setFrame_(NSMakeRect(18, max(35, preview_height / 2 - 50), preview_width - 36, 100))
        self.file_types.setFrame_(NSMakeRect(preview_x, card_y + 91, preview_width, 18))
        button_y = card_y + 38
        upload_width, submit_width = 145, min(220, preview_width - 155)
        self.upload_button.setFrame_(NSMakeRect(preview_x, button_y, upload_width, 38))
        self.submit_button.setFrame_(NSMakeRect(preview_x + preview_width - submit_width, button_y,
                                                submit_width, 38))

        rx = right_x + inset
        rw = right_width - inset * 2
        top = card_y + card_height
        self.report_title.setFrame_(NSMakeRect(rx, top - v(38), rw, v(20)))
        self.prediction.setFrame_(NSMakeRect(rx, top - v(95), rw, v(42)))
        self.spinner.setFrame_(NSMakeRect(rx + rw - 24, top - v(88), 20, 20))
        self.confidence.setFrame_(NSMakeRect(rx, top - v(126), rw, v(22)))
        self.confidence_bar.setFrame_(NSMakeRect(rx, top - v(149), rw, v(9)))
        self.alternatives_title.setFrame_(NSMakeRect(rx, top - v(188), rw, v(18)))
        row_y = [top - v(offset) for offset in (229, 266, 303)]
        for idx, y in enumerate(row_y):
            self.alt_numbers[idx].setFrame_(NSMakeRect(rx, y, 32, v(22)))
            self.alt_labels[idx].setFrame_(NSMakeRect(rx + 38, y, rw - 150, v(22)))
            self.alt_scores[idx].setFrame_(NSMakeRect(rx + rw - 105, y, 105, v(22)))
        action_height = v(138)
        action_y = max(card_y + v(50), row_y[2] - v(15) - action_height)
        self.action_card.setFrame_(NSMakeRect(rx, action_y, rw, action_height))
        self.action_heading.setFrame_(NSMakeRect(rx + 18, action_y + action_height - v(30), rw - 36, v(18)))
        self.action_title.setFrame_(NSMakeRect(rx + 18, action_y + action_height - v(60), rw - 36, v(24)))
        self.action_text.setFrame_(NSMakeRect(rx + 14, action_y + v(12), rw - 28, v(58)))
        self.warning.setFrame_(NSMakeRect(rx, action_y - v(32), rw, v(22)))
        self.status.setFrame_(NSMakeRect(rx, action_y - v(58), rw, v(18)))

    def chooseImage_(self, sender):
        panel = NSOpenPanel.openPanel()
        panel.setCanChooseFiles_(True)
        panel.setCanChooseDirectories_(False)
        panel.setAllowsMultipleSelection_(False)
        panel.setTitle_("Upload a satellite or aerial image")
        panel.setPrompt_("Upload Image")
        panel.setAllowedFileTypes_(["jpg", "jpeg", "png", "tif", "tiff", "bmp"])
        response = panel.runModal()
        if response != 1:
            return
        selected = panel.URL()
        if selected is None:
            self.status.setStringValue_("No image was selected. Try Upload Image again.")
            return
        self.loadImageFromPath(str(selected.path()))

    @objc.python_method
    def loadImageFromPath(self, path):
        path = os.path.expanduser(str(path))
        if not os.path.isfile(path):
            self.status.setStringValue_("The selected file is missing or is not a file.")
            return False
        if Path(path).suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
            self.status.setStringValue_("Choose a JPG, PNG, TIFF, or BMP image.")
            return False
        image = NSImage.alloc().initWithContentsOfFile_(path)
        if image is None:
            self.status.setStringValue_("Could not read that image. Try a JPG, PNG, or TIFF file.")
            return False
        self.image_path = path
        self.preview.setImage_(image)
        self.drop_hint.setHidden_(True)
        self.file_types.setStringValue_(f"Selected image: {Path(path).name}")
        self.submit_button.setEnabled_(True)
        self.upload_button.setEnabled_(True)
        self.status.setStringValue_(f"Image ready · {Path(path).name}")
        self.prediction.setStringValue_("Image ready to submit")
        self.confidence.setStringValue_("Submit this image to run the model")
        self.confidence_bar.setDoubleValue_(0.0)
        self.warning.setStringValue_("")
        for name_view, score_view in zip(self.alt_labels, self.alt_scores):
            name_view.setStringValue_("—")
            score_view.setStringValue_("")
        self.action_title.setStringValue_("Waiting for model output")
        self.action_text.setString_("")
        self.agent_status.setStringValue_("●  IMAGE READY")
        self.spinner.stopAnimation_(None)
        self.submitImage_(None)
        return True

    def submitImage_(self, sender):
        if not self.image_path or self.is_processing:
            return
        self.is_processing = True
        self.submit_button.setEnabled_(False)
        self.upload_button.setEnabled_(False)
        self.status.setStringValue_("Image submitted · running model in the backend…")
        self.agent_status.setStringValue_("●  ANALYZING")
        self.prediction.setStringValue_("Model is processing the image…")
        self.spinner.startAnimation_(None)
        self.confidence.setStringValue_("Results appear here when inference finishes")
        for name_view, score_view in zip(self.alt_labels, self.alt_scores):
            name_view.setStringValue_("—")
            score_view.setStringValue_("")
        self.action_title.setStringValue_("Generating follow-up suggestion…")
        self.action_text.setString_("")
        self.warning.setStringValue_("")
        threading.Thread(target=self.runPrediction, args=(self.image_path,), daemon=True).start()

    @objc.python_method
    def runPrediction(self, path):
        try:
            result = self.agent.analyze_structured(path)
            self.result_queue.put(("ok", result))
        except Exception as exc:
            self.result_queue.put(("error", str(exc)))

    def showResult_(self, result):
        self.prediction.setStringValue_(result["class"])
        self.confidence.setStringValue_(f"Model confidence  ·  {result['confidence'] * 100:.1f}%")
        self.confidence_bar.setDoubleValue_(result["confidence"] * 100.0)
        for idx, (name_view, score_view) in enumerate(zip(self.alt_labels, self.alt_scores)):
            if idx < len(result["top_k"]):
                name, probability = result["top_k"][idx]
                name_view.setStringValue_(name)
                score_view.setStringValue_(f"{probability * 100:.1f}%")
            else:
                name_view.setStringValue_("—")
                score_view.setStringValue_("")
        self.action_title.setStringValue_(result["title"])
        self.action_text.setString_(result["suggested_follow_up"])
        self.warning.setStringValue_(
            "⚠  LOW CONFIDENCE — review this label before acting" if result["low_confidence"] else ""
        )
        self.agent_status.setStringValue_("")
        self.spinner.stopAnimation_(None)
        self.is_processing = False
        self.submit_button.setEnabled_(True)
        self.upload_button.setEnabled_(True)

    def showError_(self, error):
        self.prediction.setStringValue_("Analysis failed")
        self.status.setStringValue_(str(error))
        self.agent_status.setStringValue_("●  CHECK INPUT")
        self.spinner.stopAnimation_(None)
        self.is_processing = False
        self.submit_button.setEnabled_(bool(self.image_path))
        self.upload_button.setEnabled_(True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the native macOS SatScene window")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    checkpoint = args.checkpoint or os.path.join(cfg["output"]["dir"], "best_model.pth")
    if not Path(checkpoint).is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}. Train first or pass --checkpoint.")

    device = best_device()
    metadata = checkpoint_metadata(checkpoint)
    names = resolve_classes(cfg, checkpoint)
    model = build_model(len(names), metadata.get("model_name", cfg["model"]["name"]),
                        pretrained=False, img_size=int(metadata.get("img_size", cfg["data"]["img_size"])))
    model = load_checkpoint(model, checkpoint, device)

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
    delegate = DesktopController.alloc().initWithAgent_device_(
        SatelliteAgent(model, names, device, int(metadata.get("img_size", cfg["data"]["img_size"]))),
        device,
    )
    app.setDelegate_(delegate)
    app.run()


if __name__ == "__main__":
    main()
