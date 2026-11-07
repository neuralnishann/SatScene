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
        NSGradient,
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
from .model import build_model, checkpoint_metadata, checkpoint_not_found_message, load_checkpoint
from .runtime import best_device
from .ui_style import UI_COLORS, UI_COPY


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


def analyze_comparison(comparison_agents, image_path: str):
    results = []
    for name, agent in comparison_agents:
        try:
            results.append((name, agent.analyze_structured(image_path), None))
        except (OSError, ValueError, RuntimeError) as exc:
            results.append((name, None, str(exc)))
    return results


def color(rgb):
    return NSColor.colorWithCalibratedRed_green_blue_alpha_(rgb[0], rgb[1], rgb[2], 1.0)


def themed_color(name):
    value = UI_COLORS[name].lstrip("#")
    rgb = tuple(int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return color(rgb)


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


def themed_label(text: str, frame, size: float = 14, color_name="text", bold=False):
    value = UI_COLORS[color_name].lstrip("#")
    rgb = tuple(int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return label(text, frame, size=size, foreground=rgb, bold=bold)


class GradientView(NSView):
    def initWithFrame_(self, frame):
        self = objc.super(GradientView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.gradient = NSGradient.alloc().initWithColors_([
            themed_color("hero"),
            themed_color("hero_mid"),
            themed_color("hero_end"),
        ])
        return self

    def drawRect_(self, rect):
        self.gradient.drawInRect_angle_(self.bounds(), 115.0)


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
    def initWithAgent_device_comparisonAgents_(self, agent, device, comparison_agents):
        self = objc.super(DesktopController, self).init()
        if self is None:
            return None
        self.agent = agent
        self.device = device
        self.comparison_agents = comparison_agents
        self.comparison_columns = []
        self.image_path = None
        self.window = None
        self.preview = None
        self.report = None
        self.status = None
        self.status_detail = None
        self.status_card = None
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
        self.analyze_button = None
        self.clear_button = None
        self.is_processing = False
        self.result_queue = queue.Queue()
        self.result_timer = None
        return self

    def applicationShouldTerminateAfterLastWindowClosed_(self, app):
        return True

    def applicationDidFinishLaunching_(self, notification):
        width, height = 1240, 900
        style = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                 NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable)
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, width, height), style, NSBackingStoreBuffered, False
        )
        self.window.setTitle_("SatScene — Satellite Scene Classifier")
        self.window.center()
        self.window.setMinSize_((940, 700))
        self.window.setDelegate_(self)
        old_content = self.window.contentView()
        content = ResponsiveContentView.alloc().initWithFrame_controller_(old_content.frame(), self)
        self.window.setContentView_(content)
        self.window.setBackgroundColor_(themed_color("page"))

        header_height = 154
        self.header = GradientView.alloc().initWithFrame_(
            NSMakeRect(0, height - header_height, width, header_height)
        )
        content.addSubview_(self.header)
        self.title_label = themed_label(UI_COPY["title"], NSMakeRect(38, 49, 450, 44),
                                        31, "hero_text", bold=True)
        self.header.addSubview_(self.title_label)
        self.subtitle = themed_label(UI_COPY["kicker"], NSMakeRect(40, 103, 680, 18),
                                     10, "kicker", bold=True)
        self.header.addSubview_(self.subtitle)
        self.agent_status = themed_label(UI_COPY["intro"], NSMakeRect(40, 20, width - 80, 24),
                                         13, "hero_muted")
        self.header.addSubview_(self.agent_status)

        self.left_card = panel(NSMakeRect(24, 24, 530, height - header_height - 48),
                               tuple(int(UI_COLORS["surface"][i:i + 2], 16) / 255 for i in (1, 3, 5)),
                               tuple(int(UI_COLORS["border"][i:i + 2], 16) / 255 for i in (1, 3, 5)), radius=18)
        content.addSubview_(self.left_card)
        self.input_title = themed_label(UI_COPY["add_image"], NSMakeRect(46, height - 202, 480, 18),
                                        11, "section", bold=True)
        content.addSubview_(self.input_title)
        self.preview = ImageDropView.alloc().initWithFrame_controller_(NSMakeRect(46, 120, 486, 530), self)
        self.preview.setImageFrameStyle_(NSImageFrameGroove)
        content.addSubview_(self.preview)
        self.drop_hint = themed_label("Choose an image or drop one here",
                                      NSMakeRect(54, 248, 378, 42), 14, "muted", bold=True)
        self.drop_hint.setAlignment_(__import__("AppKit").NSTextAlignmentCenter)
        self.preview.addSubview_(self.drop_hint)
        self.file_types = themed_label(UI_COPY["image_helper"], NSMakeRect(46, 88, 486, 18),
                                       10, "muted")
        content.addSubview_(self.file_types)
        self.analyze_button = button(UI_COPY["analyze"], NSMakeRect(46, 34, 300, 40), self, "submitImage:")
        self.analyze_button.setContentTintColor_(themed_color("accent"))
        self.analyze_button.setToolTip_("Analyze the selected satellite or aerial image")
        self.analyze_button.setEnabled_(False)
        content.addSubview_(self.analyze_button)
        self.clear_button = button(UI_COPY["clear"], NSMakeRect(362, 34, 170, 40), self, "clearImage:")
        self.clear_button.setContentTintColor_(themed_color("text"))
        self.clear_button.setEnabled_(True)
        content.addSubview_(self.clear_button)

        self.right_card = panel(NSMakeRect(576, 24, width - 600, height - header_height - 48),
                                tuple(int(UI_COLORS["surface"][i:i + 2], 16) / 255 for i in (1, 3, 5)),
                                tuple(int(UI_COLORS["border"][i:i + 2], 16) / 255 for i in (1, 3, 5)), radius=18)
        content.addSubview_(self.right_card)
        self.report_title = themed_label(UI_COPY["scene_report"], NSMakeRect(598, height - 202, 520, 18),
                                         11, "section", bold=True)
        content.addSubview_(self.report_title)

        self.status_card = panel(NSMakeRect(598, height - 282, width - 644, 62),
                                 tuple(int(UI_COLORS["status_bg"][i:i + 2], 16) / 255 for i in (1, 3, 5)),
                                 tuple(int(UI_COLORS["status_border"][i:i + 2], 16) / 255 for i in (1, 3, 5)), radius=14)
        content.addSubview_(self.status_card)
        self.status = themed_label(UI_COPY["ready"], NSMakeRect(616, height - 247, width - 680, 20),
                                   13, "text", bold=True)
        content.addSubview_(self.status)
        self.status_detail = themed_label(UI_COPY["ready_detail"], NSMakeRect(616, height - 269, width - 680, 18),
                                          11, "muted")
        content.addSubview_(self.status_detail)

        self.prediction = themed_label("",
                                       NSMakeRect(598, height - 334, width - 644, 32),
                                       22, "text", bold=True)
        content.addSubview_(self.prediction)
        self.spinner = NSProgressIndicator.alloc().initWithFrame_(NSMakeRect(width - 76, height - 330, 20, 20))
        self.spinner.setStyle_(NSProgressIndicatorStyleSpinning)
        self.spinner.setDisplayedWhenStopped_(False)
        content.addSubview_(self.spinner)
        self.confidence = themed_label("", NSMakeRect(598, height - 360, width - 644, 18), 11, "muted")
        content.addSubview_(self.confidence)
        self.confidence_bar = NSProgressIndicator.alloc().initWithFrame_(NSMakeRect(598, height - 378, width - 644, 7))
        self.confidence_bar.setStyle_(NSProgressIndicatorStyleBar)
        self.confidence_bar.setMinValue_(0.0)
        self.confidence_bar.setMaxValue_(100.0)
        self.confidence_bar.setDoubleValue_(0.0)
        content.addSubview_(self.confidence_bar)

        self.alternatives_title = themed_label(UI_COPY["predictions"],
                                               NSMakeRect(598, height - 418, width - 644, 18),
                                               11, "section", bold=True)
        content.addSubview_(self.alternatives_title)
        self.alt_numbers, self.alt_labels, self.alt_scores = [], [], []
        for idx, y in enumerate((height - 454, height - 490, height - 526), start=1):
            number_view = themed_label(f"0{idx}", NSMakeRect(598, y, 32, 20), 11, "accent", bold=True)
            name_view = themed_label("—", NSMakeRect(638, y, 330, 20), 12, "text")
            score_view = themed_label("", NSMakeRect(width - 130, y, 55, 20), 11, "muted")
            self.alt_numbers.append(number_view)
            self.alt_labels.append(name_view)
            self.alt_scores.append(score_view)
            content.addSubview_(number_view)
            content.addSubview_(name_view)
            content.addSubview_(score_view)

        self.action_card = panel(NSMakeRect(598, 56, width - 644, 132),
                                 tuple(int(UI_COLORS["result_bg"][i:i + 2], 16) / 255 for i in (1, 3, 5)),
                                 tuple(int(UI_COLORS["result_border"][i:i + 2], 16) / 255 for i in (1, 3, 5)), radius=14)
        content.addSubview_(self.action_card)
        self.action_heading = themed_label(UI_COPY["follow_up"], NSMakeRect(616, 158, width - 680, 18),
                                           11, "accent", bold=True)
        content.addSubview_(self.action_heading)
        self.action_title = themed_label("Your analysis suggestion will appear here",
                                         NSMakeRect(616, 132, width - 680, 20),
                                         13, "text", bold=True)
        content.addSubview_(self.action_title)
        self.action_text = NSTextView.alloc().initWithFrame_(NSMakeRect(612, 69, width - 672, 56))
        self.action_text.setEditable_(False)
        self.action_text.setSelectable_(False)
        self.action_text.setDrawsBackground_(False)
        self.action_text.setFont_(NSFont.systemFontOfSize_(10))
        self.action_text.setTextColor_(themed_color("muted"))
        self.action_text.setString_("")
        content.addSubview_(self.action_text)

        self.warning = themed_label("", NSMakeRect(598, 202, width - 644, 20),
                                    11, "warning", bold=True)
        content.addSubview_(self.warning)
        if self.comparison_agents:
            for view in (
                self.prediction, self.confidence, self.confidence_bar,
                self.alternatives_title, *self.alt_numbers, *self.alt_labels,
                *self.alt_scores, self.action_card, self.action_heading,
                self.action_title, self.action_text, self.warning,
            ):
                view.setHidden_(True)
            for name, _ in self.comparison_agents:
                card = panel(
                    NSMakeRect(0, 0, 100, 100),
                    tuple(int(UI_COLORS["status_bg"][i:i + 2], 16) / 255 for i in (1, 3, 5)),
                    tuple(int(UI_COLORS["status_border"][i:i + 2], 16) / 255 for i in (1, 3, 5)),
                    radius=12,
                )
                heading = themed_label(name, NSMakeRect(0, 0, 100, 24), 11, "section", bold=True)
                prediction = themed_label("—", NSMakeRect(0, 0, 100, 42), 17, "text", bold=True)
                confidence = themed_label("", NSMakeRect(0, 0, 100, 20), 10, "muted")
                alternatives = [
                    (
                        themed_label(f"0{idx}", NSMakeRect(0, 0, 26, 20), 10, "accent", bold=True),
                        themed_label("—", NSMakeRect(0, 0, 100, 20), 10, "text"),
                        themed_label("", NSMakeRect(0, 0, 52, 20), 10, "muted"),
                    )
                    for idx in range(1, 4)
                ]
                content.addSubview_(card)
                content.addSubview_(heading)
                content.addSubview_(prediction)
                content.addSubview_(confidence)
                for rank, label_view, score in alternatives:
                    content.addSubview_(rank)
                    content.addSubview_(label_view)
                    content.addSubview_(score)
                self.comparison_columns.append({
                    "card": card,
                    "heading": heading,
                    "prediction": prediction,
                    "confidence": confidence,
                    "alternatives": alternatives,
                })
        self.input_title.setStringValue_(UI_COPY["add_image"])
        self.report_title.setStringValue_(UI_COPY["scene_report"])
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
        elif state == "comparison":
            self.showComparisonResult_(payload)
        else:
            self.showError_(payload)

    @objc.python_method
    def layoutWindow(self):
        """Keep the two native panels in the same proportions as the browser UI."""
        if self.window is None or self.preview is None:
            return
        bounds = self.window.contentView().bounds()
        width, height = bounds.size.width, bounds.size.height
        header_height, margin, gap, card_y = 154, 24, 22, 24
        card_height = max(500, height - header_height - 48)
        available = width - margin * 2 - gap
        left_width = available * 5.0 / 11.0
        right_x = margin + left_width + gap
        right_width = available - left_width
        top = card_y + card_height
        self.header.setFrame_(NSMakeRect(0, height - header_height, width, header_height))
        self.title_label.setFrame_(NSMakeRect(38, 49, 450, 44))
        self.subtitle.setFrame_(NSMakeRect(40, 103, 700, 18))
        self.agent_status.setFrame_(NSMakeRect(40, 20, width - 80, 24))
        self.left_card.setFrame_(NSMakeRect(margin, card_y, left_width, card_height))
        self.right_card.setFrame_(NSMakeRect(right_x, card_y, right_width, card_height))

        inner_x, inner_w = margin + 22, left_width - 44
        self.input_title.setFrame_(NSMakeRect(inner_x, top - 42, inner_w, 18))
        preview_y = card_y + 112
        preview_h = max(250, card_height - 178)
        self.preview.setFrame_(NSMakeRect(inner_x, preview_y, inner_w, preview_h))
        self.drop_hint.setFrame_(NSMakeRect(18, preview_h / 2 - 24, inner_w - 36, 48))
        self.file_types.setFrame_(NSMakeRect(inner_x, card_y + 82, inner_w, 18))
        self.analyze_button.setFrame_(NSMakeRect(inner_x, card_y + 26, inner_w * 0.62, 40))
        self.clear_button.setFrame_(NSMakeRect(inner_x + inner_w * 0.68, card_y + 26,
                                                inner_w * 0.32, 40))

        rx, rw = right_x + 22, right_width - 44
        self.report_title.setFrame_(NSMakeRect(rx, top - 42, rw, 18))
        status_y = top - 126
        self.status_card.setFrame_(NSMakeRect(rx, status_y, rw, 62))
        self.status.setFrame_(NSMakeRect(rx + 18, status_y + 35, rw - 36, 18))
        self.status_detail.setFrame_(NSMakeRect(rx + 18, status_y + 13, rw - 36, 18))
        if self.comparison_agents:
            self.spinner.setFrame_(NSMakeRect(rx + rw - 22, status_y + 21, 20, 20))
            gap = 10
            column_width = (rw - gap * (len(self.comparison_columns) - 1)) / len(self.comparison_columns)
            comparison_y = card_y + 30
            card_height = max(300, status_y - comparison_y - 18)
            for idx, column in enumerate(self.comparison_columns):
                x = rx + idx * (column_width + gap)
                column["card"].setFrame_(NSMakeRect(x, comparison_y, column_width, card_height))
                column["heading"].setFrame_(
                    NSMakeRect(x + 10, card_height + comparison_y - 40, column_width - 20, 32)
                )
                column["prediction"].setFrame_(
                    NSMakeRect(x + 10, card_height + comparison_y - 94, column_width - 20, 46)
                )
                column["confidence"].setFrame_(
                    NSMakeRect(x + 10, card_height + comparison_y - 124, column_width - 20, 18)
                )
                for row, (rank, name_view, score_view) in enumerate(column["alternatives"]):
                    y = card_height + comparison_y - 174 - row * 34
                    rank.setFrame_(NSMakeRect(x + 10, y, 24, 20))
                    name_view.setFrame_(NSMakeRect(x + 38, y, column_width - 100, 20))
                    score_view.setFrame_(NSMakeRect(x + column_width - 58, y, 48, 20))
            return
        self.prediction.setFrame_(NSMakeRect(rx, top - 174, rw, 30))
        self.spinner.setFrame_(NSMakeRect(rx + rw - 22, top - 172, 20, 20))
        self.confidence.setFrame_(NSMakeRect(rx, top - 202, rw, 18))
        self.confidence_bar.setFrame_(NSMakeRect(rx, top - 220, rw, 7))
        self.alternatives_title.setFrame_(NSMakeRect(rx, top - 258, rw, 18))
        row_ys = [top - offset for offset in (294, 330, 366)]
        for idx, y in enumerate(row_ys):
            self.alt_numbers[idx].setFrame_(NSMakeRect(rx, y, 32, 20))
            self.alt_labels[idx].setFrame_(NSMakeRect(rx + 40, y, rw - 115, 20))
            self.alt_scores[idx].setFrame_(NSMakeRect(rx + rw - 68, y, 68, 20))
        action_y, action_h = card_y + 30, 132
        self.action_card.setFrame_(NSMakeRect(rx, action_y, rw, action_h))
        self.action_heading.setFrame_(NSMakeRect(rx + 18, action_y + action_h - 30, rw - 36, 18))
        self.action_title.setFrame_(NSMakeRect(rx + 18, action_y + action_h - 56, rw - 36, 20))
        self.action_text.setFrame_(NSMakeRect(rx + 14, action_y + 10, rw - 28, 56))
        self.warning.setFrame_(NSMakeRect(rx, action_y + action_h + 10, rw, 20))
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
            self.status.setStringValue_(UI_COPY["ready"])
            self.status_detail.setStringValue_("No image was selected. Choose or drop an image to continue.")
            return
        self.loadImageFromPath(str(selected.path()))

    @objc.python_method
    def loadImageFromPath(self, path):
        path = os.path.expanduser(str(path))
        if not os.path.isfile(path):
            self.status.setStringValue_("Could not load image")
            self.status_detail.setStringValue_("The selected file is missing or is not a file.")
            return False
        if Path(path).suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
            self.status.setStringValue_("Could not load image")
            self.status_detail.setStringValue_("Choose a JPG, PNG, TIFF, or BMP image.")
            return False
        image = NSImage.alloc().initWithContentsOfFile_(path)
        if image is None:
            self.status.setStringValue_("Could not load image")
            self.status_detail.setStringValue_("Could not read that image. Try a JPG, PNG, or TIFF file.")
            return False
        self.image_path = path
        self.preview.setImage_(image)
        self.drop_hint.setHidden_(True)
        self.file_types.setStringValue_(UI_COPY["image_helper"])
        self.clear_button.setEnabled_(True)
        self.analyze_button.setEnabled_(True)
        self.status.setStringValue_("Image ready")
        self.status_detail.setStringValue_(
            "Comparing all three models…" if self.comparison_agents
            else "Select Analyze image to view ranked predictions."
        )
        self.prediction.setStringValue_("")
        self.confidence.setStringValue_("")
        self.confidence_bar.setDoubleValue_(0.0)
        self.warning.setStringValue_("")
        for name_view, score_view in zip(self.alt_labels, self.alt_scores):
            name_view.setStringValue_("—")
            score_view.setStringValue_("")
        self.action_title.setStringValue_("Your analysis suggestion will appear here")
        self.action_text.setString_("")
        self.resetComparisonResults()
        self.spinner.stopAnimation_(None)
        if self.comparison_agents:
            self.submitImage_(None)
        return True

    @objc.python_method
    def resetComparisonResults(self):
        for column in self.comparison_columns:
            column["prediction"].setStringValue_("—")
            column["confidence"].setStringValue_("")
            for _, name_view, score_view in column["alternatives"]:
                name_view.setStringValue_("—")
                score_view.setStringValue_("")

    def clearImage_(self, sender):
        if self.is_processing:
            return
        self.image_path = None
        self.preview.setImage_(None)
        self.drop_hint.setHidden_(False)
        self.status.setStringValue_(UI_COPY["ready"])
        self.status_detail.setStringValue_(UI_COPY["ready_detail"])
        self.prediction.setStringValue_("")
        self.confidence.setStringValue_("")
        self.confidence_bar.setDoubleValue_(0.0)
        self.warning.setStringValue_("")
        for name_view, score_view in zip(self.alt_labels, self.alt_scores):
            name_view.setStringValue_("—")
            score_view.setStringValue_("")
        self.action_title.setStringValue_("Your analysis suggestion will appear here")
        self.action_text.setString_("")
        self.resetComparisonResults()
        self.analyze_button.setEnabled_(False)
        self.clear_button.setEnabled_(True)

    def submitImage_(self, sender):
        if not self.image_path or self.is_processing:
            return
        self.is_processing = True
        self.clear_button.setEnabled_(False)
        self.analyze_button.setEnabled_(False)
        self.status.setStringValue_("Analyzing image")
        self.status_detail.setStringValue_(
            "Running all three models…" if self.comparison_agents else "Running the model…"
        )
        if not self.comparison_agents:
            self.prediction.setStringValue_("Analyzing image…")
        self.spinner.startAnimation_(None)
        self.resetComparisonResults()
        if self.comparison_agents:
            for column in self.comparison_columns:
                column["prediction"].setStringValue_("Analyzing…")
        self.confidence.setStringValue_("Results appear here when inference finishes")
        for name_view, score_view in zip(self.alt_labels, self.alt_scores):
            name_view.setStringValue_("—")
            score_view.setStringValue_("")
        self.action_title.setStringValue_("Preparing suggestion…")
        self.action_text.setString_("")
        self.warning.setStringValue_("")
        threading.Thread(target=self.runPrediction, args=(self.image_path,), daemon=True).start()

    @objc.python_method
    def runPrediction(self, path):
        if self.comparison_agents:
            self.result_queue.put(("comparison", analyze_comparison(self.comparison_agents, path)))
            return
        try:
            result = self.agent.analyze_structured(path)
            self.result_queue.put(("ok", result))
        except Exception as exc:
            self.result_queue.put(("error", str(exc)))

    def showResult_(self, result):
        self.prediction.setStringValue_(result["class"])
        self.confidence.setStringValue_(f"Confidence  ·  {result['confidence'] * 100:.1f}%")
        self.confidence_bar.setDoubleValue_(result["confidence"] * 100.0)
        for idx, (name_view, score_view) in enumerate(zip(self.alt_labels, self.alt_scores)):
            if idx < len(result["top_k"]):
                name, probability = result["top_k"][idx]
                name_view.setStringValue_(name)
                score_view.setStringValue_(f"{probability * 100:.1f}%")
            else:
                name_view.setStringValue_("—")
                score_view.setStringValue_("")
        if result["low_confidence"]:
            self.status.setStringValue_("Review recommended")
            self.status_detail.setStringValue_("The model is uncertain; treat this prediction as tentative.")
        else:
            self.status.setStringValue_("Analysis complete")
            self.status_detail.setStringValue_("Review the ranked scene predictions below.")
        self.warning.setStringValue_("")
        self.action_title.setStringValue_(result["title"])
        self.action_text.setString_(
            f"{result['suggested_follow_up']}\n\n{UI_COPY['follow_up_disclaimer']}"
        )
        self.spinner.stopAnimation_(None)
        self.is_processing = False
        self.clear_button.setEnabled_(True)
        self.analyze_button.setEnabled_(True)

    @objc.python_method
    def showComparisonResult_(self, results):
        errors = []
        for (name, result, error), column in zip(results, self.comparison_columns):
            if error:
                errors.append(f"{name}: {error}")
                column["prediction"].setStringValue_("Analysis failed")
                column["confidence"].setStringValue_(error)
                continue
            column["prediction"].setStringValue_(result["class"])
            column["confidence"].setStringValue_(
                f"Confidence · {result['confidence'] * 100:.1f}%"
            )
            for idx, (_, name_view, score_view) in enumerate(column["alternatives"]):
                if idx < len(result["top_k"]):
                    label_name, probability = result["top_k"][idx]
                    name_view.setStringValue_(label_name)
                    score_view.setStringValue_(f"{probability * 100:.1f}%")
                else:
                    name_view.setStringValue_("—")
                    score_view.setStringValue_("")
        if errors:
            self.status.setStringValue_("Comparison completed with errors")
            self.status_detail.setStringValue_(" · ".join(errors))
        else:
            self.status.setStringValue_("Comparison complete")
            self.status_detail.setStringValue_("All three models analyzed the image.")
        self.spinner.stopAnimation_(None)
        self.is_processing = False
        self.clear_button.setEnabled_(True)
        self.analyze_button.setEnabled_(True)

    def showError_(self, error):
        self.prediction.setStringValue_("Analysis failed")
        self.status.setStringValue_("Could not analyze this image")
        self.status_detail.setStringValue_(str(error))
        self.spinner.stopAnimation_(None)
        self.is_processing = False
        self.clear_button.setEnabled_(bool(self.image_path))
        self.analyze_button.setEnabled_(True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the native macOS SatScene window")
    parser.add_argument("--config", default="configs/default.yaml")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--checkpoint", default=None)
    checkpoint_group.add_argument(
        "--compare-checkpoints",
        nargs=3,
        metavar=("MODEL_1", "MODEL_2", "MODEL_3"),
        help="compare exactly three model checkpoints for each uploaded image",
    )
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    device = best_device()
    comparison_agents = []
    if args.compare_checkpoints:
        for checkpoint in args.compare_checkpoints:
            if not Path(checkpoint).is_file():
                raise SystemExit(checkpoint_not_found_message(checkpoint))
            metadata = checkpoint_metadata(checkpoint)
            names = resolve_classes(cfg, checkpoint)
            model_name = metadata.get("model_name", cfg["model"]["name"])
            img_size = int(metadata.get("img_size", cfg["data"]["img_size"]))
            model = build_model(len(names), model_name, pretrained=False, img_size=img_size)
            model = load_checkpoint(model, checkpoint, device)
            comparison_agents.append(
                (model_name, SatelliteAgent(model, names, device, img_size))
            )
            print(f"Loaded {model_name} ({len(names)} classes) on {device}")
        agent = comparison_agents[0][1]
    else:
        checkpoint = args.checkpoint or os.path.join(cfg["output"]["dir"], "best_model.pth")
        if not Path(checkpoint).is_file():
            raise SystemExit(checkpoint_not_found_message(checkpoint))
        metadata = checkpoint_metadata(checkpoint)
        names = resolve_classes(cfg, checkpoint)
        img_size = int(metadata.get("img_size", cfg["data"]["img_size"]))
        model = build_model(
            len(names),
            metadata.get("model_name", cfg["model"]["name"]),
            pretrained=False,
            img_size=img_size,
        )
        model = load_checkpoint(model, checkpoint, device)
        agent = SatelliteAgent(model, names, device, img_size)

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
    delegate = DesktopController.alloc().initWithAgent_device_comparisonAgents_(
        agent,
        device,
        comparison_agents,
    )
    app.setDelegate_(delegate)
    app.run()


if __name__ == "__main__":
    main()
