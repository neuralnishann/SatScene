"""Build an editable seven-slide PowerPoint deck using only the Python standard library."""
from __future__ import annotations

from html import escape
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


OUT = Path("Nishan_SMGazzaliArafat_individual_SatScene.pptx")
SLIDES = [
    ("SatScene", "Land-use scene classification from overhead images\n\nGeospatial Data Science · Home Assignment 1 · Software Project",
     "Problem and scope · 0:45"),
    ("Data and labels", "1,000 images across 14 classes\n700 train · 200 validation · 100 test\n\nScene label = most frequent object class in the YOLO annotation.\nLabels are derived proxies; images have no CRS or geotransform.",
     "Data and label limitations · 1:00"),
    ("Method and software", "ConvNeXt-Tiny pretrained on ImageNet, provided by timm\n\nProject code: YOLO parsing, training, validation, test metrics, local Gradio GUI, and rule-based follow-up suggestions.\n\nThe suggestions support an analyst; they do not run GIS operations.",
     "Method and implementation · 1:15"),
    ("Validation result", "Seeded packaged run · checkpoint selected by validation macro-F1\n\nAccuracy: 77.5% (155/200)\nMacro-F1: 0.761\nSelected epoch: 4 of 10\n\nOne run; validation was used for checkpoint selection.",
     "Validation · 1:00"),
    ("Held-out test: failed", "Accuracy: 6% (6/100)\nMacro-F1: 0.008\nBalanced accuracy: 0.071\n\nAll 100 images were predicted as Agriculture with near-unit confidence.\n\nThis is not a usable classifier result. Cause is not established; the test set was not used for tuning.",
     "Test result · 1:00"),
    ("Software demonstration", "Show the local GUI: launch · upload · prediction · alternatives · follow-up suggestion\n\nUse iodata/sample_input as the reproducible example. Explain that this checkpoint is unreliable and show its failure honestly.\n\nReplace this text with a live capture or screenshot in the recorded presentation.",
     "Live GUI demo · 1:30"),
    ("Limits and next steps", "Investigate label and split differences using train/validation evidence.\n\nDo not tune on the held-out test set. After fixing and selecting a recipe on validation, reserve an untouched final evaluation.\n\nSoftware workflow is implemented; model quality remains unresolved.",
     "Conclusion · 0:30"),
]

NS = 'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
RNS = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'


def text_shape(shape_id: int, name: str, text: str, x: int, y: int, w: int, h: int,
               size: int, color: str, bold: bool = False) -> str:
    paras = []
    for line in text.split("\n"):
        if line == "":
            paras.append("<a:p><a:endParaRPr lang=\"en-US\"/></a:p>")
        else:
            paras.append(
                f'<a:p><a:pPr/><a:r><a:rPr lang="en-US" sz="{size}" b="{int(bold)}">'
                f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
                f'<a:latin typeface="Aptos"/></a:rPr><a:t>{escape(line)}</a:t></a:r>'
                f'<a:endParaRPr lang="en-US"/></a:p>'
            )
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{shape_id}" name="{escape(name)}"/>'
        '<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr>'
        f'<a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln>'
        '</p:spPr><p:txBody><a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0"/>'
        '<a:lstStyle/>' + ''.join(paras) + '</p:txBody></p:sp>'
    )


def slide_xml(title: str, body: str, footer: str) -> str:
    shapes = (
        text_shape(2, "Title", title, 700000, 480000, 10800000, 1100000, 3400, "17365D", True)
        + text_shape(3, "Content", body, 800000, 2050000, 10500000, 4400000, 2200, "202A35")
        + text_shape(4, "Timing", footer, 800000, 6500000, 10500000, 350000, 1300, "3973AC")
    )
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<p:sld {NS} {RNS}><p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="F7F9FC"/>'
            '</a:solidFill><a:effectLst/></p:bgPr></p:bg><p:spTree>'
            '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
            '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
            '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
            + shapes + '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>')


def main() -> None:
    content = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
               '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
               '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
               '<Default Extension="xml" ContentType="application/xml"/>',
               '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
               '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
               '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
               '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>']
    for i in range(1, len(SLIDES) + 1):
        content.append(f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>')
    content.append('</Types>')
    pres_ids = ''.join(f'<p:sldId id="{255+i}" r:id="rId{i+1}"/>' for i in range(len(SLIDES)))
    rels = ['<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>']
    rels.extend(f'<Relationship Id="rId{i+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>' for i in range(1, len(SLIDES)+1))
    master = (f'<p:sldMaster {NS} {RNS}><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/>'
              '<p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/>'
              '<a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/>'
              '</a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMap accent1="accent1" accent2="accent2" '
              'accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" '
              'dk1="dk1" dk2="dk2" folHlink="folHlink" hlink="hlink" lt1="lt1" lt2="lt2"/><p:sldLayoutIdLst>'
              '<p:sldLayoutId id="1" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles/></p:sldMaster>')
    layout = (f'<p:sldLayout {NS} type="blank" preserve="1"><p:cSld name="Blank">'
              '<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/>'
              '</p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
              '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
              '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>')
    theme = ('<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="SatScene">'
             '<a:themeElements><a:clrScheme name="SatScene"><a:dk1><a:srgbClr val="202A35"/></a:dk1>'
             '<a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="17365D"/></a:dk2>'
             '<a:lt2><a:srgbClr val="F7F9FC"/></a:lt2><a:accent1><a:srgbClr val="3973AC"/></a:accent1>'
             '<a:accent2><a:srgbClr val="70AD47"/></a:accent2><a:accent3><a:srgbClr val="ED7D31"/></a:accent3>'
             '<a:accent4><a:srgbClr val="A5A5A5"/></a:accent4><a:accent5><a:srgbClr val="FFC000"/></a:accent5>'
             '<a:accent6><a:srgbClr val="5B9BD5"/></a:accent6><a:hlink><a:srgbClr val="0563C1"/></a:hlink>'
             '<a:folHlink><a:srgbClr val="954F72"/></a:folHlink></a:clrScheme>'
             '<a:fontScheme name="Aptos"><a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont>'
             '<a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme>'
             '<a:fmtScheme name="SatScene"><a:fillStyleLst/><a:lnStyleLst/><a:effectStyleLst/><a:bgFillStyleLst/></a:fmtScheme>'
             '</a:themeElements></a:theme>')
    with ZipFile(OUT, "w", ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ''.join(content))
        z.writestr("_rels/.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/></Relationships>')
        z.writestr("ppt/presentation.xml", f'<p:presentation {NS} {RNS}><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:sldIdLst>{pres_ids}</p:sldIdLst><p:sldSz cx="12192000" cy="6858000" type="wide"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>')
        z.writestr("ppt/_rels/presentation.xml.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(rels)+'</Relationships>')
        z.writestr("ppt/slideMasters/slideMaster1.xml", master)
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/></Relationships>')
        z.writestr("ppt/slideLayouts/slideLayout1.xml", layout)
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/></Relationships>')
        z.writestr("ppt/theme/theme1.xml", theme)
        for i, (title, body, timing) in enumerate(SLIDES, 1):
            z.writestr(f"ppt/slides/slide{i}.xml", slide_xml(title, body, timing))
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/></Relationships>')
    print(f"Created {OUT} with {len(SLIDES)} slides")


if __name__ == "__main__":
    main()
