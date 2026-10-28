"""Create a one-page PDF draft; author must complete the prior-use statement."""
from pathlib import Path


OUT = Path("Nishan_AI_in_my_work.pdf")
LINES = [
    "AI USE DECLARATION - DRAFT, AUTHOR COMPLETION REQUIRED",
    "",
    "Student: S M Gazzali Arafat Nishan",
    "Project: SatScene - Home Assignment 1 Software Project",
    "",
    "Current-session AI assistance:",
    "An AI assistant reviewed the assignment specification and project files,",
    "compared deliverables with the checklist, and helped draft resource notes,",
    "input/output documentation, and presentation and recording guidance.",
    "The assistant did not train or modify the classifier during this review,",
    "and did not record a demonstration video.",
    "",
    "AUTHOR: COMPLETE BEFORE SUBMISSION",
    "This draft does not describe any AI use before this review. Add a truthful",
    "description of earlier AI assistance in code, reports, slides, or other",
    "project work, or explicitly state that there was no earlier AI use.",
    "Review the university AI guidelines and replace this draft with the final",
    "declaration. Do not submit this file until the author-completion section",
    "has been completed.",
]


def pdf_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def main() -> None:
    stream = ["BT", "/F1 11 Tf", "50 790 Td", "14 TL"]
    for i, line in enumerate(LINES):
        if i:
            stream.append("T*")
        stream.append(f"({pdf_escape(line)}) Tj")
    stream.append("ET")
    data = "\n".join(stream).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Length " + str(len(data)).encode() + b" >>\nstream\n" + data + b"\nendstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(f"{i} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    OUT.write_bytes(pdf)
    print(f"Created draft {OUT}; author completion is required")


if __name__ == "__main__":
    main()
