from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet


def generate_report(prediction, chart_paths, output="report.pdf"):
    doc = SimpleDocTemplate(output)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph(f"Predicted Price: {prediction}", styles["Title"]))

    for path in chart_paths:
        content.append(Image(path, width=400, height=300))

    doc.build(content)
    return output