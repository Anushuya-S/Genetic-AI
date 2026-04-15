from fpdf import FPDF

# Create book.pdf
pdf1 = FPDF()
pdf1.add_page()
pdf1.set_font("Arial", size=12)

for i in range(1, 6):
    pdf1.cell(200, 10, txt=f"Book Content Page {i}", ln=True)

pdf1.output("book.pdf")

# Create notes.pdf
pdf2 = FPDF()
pdf2.add_page()
pdf2.set_font("Arial", size=12)

for i in range(1, 6):
    pdf2.cell(200, 10, txt=f"Notes Content Page {i}", ln=True)

pdf2.output("notes.pdf")

print("PDFs created successfully!")