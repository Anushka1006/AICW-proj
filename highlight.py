import fitz
import os

keyword = input("Enter the keyword to highlight: ").strip()
pdf_folder = r"./PDF"

# Get all PDF files from the folder
pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

found = False

for pdf_path in pdf_files:
    doc = fitz.open(pdf_path)
    
    for page in doc:
        matches = page.search_for(keyword)

        if matches:
            found = True

        for match in matches:
            highlight = page.add_highlight_annot(match)
            highlight.update()
    
    # Save each file with highlighted content
    doc.save(f"highlighted_{os.path.basename(pdf_path)}")
    doc.close()

if found:
    print(f"'{keyword}' highlighted successfully in all PDFs!")
else:
    print("Keyword not found in any PDF.")