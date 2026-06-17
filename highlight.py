import fitz

keyword = input("Enter the keyword to highlight: ").strip()

doc = fitz.open("pdf/Nature.pdf")

found = False

for page in doc:
    matches = page.search_for(keyword)

    if matches:
        found = True

    for match in matches:
        highlight = page.add_highlight_annot(match)
        highlight.update()

if found:
    doc.save("highlighted_output.pdf")
    print(f"'{keyword}' highlighted successfully!")
else:
    print("Keyword not found.")

doc.close()