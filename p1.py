
import streamlit as st
import PyPDF2
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
import tempfile

def add_page_numbers_to_pdf(input_pdf_path, output_pdf_path):
    """Add sequential page numbers to the bottom-right corner of each page."""
    pdf_reader = PyPDF2.PdfReader(input_pdf_path)
    pdf_writer = PyPDF2.PdfWriter()
    
    for page_num, page in enumerate(pdf_reader.pages):
        # Create a new PDF with just the page number
        packet = BytesIO()
        can = canvas.Canvas(packet, pagesize=letter)
        can.setFont("Helvetica", 10)
        can.drawRightString(550, 30, f"Page {page_num + 1}")
        can.save()
        packet.seek(0)
        
        # Read the page number PDF
        number_pdf = PyPDF2.PdfReader(packet)
        
        # Merge the page number onto the original page
        page.merge_page(number_pdf.pages[0])
        pdf_writer.add_page(page)
    
    # Write the result
    with open(output_pdf_path, 'wb') as output_file:
        pdf_writer.write(output_file)

def merge_pdfs(uploaded_files, output_file):
    """Merge uploaded PDFs and add page numbers."""
    if not uploaded_files:
        st.error("No PDF files uploaded.")
        return None
    
    # First, merge all PDFs without page numbers
    pdf_writer = PyPDF2.PdfWriter()
    
    # Process each uploaded PDF
    for uploaded_file in uploaded_files:
        try:
            # Reset file pointer to beginning
            uploaded_file.seek(0)
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                pdf_writer.add_page(page)
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            return None
    
    # Write merged PDF to temporary file first
    temp_merged_file = output_file + "_temp"
    try:
        with open(temp_merged_file, 'wb') as temp_output:
            pdf_writer.write(temp_output)
    except Exception as e:
        st.error(f"Error writing temporary file: {str(e)}")
        return None
    
    # Now add page numbers to the merged PDF
    try:
        add_page_numbers_to_pdf(temp_merged_file, output_file)
        # Clean up temporary file
        os.unlink(temp_merged_file)
        return output_file
    except Exception as e:
        st.error(f"Error adding page numbers: {str(e)}")
        # Clean up temporary file if it exists
        if os.path.exists(temp_merged_file):
            os.unlink(temp_merged_file)
        return None

def main():
    st.title("PDF Merger with Page Numbering")
    st.write("Upload PDF files to merge them into a single PDF with sequential page numbers in the bottom-right corner.")
    
    # File uploader for multiple PDFs
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        st.write(f"{len(uploaded_files)} PDF(s) uploaded. Click below to merge.")
        
        if st.button("Merge PDFs"):
            with st.spinner("Merging PDFs..."):
                # Use a temporary file for the output
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    output_file = tmp_file.name
                    result = merge_pdfs(uploaded_files, output_file)
                
                if result:
                    st.success("PDFs merged successfully!")
                    # Read the file content into memory
                    with open(result, "rb") as file:
                        pdf_data = file.read()
                    
                    st.download_button(
                        label="Download Merged PDF",
                        data=pdf_data,
                        file_name="merged_output.pdf",
                        mime="application/pdf"
                    )
                    # Clean up temporary file
                    os.unlink(result)
    else:
        st.info("Please upload at least one PDF file to proceed.")

if __name__ == "__main__":
    main()