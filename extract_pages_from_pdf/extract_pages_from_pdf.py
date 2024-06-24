import PyPDF2

def extract_pages_from_pdf(input_pdf_path, output_pdf_path, start_page, end_page):
    try:
        # 원본 PDF 파일 열기
        with open(input_pdf_path, "rb") as input_pdf_file:
            reader = PyPDF2.PdfReader(input_pdf_file)
            writer = PyPDF2.PdfWriter()

            # 페이지 범위 검사
            total_pages = len(reader.pages)
            if start_page < 1 or end_page > total_pages or start_page > end_page:
                raise ValueError("Invalid page range specified.")

            # 페이지 추출 (페이지 번호는 0부터 시작)
            for page_number in range(start_page - 1, end_page):
                page = reader.pages[page_number]
                writer.add_page(page)

            # 새로운 PDF 파일로 저장
            with open(output_pdf_path, "wb") as output_pdf_file:
                writer.write(output_pdf_file)

        print(f"Pages {start_page} to {end_page} successfully extracted to {output_pdf_path}")

    except FileNotFoundError:
        print(f"File not found: {input_pdf_path}")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# 예제 사용법
input_pdf_path = "example.pdf"  # 원본 PDF 파일 경로
output_pdf_path = "extracted_pages.pdf"  # 추출된 페이지들을 저장할 새로운 PDF 파일 경로
start_page = 2  # 추출할 시작 페이지 번호 (예: 2번째 페이지)
end_page = 5  # 추출할 끝 페이지 번호 (예: 5번째 페이지)

extract_pages_from_pdf(input_pdf_path, output_pdf_path, start_page, end_page)
