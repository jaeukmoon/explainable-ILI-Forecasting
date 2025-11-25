import PyPDF2

def extract_text_from_pdf(pdf_file_path): # pdf_file_path: PDF 파일의 경로입니다.
  # PDF 파일에서 텍스트를 추출하는 함수입니다.
  try:
    with open(pdf_file_path, 'rb') as pdf_file:

      pdf_reader = PyPDF2.PdfReader(pdf_file)
      text = ""

      for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()

      return text
  except FileNotFoundError:
    return "파일을 찾을 수 없습니다."
  except PyPDF2.utils.PdfReadError:
    return "PDF 파일을 읽을 수 없습니다."

'''
년도별로 파일주차를 자동계산이 아닌
직접 확인후 입력해주는 방식으로 만들었습니다.

예를 들어 23년도 40주차 부터 45주차 까지 변환한다면
1) year=2023, start=40, end=45
이렇게 입력하면 됩니다.

또 다른 예시로 22년도 30주차 부터 23년도 10주차 까지 변환은
1) year=2022, start=30, end=52
2) year=2023, start=1, end=10
이렇게 코드 2번 돌리면 됩니다.
'''
if __name__ == "__main__":
    year=2019
    start=40
    end=52
    for i in range(start,end+1):
        week="{:02d}".format(i)
        pdf_file_path = f"C:/Users/Korea/Desktop/jung/day3/cdc_pdf_2019/cdc_{year}{week}.pdf"
        extracted_text = extract_text_from_pdf(pdf_file_path)
        
        with open(f'cdc_{year}{week}.txt','w',encoding='utf-8') as file:
            file.write(extracted_text)
        print(f'{year}-{week}주차 저장')