import os
from PyPDF2 import PdfMerger


def main(args):

    directory_name, path_output = args.dir_name, args.path_output

    # 지정한 디렉토리 내의 pdf 파일들만 리스트에 저장
    pdfs = [f for f in os.listdir(directory_name) if f.endswith('.pdf')]

    # 알파벳 순으로 정렬
    pdfs.sort()

    merger = PdfMerger()

    # 정렬된 pdf 파일들을 순차적으로 병합
    for pdf in pdfs:
        merger.append(os.path.join(directory_name, pdf))

    # 병합된 결과 저장
    merger.write(path_output)
    merger.close()


import argparse


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='dir_name', type=str, default='.')
    parser.add_argument('-o', dest='path_output', type=str, default='results.pdf')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(get_options())
