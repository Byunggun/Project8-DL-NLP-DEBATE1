# 목표
# 1. 파일 읽기
# 2. 분석하고 싶은 내용만 추출해 텍스트 파일로 저장

from konlpy.tag import Twitter
import codecs
from bs4 import BeautifulSoup
from gensim.models import word2vec
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager,rc
import platform
from collections import Counter



#신문 개혁
OUTPUT_FILE_NAME_corpus='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus/corpus_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_head='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus/corpus_of_head_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_personalinformation='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus/corpus_of_personalinformation_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_actor1='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus/corpus_of_actor1_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_actor2='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus/corpus_of_actor2_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_actor3='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus/corpus_of_actor3_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_actor4='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus/corpus_of_actor4_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_actor5='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus/corpus_of_actor5_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_actor6='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus/corpus_of_actor6_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_actor7='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus/corpus_of_actor7_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_actor8='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus/corpus_of_actor8_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_actor9='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus/corpus_of_actor9_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_actor10='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus/corpus_of_actor10_2001.01.11.txt'


#1. 파일 읽기
fp=codecs.open("D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\RawData/4BK01001.txt","r",encoding="utf-16")#encoding방식5은 utf-16,utf-8,ms949,euc-kr 중 선택 #MS949 : 한글 확장 완성형 (똠방각하 표현 가능)
soup=BeautifulSoup(fp,"html.parser")
#print(soup)


# 태그 처리
def tag():    
    corpus=[] #모든 행위자의 담화 추출 for 무엇, 어떻게 말하는가
    corpus_of_head=[] #제목 추출   
    corpus_of_personalinformation=[] #모든 행위자 인적사항 리스트 추출 for 누가 말하는가     
    corpus_of_actor1=[] #행위자1(사회자) 담화 추출 
    corpus_of_actor2=[] #행위자2 담화 추출
    corpus_of_actor3=[] #행위자3 담화 추출
    corpus_of_actor4=[] #행위자4 담화 추출
    corpus_of_actor5=[] #행위자5 담화 추출
    corpus_of_actor6=[] #행위자6 담화 추출
    corpus_of_actor7=[] #행위자7 담화 추출
    corpus_of_actor8=[] #행위자8 담화 추출 
    corpus_of_actor9=[] #행위자9 담화 추출
    corpus_of_actor10=[] #행위자10 담화 추출
    
    #모든 행위자의 담화 추출
    for body in soup.find_all("body"): 
#        print(body)
        corpus=body.getText() #getText() : body에서 텍스트 뽑아내기              
    #제목 추출   
    for head in soup.find_all("head"): 
#        print(head)
        corpus_of_head=head.getText() 
    #모든 행위자 인적사항
    for actors in soup.find_all("particdesc"): 
#        print(actors)
        corpus_of_personalinformation=actors.getText()   
    #행위자1(사회자) 담화 추출
    for discours1 in soup.find_all("u",who="P1"):        
#        print(discours1)    
        corpus_of_actor1.append(discours1.getText()) 
    #행위자2 담화 추출
    for discours2 in soup.find_all("u",who="P2"):        
#        print(discours2)    
        corpus_of_actor2.append(discours2.getText())
    #행위자3 담화 추출
    for discours3 in soup.find_all("u",who="P3"):        
#        print(discours3)    
        corpus_of_actor3.append(discours3.getText())
    #행위자4 담화 추출
    for discours4 in soup.find_all("u",who="P4"):        
#        print(discours4)    
        corpus_of_actor4.append(discours4.getText())        
    #행위자5 담화 추출
    for discours5 in soup.find_all("u",who="P5"):        
#        print(discours5)    
        corpus_of_actor5.append(discours5.getText())
    #행위자6 담화 추출
    for discours6 in soup.find_all("u",who="P6"):        
#        print(discours6)    
        corpus_of_actor6.append(discours6.getText())
    #행위자7 담화 추출
    for discours7 in soup.find_all("u",who="P7"):        
#        print(discours7)    
        corpus_of_actor7.append(discours7.getText())
    #행위자8 담화 추출
    for discours8 in soup.find_all("u",who="P8"):        
#        print(discours8)    
        corpus_of_actor8.append(discours8.getText())
    #행위자9 담화 추출
    for discours9 in soup.find_all("u",who="P9"):        
#        print(discours9)    
        corpus_of_actor9.append(discours9.getText())
    #행위자10 담화 추출
    for discours10 in soup.find_all("u",who="P10"):        
#        print(discours10)    
        corpus_of_actor10.append(discours10.getText())    
    return corpus, corpus_of_head,corpus_of_personalinformation, corpus_of_actor1, corpus_of_actor2,corpus_of_actor3,corpus_of_actor4,corpus_of_actor5,corpus_of_actor6,corpus_of_actor7,corpus_of_actor8,corpus_of_actor9,corpus_of_actor10

corpus, corpus_of_head,corpus_of_personalinformation, corpus_of_actor1, corpus_of_actor2,corpus_of_actor3,corpus_of_actor4,corpus_of_actor5,corpus_of_actor6,corpus_of_actor7,corpus_of_actor8,corpus_of_actor9,corpus_of_actor10=tag()

#작업 중 : 리스트->str 변환
corpus_of_actor1=" ".join(corpus_of_actor1)
corpus_of_actor2=" ".join(corpus_of_actor2)
corpus_of_actor3=" ".join(corpus_of_actor3)
corpus_of_actor4=" ".join(corpus_of_actor4)
corpus_of_actor5=" ".join(corpus_of_actor5)
corpus_of_actor6=" ".join(corpus_of_actor6)
corpus_of_actor7=" ".join(corpus_of_actor7)
corpus_of_actor8=" ".join(corpus_of_actor8)
corpus_of_actor9=" ".join(corpus_of_actor9)
corpus_of_actor10=" ".join(corpus_of_actor10)


#2. 메인함수
#프로그램의 시작과 끝은 main함수로 시작 끝남.
def main():    
    open_output_file=open(OUTPUT_FILE_NAME_corpus, "w")
    open_output_file.write(corpus)
    open_output_file.close()

    open_output_file=open(OUTPUT_FILE_NAME_corpus_of_head, "w")
    open_output_file.write(corpus_of_head) #url->text
    open_output_file.close()
    
    open_output_file=open(OUTPUT_FILE_NAME_corpus_of_personalinformation, "w")
    open_output_file.write(corpus_of_personalinformation) #url->text
    open_output_file.close()

    open_output_file=open(OUTPUT_FILE_NAME_corpus_of_actor1, "w")
    open_output_file.write(corpus_of_actor1) #url->text
    open_output_file.close()

    open_output_file=open(OUTPUT_FILE_NAME_corpus_of_actor2, "w")
    open_output_file.write(corpus_of_actor2) #url->text
    open_output_file.close()

    open_output_file=open(OUTPUT_FILE_NAME_corpus_of_actor3, "w")
    open_output_file.write(corpus_of_actor3) #url->text
    open_output_file.close()

    open_output_file=open(OUTPUT_FILE_NAME_corpus_of_actor4, "w")
    open_output_file.write(corpus_of_actor4) #url->text
    open_output_file.close()

    open_output_file=open(OUTPUT_FILE_NAME_corpus_of_actor5, "w")
    open_output_file.write(corpus_of_actor5) #url->text
    open_output_file.close()

    open_output_file=open(OUTPUT_FILE_NAME_corpus_of_actor6, "w")
    open_output_file.write(corpus_of_actor6) #url->text
    open_output_file.close()

    open_output_file=open(OUTPUT_FILE_NAME_corpus_of_actor7, "w")
    open_output_file.write(corpus_of_actor7) #url->text
    open_output_file.close()

    open_output_file=open(OUTPUT_FILE_NAME_corpus_of_actor8, "w")
    open_output_file.write(corpus_of_actor8) #url->text
    open_output_file.close()

    open_output_file=open(OUTPUT_FILE_NAME_corpus_of_actor9, "w")
    open_output_file.write(corpus_of_actor9) #url->text
    open_output_file.close()

    open_output_file=open(OUTPUT_FILE_NAME_corpus_of_actor10, "w")
    open_output_file.write(corpus_of_actor10) #url->text
    open_output_file.close()

# #함수 호출 부분:아래 main()->def main()을 호출함
# 8-2에서 import 하게 되면,__name__에 8-2가 들어감(cf.module1)
# 아래는 함수가 아니라 파이썬 문장임
if __name__=='__main__':
    main() #main함수를 호출해라
    
