# 분석

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

#전체 % 각 행위자 별 담화 분석
INPUT_FILE_NAME_corpus='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_cleaned/corpus_cleaned_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_actor1='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_cleaned/corpus_of_actor1_cleaned_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_actor2='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_cleaned/corpus_of_actor2_cleaned_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_actor3='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_cleaned/corpus_of_actor3_cleaned_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_actor4='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_cleaned/corpus_of_actor4_cleaned_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_actor5='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_cleaned/corpus_of_actor5_cleaned_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_actor6='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_cleaned/corpus_of_actor6_cleaned_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_actor7='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_cleaned/corpus_of_actor7_cleaned_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_actor8='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_cleaned/corpus_of_actor8_cleaned_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_actor9='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_cleaned/corpus_of_actor9_cleaned_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_actor10='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_cleaned/corpus_of_actor10_cleaned_2001.01.11.txt'

OUTPUT_FILE_NAME_corpus='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_analyse/corpus_analysed_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_actor1='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_analyse/corpus_of_actor1_analysed_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_actor2='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_analyse/corpus_of_actor2_analysed_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_actor3='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_analyse/corpus_of_actor3_analysed_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_actor4='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_analyse/corpus_of_actor4_analysed_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_actor5='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_analyse/corpus_of_actor5_analysed_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_actor6='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_analyse/corpus_of_actor6_analysed_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_actor7='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_analyse/corpus_of_actor7_analysed_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_actor8='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_analyse/corpus_of_actor8_analysed_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_actor9='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_analyse/corpus_of_actor9_analysed_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_actor10='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_analyse/corpus_of_actor10_analysed_2001.01.11.txt'




#최빈값
def mode(gtext,ntags=10): ##noun_count->nrags
    twitter=Twitter()
    NOUNS=twitter.nouns(gtext) #gtext에서 nouns()명사만 추출
#    NOUNS2=str(gtext) #{첫 글자 [, 어떻게 삭제? 작업 중}
#    NOUNS_splited2=NOUNS2.split(' ')
    cnt=Counter(NOUNS)    
    
    corpus_mode=[]
    for word, cnt in cnt.most_common(ntags): #count.most_common(ntags)
        temp={'key':word,'value':cnt}
#        print(temp)    
        corpus_mode.append(temp)        
    return corpus_mode


def main():
    #전체 담화
    noun_count=10 #noun_count 최빈값 중 상위 몇개 추출  
    text_file_name=INPUT_FILE_NAME_corpus     
    output_file_name=OUTPUT_FILE_NAME_corpus  
    
    open_text_file=open(text_file_name,"r")
    text=open_text_file.read() #text->gtext
    res=mode(text, noun_count)
    open_text_file.close()

    open_output_file=open(output_file_name,"w")
    for data in res :
        nouns=data['key']
        count=data['value']
        open_output_file.write("{} {}\n".format(nouns, count)) #"{} {}\n"이 문자열에 대해 (nouns,count)로 형식(format)을 만들라. 즉 nouns->첫번째{}, count->두번째{}에 넣는다

    #행위자1 담화
    noun_count=10 #noun_count 최빈값 중 상위 몇개 추출
    
    text_file_name1=INPUT_FILE_NAME_corpus_of_actor1     
    output_file_name1=OUTPUT_FILE_NAME_corpus_of_actor1
    
    open_text_file=open(text_file_name1,"r")
    text=open_text_file.read() #text->gtext
    res=mode(text, noun_count)
    open_text_file.close()

    open_output_file=open(output_file_name1,"w")
    for data in res :
        nouns=data['key']
        count=data['value']
        open_output_file.write("{} {}\n".format(nouns, count)) #"{} {}\n"이 문자열에 대해 (nouns,count)로 형식(format)을 만들라. 즉 nouns->첫번째{}, count->두번째{}에 넣는다
    
    #행위자2 담화
    noun_count=10 #noun_count 최빈값 중 상위 몇개 추출
    
    text_file_name2=INPUT_FILE_NAME_corpus_of_actor2     
    output_file_name2=OUTPUT_FILE_NAME_corpus_of_actor2
    
    open_text_file=open(text_file_name2,"r")
    text=open_text_file.read() #text->gtext
    res=mode(text, noun_count)
    open_text_file.close()

    open_output_file=open(output_file_name2,"w")
    for data in res :
        nouns=data['key']
        count=data['value']
        open_output_file.write("{} {}\n".format(nouns, count)) #"{} {}\n"이 문자열에 대해 (nouns,count)로 형식(format)을 만들라. 즉 nouns->첫번째{}, count->두번째{}에 넣는다
    
    #행위자3 담화
    noun_count=10 #noun_count 최빈값 중 상위 몇개 추출
    
    text_file_name3=INPUT_FILE_NAME_corpus_of_actor3     
    output_file_name3=OUTPUT_FILE_NAME_corpus_of_actor3
    
    open_text_file=open(text_file_name3,"r")
    text=open_text_file.read() #text->gtext
    res=mode(text, noun_count)
    open_text_file.close()

    open_output_file=open(output_file_name3,"w")
    for data in res :
        nouns=data['key']
        count=data['value']
        open_output_file.write("{} {}\n".format(nouns, count)) #"{} {}\n"이 문자열에 대해 (nouns,count)로 형식(format)을 만들라. 즉 nouns->첫번째{}, count->두번째{}에 넣는다

    #행위자4 담화
    noun_count=10 #noun_count 최빈값 중 상위 몇개 추출
    
    text_file_name4=INPUT_FILE_NAME_corpus_of_actor4     
    output_file_name4=OUTPUT_FILE_NAME_corpus_of_actor4
    
    open_text_file=open(text_file_name4,"r")
    text=open_text_file.read() #text->gtext
    res=mode(text, noun_count)
    open_text_file.close()

    open_output_file=open(output_file_name4,"w")
    for data in res :
        nouns=data['key']
        count=data['value']
        open_output_file.write("{} {}\n".format(nouns, count)) #"{} {}\n"이 문자열에 대해 (nouns,count)로 형식(format)을 만들라. 즉 nouns->첫번째{}, count->두번째{}에 넣는다

    #행위자5 담화
    noun_count=10 #noun_count 최빈값 중 상위 몇개 추출
    
    text_file_name5=INPUT_FILE_NAME_corpus_of_actor5     
    output_file_name5=OUTPUT_FILE_NAME_corpus_of_actor5
    
    open_text_file=open(text_file_name5,"r")
    text=open_text_file.read() #text->gtext
    res=mode(text, noun_count)
    open_text_file.close()

    open_output_file=open(output_file_name5,"w")
    for data in res :
        nouns=data['key']
        count=data['value']
        open_output_file.write("{} {}\n".format(nouns, count)) #"{} {}\n"이 문자열에 대해 (nouns,count)로 형식(format)을 만들라. 즉 nouns->첫번째{}, count->두번째{}에 넣는다

    #행위자6 담화
    noun_count=10 #noun_count 최빈값 중 상위 몇개 추출
    
    text_file_name6=INPUT_FILE_NAME_corpus_of_actor6     
    output_file_name6=OUTPUT_FILE_NAME_corpus_of_actor6
    
    open_text_file=open(text_file_name6,"r")
    text=open_text_file.read() #text->gtext
    res=mode(text, noun_count)
    open_text_file.close()

    open_output_file=open(output_file_name6,"w")
    for data in res :
        nouns=data['key']
        count=data['value']
        open_output_file.write("{} {}\n".format(nouns, count)) #"{} {}\n"이 문자열에 대해 (nouns,count)로 형식(format)을 만들라. 즉 nouns->첫번째{}, count->두번째{}에 넣는다

    #행위자7 담화
    noun_count=10 #noun_count 최빈값 중 상위 몇개 추출
    
    text_file_name7=INPUT_FILE_NAME_corpus_of_actor7     
    output_file_name7=OUTPUT_FILE_NAME_corpus_of_actor7
    
    open_text_file=open(text_file_name7,"r")
    text=open_text_file.read() #text->gtext
    res=mode(text, noun_count)
    open_text_file.close()

    open_output_file=open(output_file_name7,"w")
    for data in res :
        nouns=data['key']
        count=data['value']
        open_output_file.write("{} {}\n".format(nouns, count)) #"{} {}\n"이 문자열에 대해 (nouns,count)로 형식(format)을 만들라. 즉 nouns->첫번째{}, count->두번째{}에 넣는다

    #행위자8 담화
    noun_count=10 #noun_count 최빈값 중 상위 몇개 추출
    
    text_file_name8=INPUT_FILE_NAME_corpus_of_actor8     
    output_file_name8=OUTPUT_FILE_NAME_corpus_of_actor8
    
    open_text_file=open(text_file_name8,"r")
    text=open_text_file.read() #text->gtext
    res=mode(text, noun_count)
    open_text_file.close()

    open_output_file=open(output_file_name8,"w")
    for data in res :
        nouns=data['key']
        count=data['value']
        open_output_file.write("{} {}\n".format(nouns, count)) #"{} {}\n"이 문자열에 대해 (nouns,count)로 형식(format)을 만들라. 즉 nouns->첫번째{}, count->두번째{}에 넣는다

    #행위자9 담화
    noun_count=10 #noun_count 최빈값 중 상위 몇개 추출
    
    text_file_name9=INPUT_FILE_NAME_corpus_of_actor9   
    output_file_name9=OUTPUT_FILE_NAME_corpus_of_actor9
    
    open_text_file=open(text_file_name9,"r")
    text=open_text_file.read() #text->gtext
    res=mode(text, noun_count)
    open_text_file.close()

    open_output_file=open(output_file_name9,"w")
    for data in res :
        nouns=data['key']
        count=data['value']
        open_output_file.write("{} {}\n".format(nouns, count)) #"{} {}\n"이 문자열에 대해 (nouns,count)로 형식(format)을 만들라. 즉 nouns->첫번째{}, count->두번째{}에 넣는다

    #행위자10 담화
    noun_count=10 #noun_count 최빈값 중 상위 몇개 추출
    
    text_file_name10=INPUT_FILE_NAME_corpus_of_actor10     
    output_file_name10=OUTPUT_FILE_NAME_corpus_of_actor10
    
    open_text_file=open(text_file_name10,"r")
    text=open_text_file.read() #text->gtext
    res=mode(text, noun_count)
    open_text_file.close()

    open_output_file=open(output_file_name10,"w")
    for data in res :
        nouns=data['key']
        count=data['value']
        open_output_file.write("{} {}\n".format(nouns, count)) #"{} {}\n"이 문자열에 대해 (nouns,count)로 형식(format)을 만들라. 즉 nouns->첫번째{}, count->두번째{}에 넣는다



if __name__=="__main__":
    main()