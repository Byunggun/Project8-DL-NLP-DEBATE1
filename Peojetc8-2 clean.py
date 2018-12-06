

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
#import nltk
#from nltk.corpus import stopwords 
#from nltk.tokenize import word_tokenize 



INPUT_FILE_NAME_corpus='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus/corpus_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_head='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus/corpus_of_head_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_personalinformation='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus/corpus_of_personalinformation_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_actor1='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus/corpus_of_actor1_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_actor2='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus/corpus_of_actor2_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_actor3='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus/corpus_of_actor3_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_actor4='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus/corpus_of_actor4_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_actor5='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus/corpus_of_actor5_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_actor6='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus/corpus_of_actor6_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_actor7='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus/corpus_of_actor7_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_actor8='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus/corpus_of_actor8_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_actor9='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus/corpus_of_actor9_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_actor10='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus/corpus_of_actor10_2001.01.11.txt'

OUTPUT_FILE_NAME_corpus='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_cleaned/corpus_cleaned_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_head='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_cleaned/corpus_of_head_cleaned_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_personalinformation='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_cleaned/corpus_of_personalinformation_cleaned_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_actor1='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_cleaned/corpus_of_actor1_cleaned_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_actor2='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_cleaned/corpus_of_actor2_cleaned_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_actor3='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_cleaned/corpus_of_actor3_cleaned_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_actor4='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_cleaned/corpus_of_actor4_cleaned_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_actor5='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_cleaned/corpus_of_actor5_cleaned_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_actor6='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_cleaned/corpus_of_actor6_cleaned_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_actor7='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_cleaned/corpus_of_actor7_cleaned_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_actor8='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_cleaned/corpus_of_actor8_cleaned_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_actor9='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_cleaned/corpus_of_actor9_cleaned_2001.01.11.txt'
OUTPUT_FILE_NAME_corpus_of_actor10='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_cleaned/corpus_of_actor10_cleaned_2001.01.11.txt'


###############################################################################
#2. 전처리
##2){핵심}형태소 분석 : 명사만 추출
def morpheme_analysed(text):
    twitter=Twitter()
    lines=text.split("\r\n") #\n :다음줄로 넘어가라. \r:맨 앞으로 가라.                #\r\n로 구분 안될 때, \n\r도 사용해봐라.
#    print(lines) 
#    print(len(lines)) #한꺼번에 출력, 총 몇 줄인지 확인
    
    results_morpheme_analysed=[] #명사 리스트
    for line in lines:
        malist=twitter.pos(line, norm=True, stem=True)                              ##norm은 현대적인말 그래욬ㅋㅋㅋ 같은 것을 그래요로 바꾸어 주는 것이다.#stem은 그래요를 그렇다처럼 원형으로 바꾸어 주는 것이다.
#        print("malist",malist)
        res=[] #명사 리스트에 담기
        for word in malist:
            if not word[1] in ["Verb", "Adjective", "Adverb", "Determiner", 
                       "Conjunction", "Exclamation", "Josa", "Eomi", "PreEomi", 
                       "Suffix", "Punctuation", "Foreign", "Alpha", "Number", 
                       "Unknown", "KoreanParticle", "Hashtag", "ScreenName", 
                       "Email", "URL"]:#명사만 추출
                res.append(word[0])
#                print('res',res)
        r1=(" ".join(res))
        results_morpheme_analysed.append(r1)  
#    print('results_morpheme_analysed',results_morpheme_analysed)
    return results_morpheme_analysed
#a=morpheme_analysed(text) #형태소 분석 : 명사만 추출

#print("<형태소 분석 후>", morpheme_analysed(myText))


#3)불용어 제거(stop words elimination)
#3-1) 불용어(불필요한 단어) 제거 : remove()
def remove_stop_words(text_ma0): 
    #1 글자만 추출->중복 단어 제거(set())->의미 없는 것 임의 삭제(주간성 개입!)(ex.'몫', '별', '술', '법', '답'...)->stop_words에 넣기
    oneword=[]
    ss = text_ma0[0].split(' ')        
    for word in ss:
        if len(word) == 1:        
            oneword.append(word)
#    print("<1글자>",oneword)
    a=set(oneword)
    print(a)
            
    #1차 정제
    #stop_words에 지우고 싶은 단어 기입 시 삭제됨  
    stop_words=['것', '이것','저것','그것','것인지','것으로','거', '보', '헌', '본', '뒤', '빼', '주', '근', '고', '사', '기', '쪽', '날', '바', '남', '면', '경', '편', '식', '하', '오', '자', '한', '쭉', '내', '전', '게', '구', '여', '후', '과', '언', '족', '의', '랄', '명', '어', '줄', '살', '다', '꼭', '잡', '년', '당', '못', '올', '이', '컷', '치', '준', '일', '모', '첫', '잘', '볼', '연', '세', '두', '막', '니', '설', '네', '냉', '상', '끝', '때', '양', '도', '몇', '타', '핵', '예', '울', '뭡', '늘', '더', '겁', '테', '나', '점', '종', '번', '합', '선', '시', '난', '강', '온', '위', '리', '뿐', '로', '분', '알', '현', '등', '눈', '놈', '를', '서', '백', '문', '드', '그', '신', '지', '정', '공', '비', '원', '료', '입', '컨', '돈', '피', '요', '외', '동', '김', '친', '대', '약', '협', '칼', '석', '흠', '킥', '채', '왜', '홍', '뭐', '은', '함', '각', '저', '데', '제', '업', '적', '수', '무', '학', '말', '인', '새', '건', '안', '심', '달', '조', '망', '또', '반', '걸', '및', '회', '좀', '대해', '가지', '대한','먼저','부분']

    results_stop_words=[]  
    
    tmp = text_ma0[0].split(' ')
    tmp_remove = tmp.copy()
          
    for word in tmp:
#        print("<ma_text>",ma_text)
#        print("<tmp>",tmp)
        if word in stop_words:
            tmp_remove.remove(word) #제거하기
#        print("<불용어 제거 후-tmp>",tmp)
    results_stop_words = " ".join(tmp_remove) #1나의 리스트에 담기
#    print("<불용어 제거 후>", results_stop_words)
    return results_stop_words
    
#    #2차 정제
#    regex3=re.compile("것")
#    res3=regex3.findall(results_stop_words)
#    print("<res>",res3)
    

    
    
    
#    a=str(results_stop_words)
#    result_sub=[]
#    for i in a:        
#        cleaned_text=re.sub('[^것]', '', a)
#        result_sub.append(cleaned_text)
#    print("<result_sub>",cleaned_text)
#    return results_stop_words
#    

###############################################################################
#corpus_of_actor2=tag() #전처리 전 텍스트
#corpus_of_actor2_cleaned=remove_stop_words(corpus_of_actor2_cleaned_morpheme_analysed) #전처리 후 텍스트
###############################################################################

def main():
    read_file=open(INPUT_FILE_NAME_corpus, "r")
    write_file=open(OUTPUT_FILE_NAME_corpus, "w")
    text0=read_file.read()
#    print("before : ")
#    print(text0)
    text_ma0=morpheme_analysed(text0)
#    print("after 1: ")
#    print("text_ma0",text_ma0)
#    print(type(text_ma0))
    text_cleaned0=remove_stop_words(text_ma0)
#    print("after 2: ")
#    print("",text_cleaned0)
    cleaned_text0="".join(text_cleaned0) #
    write_file.write(cleaned_text0)
    read_file.close()
    write_file.close()


    read_file1=open(INPUT_FILE_NAME_corpus_of_head, "r")
    write_file1=open(OUTPUT_FILE_NAME_corpus_of_head, "w")
    text1=read_file1.read()
#    print("before : ")
#    print(text1)
    ma1=morpheme_analysed(text1)
    text_cleaned1=remove_stop_words(ma1)
#    print("after : ")
#    print(text_cleaned1)
    cleaned_text1="".join(text_cleaned1) #
    write_file1.write(cleaned_text1)
    read_file1.close()
    write_file1.close()
    
    
    read_file2=open(INPUT_FILE_NAME_corpus_of_personalinformation, "r")
    write_file2=open(OUTPUT_FILE_NAME_corpus_of_personalinformation, "w")    
    text2=read_file2.read()
#    print("before : ")
#    print(text2)
    ma2=morpheme_analysed(text2)
    text_cleaned2=remove_stop_words(ma2)
#    print("after : ")
#    print(text_cleaned2)
    cleaned_text2="".join(text_cleaned2) #
    write_file2.write(cleaned_text2)
    read_file2.close()
    write_file2.close()    


    read_file3=open(INPUT_FILE_NAME_corpus_of_actor1, "r")
    write_file3=open(OUTPUT_FILE_NAME_corpus_of_actor1, "w")    
    text3=read_file3.read()
#    print("before : ")
#    print(text3)
    ma3=morpheme_analysed(text3)
    text_cleaned3=remove_stop_words(ma3)
#    print("after : ")
#    print(text_cleaned3)
    cleaned_text3="".join(text_cleaned3) #
    write_file3.write(cleaned_text3)
    read_file3.close()
    write_file3.close()


    read_file4=open(INPUT_FILE_NAME_corpus_of_actor2, "r")
    write_file4=open(OUTPUT_FILE_NAME_corpus_of_actor2, "w")    
    text4=read_file4.read()
#    print("before : ")
#    print(text4)
    ma4=morpheme_analysed(text4)
    text_cleaned4=remove_stop_words(ma4)
#    print("after : ")
#    print(text_cleaned4)
    cleaned_text4="".join(text_cleaned4) #
    write_file4.write(cleaned_text4)
    read_file4.close()
    write_file4.close()
    
    
    read_file5=open(INPUT_FILE_NAME_corpus_of_actor3, "r")
    write_file5=open(OUTPUT_FILE_NAME_corpus_of_actor3, "w")    
    text5=read_file5.read()
#    print("before : ")
#    print(text5)
    ma5=morpheme_analysed(text5)
    text_cleaned5=remove_stop_words(ma5)
#    print("after : ")
#    print(text_cleaned5)
    cleaned_text5="".join(text_cleaned5) #
    write_file5.write(cleaned_text5)
    read_file5.close()
    write_file5.close()


    read_file6=open(INPUT_FILE_NAME_corpus_of_actor4, "r")
    write_file6=open(OUTPUT_FILE_NAME_corpus_of_actor4, "w")    
    text6=read_file6.read()
#    print("before : ")
#    print(text6)
    ma6=morpheme_analysed(text6)
    text_cleaned6=remove_stop_words(ma6)
#    print("after : ")
#    print(text_cleaned6)
    cleaned_text6="".join(text_cleaned6) #
    write_file6.write(cleaned_text6)
    read_file6.close()
    write_file6.close()
    

    read_file7=open(INPUT_FILE_NAME_corpus_of_actor5, "r")
    write_file7=open(OUTPUT_FILE_NAME_corpus_of_actor5, "w")    
    text7=read_file7.read()
#    print("before : ")
#    print(text7)
    ma7=morpheme_analysed(text7)
    text_cleaned7=remove_stop_words(ma7)
#    print("after : ")
#    print(text_cleaned7)
    cleaned_text7="".join(text_cleaned7) #
    write_file7.write(cleaned_text7)
    read_file7.close()
    write_file7.close()


    read_file8=open(INPUT_FILE_NAME_corpus_of_actor6, "r")
    write_file8=open(OUTPUT_FILE_NAME_corpus_of_actor6, "w")    
    text8=read_file8.read()
#    print("before : ")
#    print(text8)
    ma8=morpheme_analysed(text8)
    text_cleaned8=remove_stop_words(ma8)
#    print("after : ")
#    print(text_cleaned8)
    cleaned_text8="".join(text_cleaned8) #
    write_file8.write(cleaned_text8)
    read_file8.close()
    write_file8.close()


    read_file9=open(INPUT_FILE_NAME_corpus_of_actor7, "r")
    write_file9=open(OUTPUT_FILE_NAME_corpus_of_actor7, "w")    
    text9=read_file9.read()
#    print("before : ")
#    print(text9)
    ma9=morpheme_analysed(text9)
    text_cleaned9=remove_stop_words(ma9)
#    print("after : ")
#    print(text_cleaned9)
    cleaned_text9="".join(text_cleaned9) #
    write_file9.write(cleaned_text9)
    read_file9.close()
    write_file9.close()


    read_file10=open(INPUT_FILE_NAME_corpus_of_actor8, "r")
    write_file10=open(OUTPUT_FILE_NAME_corpus_of_actor8, "w")    
    text10=read_file10.read()
#    print("before : ")
#    print(text10)
    ma10=morpheme_analysed(text10)
    text_cleaned10=remove_stop_words(ma10)
#    print("after : ")
#    print(text_cleaned10)
    cleaned_text10="".join(text_cleaned10) #
    write_file10.write(cleaned_text10)
    read_file10.close()
    write_file10.close()   


    read_file11=open(INPUT_FILE_NAME_corpus_of_actor9, "r")
    write_file11=open(OUTPUT_FILE_NAME_corpus_of_actor9, "w")    
    text11=read_file11.read()
#    print("before : ")
#    print(text11)
    ma11=morpheme_analysed(text11)
    text_cleaned11=remove_stop_words(ma11)
#    print("after : ")
#    print(text_cleaned11)
    cleaned_text11="".join(text_cleaned11) #
    write_file11.write(cleaned_text11)
    read_file11.close()
    write_file11.close() 


    read_file12=open(INPUT_FILE_NAME_corpus_of_actor10, "r")
    write_file12=open(OUTPUT_FILE_NAME_corpus_of_actor10, "w")    
    text12=read_file12.read()
#    print("before : ")
#    print(text12)
    ma12=morpheme_analysed(text12)
    text_cleaned12=remove_stop_words(ma12)
#    print("after : ")
#    print(text_cleaned12)
    cleaned_text12="".join(text_cleaned12) #
    write_file12.write(cleaned_text12)
    read_file12.close()
    write_file12.close() 

if __name__ == "__main__":
    main()
    
    