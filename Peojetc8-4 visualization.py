# 상위 몇개에 해당하는 단어를 시각화

#막대그래프 그리기
from matplotlib import pyplot as plt #matplotlib:화면에 시각화
import matplotlib
from matplotlib import font_manager, rc
import platform

INPUT_FILE_NAME_corpus='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_analyse/corpus_analysed_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_actor1='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_analyse/corpus_of_actor1_analysed_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_actor2='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_analyse/corpus_of_actor2_analysed_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_actor3='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_analyse/corpus_of_actor3_analysed_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_actor4='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_analyse/corpus_of_actor4_analysed_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_actor5='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_analyse/corpus_of_actor5_analysed_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_actor6='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_analyse/corpus_of_actor6_analysed_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_actor7='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_analyse/corpus_of_actor7_analysed_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_actor8='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_analyse/corpus_of_actor8_analysed_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_actor9='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_analyse/corpus_of_actor9_analysed_2001.01.11.txt'
INPUT_FILE_NAME_corpus_of_actor10='D:\Python\MyGitHub\Project8-DL-NLP-DEBATE\OutputData\corpus_analyse/corpus_of_actor10_analysed_2001.01.11.txt'

#시각화 시 한글이 깨지는 문제(영어는 제대로 출력)를 해결해 줌. 아래 4문장은 늘 씀.
if platform.system()=="Windows": #현재 시스템이 윈도우 운영 체제라면,
    font_name=font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name() #ttf는 윈도우에 쓰이는 '글자'들이 모여있는 파일임.
    rc('font',family=font_name) #이 문장은 늘 사용
matplotlib.rcParams['axes.unicode_minus']=False

      
#  
f=open(INPUT_FILE_NAME_corpus_of_actor10,"r")
#    print(f)
i=1
news_word=[] #x축-단어
word_cnt=[] #y축-빈도수
while True:
    line=f.readline()
    word, count=line.split(" ") #word('날씨'가리킴)와  count(5가리킴)를 split로 공백 나누기
    news_word.append(word) #news_word에 word추가
    word_cnt.append(int(count[0])) #int(count[0]):순수하게 숫자만 나옴 #빈도수
    if i==10 : break
    i+=1
f.close()
print(news_word)
print(word_cnt)
      

#그래프 꾸미기
xs=[i+0.1 for i, _ in enumerate(news_word)] #enumerate:열거형. news_word의 단어와 인덱스(위치)가 리턴되어짐. ex)날씨-0번째->i에 들어감.
plt.bar(xs,word_cnt) #bar:막대그래프 출력
plt.ylabel('등장 단어의 수') #ylabel:y축에 해당하는 값 쓰기
plt.title('조동근, 시청자(신촌)')
plt.xticks([i for i, _ in enumerate(news_word)],news_word) #xticks : x축
# plt.xticks([i+0.5 for i, _ in enumerate(news_word)],news_word) #xticks : x축의 위치가0.5(i+0.5)만큼 이동
plt.show()

print("<시각화 완료>")

