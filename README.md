# Seq2Seq_ChatBot
공부용으로 만든 Seq2Seq 챗봇 모듈입니다.  
아직 코드를 직접 짜진 못해서 다른 분의 코드를 수정해서 만들었습니다.  
기본으로 들어있는 데이터셋도 다른 분이 공개한 데이터셋 입니다.  
혹시 사용하시거나 참고하실 분 있을까봐 docstring 하고 Example 만들어 놨습니다.  
파이썬 3.6 이상이 필요합니다.   
(버그나 추가할만한 점 이슈로 남겨주시면 할 수 있는 선에서 해보겠습니다)

## Required modules
* tensorflow>=2.0  
* numpy  
* pandas


# Examples
* 챗봇 학습 및 저장
```
import Seq2Seq_ChatBot.bot as chatbot
import Seq2Seq_ChatBot.function as func

# GPU 설정
# GPU 메모리의 사용량을 4096MB로 제한
# 사용할 GPU를 GPU 0 으로 설정
func.set_gpu(memory_limit=4096, gpu_number=0)

# 챗봇 생성
# 데이터셋은 기본 데이터셋을 사용
# 처음부터 1000번째 라인까지만 사용
bot = chatbot.ChatBot(dataset_path=["./Seq2Seq_ChatBot/dataset/ChatbotData.csv"], data_start=[0], data_end=[1000])

# 챗봇 학습
# 100 * 20, 총 2000번 학습
bot.repeat_train(epochs=100, steps=20)

# 챗봇 저장
# ./training 경로에 저장
bot.save("./training")

while 1:
    # 챗봇 테스트
    # 학습한 내용을 바탕으로 대답 예측
    print(bot.predict(input("input : ")))
```
* 챗봇 불러오기
```
import Seq2Seq_ChatBot.bot as chatbot
import Seq2Seq_ChatBot.function as func

func.set_gpu(memory_limit=4096, gpu_number=0)

bot = chatbot.ChatBot(dataset_path=["./Seq2Seq_ChatBot/dataset/ChatbotData.csv"], data_start=[0], data_end=[1000])

# 챗봇 불러오기
# ./training 경로에 저장된 모델을 불러옴
bot.load("./training")

while 1:
    print(bot.predict(input("input : ")))
```

## Links
[**원본 코드**](https://github.com/deepseasw/seq2seq_chatbot)  
[**한글 데이터셋**](https://github.com/songys/Chatbot_data)
