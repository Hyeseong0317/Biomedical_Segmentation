# 0.jpeg, 1.jpeg. 2.jpeg, 3.jepg -> .을 기준으로 split후 앞의 숫자를 0으로 인덱싱하여 가져온다
import os
for i in sorted(os.listdir("C:/intern/"), key=lambda x: int(x.split(".")[0])):
    print(i)

  
