for i in range(1200):
    if ((201<=(i+1)<=399) and (((i+1)-201)%2)) or ((402<=(i+1)<=598) and (((i+1)-402)%2)) or ((802<=(i+1)<=998) and (((i+1)-802)%2)):
        print(i)