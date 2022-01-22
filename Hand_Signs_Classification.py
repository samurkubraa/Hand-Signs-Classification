# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 20:35:58 2022

@author: Kübra
"""


import cv2
import numpy as np
import math

vid = cv2.VideoCapture(0)
     
while(1):
        
    try:  
          
        ret, frame = vid.read()
        frame=cv2.flip(frame,1)
        kernel = np.ones((3,3),np.uint8)
        
        roi=frame[100:300, 100:300]
        
        
        cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)    
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        
     
         # deri rengi alt sınır 0,20,70 , üst sınır 20,255,255
        lower_skin = np.array([0,20,70], dtype=np.uint8)
        upper_skin = np.array([20,255,255], dtype=np.uint8)
       
        
        #maskeleme işlemleri
        #inrange : görüntüden belirli bir bölgeyi veya rengi  bölme
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        mask = cv2.dilate(mask,kernel,iterations = 4)
        
        mask = cv2.GaussianBlur(mask,(5,5),100) 
        
        
        #findContours : bir görüntünün dış hatlarını(sınır çizgileri)  bulma
        #mask : konturları arayacağımız değer , RETR_TREE : kontur alma komutu ,
        #CHAIN_APPROX_SIMPLE : kontur yaklasım yöntemi
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # _,contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
        cnt = max(contours, key = lambda x: cv2.contourArea(x)) # en büyük kontur değerini(alanları) döndürme
        
        
        #epsilon, approx konturlara daha iyi bir yaklaşım
        epsilon = 0.0005*cv2.arcLength(cnt,True)
        # kontura yaklaşarak sınır çizgilerinin çizilmesini sağlar
        #arclength : Konturların alanını elde etmek için uygulanır Kapalı parametresi, eğrinin kapatılıp kapatılmaması gerektiğini gösterir.
        
        
        approx= cv2.approxPolyDP(cnt,epsilon,True)
        #aprroxPolyDp : Bir kontur şeklinin yaklaşıklığını gerçekleştirmek için kullanılır
        
        
        hull = cv2.convexHull(cnt)
        
        # Convex Hull ile elimize dışbükey örtü oluşturma
        
        
        areaHull = cv2.contourArea(hull)
        # Dışbükey çokgen alanı
        
        areaCnt = cv2.contourArea(cnt)
        # kontur alanı
        
        
        areaRatio=((areaHull-areaCnt)/areaCnt)*100 # elimizin olmadığı alan
        # bütün alanın oranı
        
        
        #Cismin gövdeden herhangi bir sapması varsa bu dışbükeylik kusuru olarak kabul edilebilir.
        #OpenCV, bunu bulmak için hazır bir işlevle birlikte gelir.
        hull = cv2.convexHull(approx, returnPoints=False) 
        #dış bükey gövdeyi bulmak için returnPoints=False
        defects = cv2.convexityDefects(approx, hull)
        #Dışbükey kusuru convexityDefects
        
        
        # l= kusur sayısı , başta 0 verdim
        # defects içindeki değişkenlerin değerlerini kusurlara atama (çizim yapmak için)
        l=0
        
        for i in range(defects.shape[0]):  # defects değerlerinin hepsini dolaştırma
            s,e,f,d = defects[i,0] # başlangıç ve bitiş değerleri
            start = tuple(approx[s][0]) # 0.indisteki değerler s ye eşitlenir
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            
            
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a+b+c)/2
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
            
            
            d=(2*ar)/a
            
            
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            #acos : cos kuralını uygulama , iki kenar arasındaki açıyı bulma
            # bu ifade bir kuraldır
        
        
            if angle <= 90 and d>30: 
                # şartı sağlıyorsa kusur var
                l += 1
                
                cv2.circle(roi, far, 3, [255,0,0], -1) # kusurları içi dolu çemberler olarak çizme
            
            cv2.line(roi,start, end, [0,255,0], 2) #start ve end kullanark çizgi çiz
            
            
        l+=1
        
        
        
        # cv2.putText() herhangi bir görüntüye bir metin dizesi çizmek için kullanılır
        # org: Resimdeki metin dizesinin sol alt köşesinin koordinatları,( X değeri, Y değeri).
        #font: Yazı tipini belirtir. Yazı tiplerinden bazıları FONT_HERSHEY_SIMPLE
        #fontScale: Yazı tipine özgü yazı tipi ölçek faktörü
        #(0,0,255) = kırmızı renk BGR(RGB de 255 0 0 dır) 
        #3 kalınlık 
        #cv2.LINE_AA daha iyi görüntü sağlamak için
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        if l==1:
            if areaCnt<2000:
                cv2.putText(frame,'Elinizi Alana Getirin',(0,50), font, 1, (0,0,255), 3, cv2.LINE_AA)
                
            else:
                
                if areaRatio<12:
                    
                    cv2.putText(frame,'Sonunda Oldu',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    
                elif areaRatio<17.5:
                    
                    cv2.putText(frame,'Bitirdiniz',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
         
                else:
                    
                    cv2.putText(frame,'Bekle',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    
                    
        elif l==2:
            
            if areaRatio<25:
                
               cv2.putText(frame,'2 Hak Kaldi',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
               
            elif areaRatio<35:
                
               cv2.putText(frame,' Hata Yaptiniz',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
               
            else:
                
                 cv2.putText(frame,'Alani Gecti',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        elif l==3:
         
              if areaRatio<25:
                  
                    cv2.putText(frame,'3',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA) 
                    
              elif areaRatio<35:
                  
                    cv2.putText(frame,'Onaylandi',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
              else:
                  
                    cv2.putText(frame,'Onaylanmadi',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    
       
        elif l==4:
            
            if areaRatio<15:
                
                 cv2.putText(frame,'Hazirlan',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                 
            elif areaRatio<20:
                
                cv2.putText(frame,'Cizgi Gecildi',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                
            else:
                
                 cv2.putText(frame,'Bastan Al',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                 
        elif l==5:
            
            if areaRatio<25:
                
               cv2.putText(frame,'Dur',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
               
            elif areaRatio<35:
                
                cv2.putText(frame,'Hata Aldiniz',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                
        elif l==6:
            
            cv2.putText(frame,'reposition',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        else :
            
            cv2.putText(frame,'reposition',(10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
            
        cv2.imshow('mask',mask)
        
        cv2.imshow('frame',frame)
        
    except:
        
        pass
    

# cv2 waitkey() ; anahtar kelimedeki herhangi bir düğmeye basana kadar milisaniye 
#cinsinden belirli bir süre beklemenize olanak tanır.Argüman olarak zamanı milisaniye cinsinden
#kabul eder.O sırada herhangi bir tuşa basarsanız program devam eder. 0 iletilirse 
#bir tuş vuruşu için süresiz olarak bekler.
 
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
    #cv2.destroyAllWindows() , oluşturulan tüm pencereleri basitçe yok eder
    
cv2.destroyAllWindows()

vid.release()    
    

