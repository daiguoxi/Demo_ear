import re
import os
import matplotlib.pyplot as plt

f1=open(r'E:\Python项目集\Pytorch(new)\第二次文件\desnenet121_record.txt','r')
f2=open(r'E:\Python项目集\Pytorch(new)\第二次文件\desnenet121_record_val.txt','r')
f3=open(r'E:\Python项目集\Pytorch(new)\第二次文件\resnet101_record.txt','r')
f4=open(r'E:\Python项目集\Pytorch(new)\第二次文件\resnet101_record_val.txt','r')
ta121=[]
tl121=[]
va121=[]
vl121=[]

ta101=[]
tl101=[]
va101=[]
vl101=[]

def ini(fi1,fi2,ta,tl,va,vl):
	while True:
		line=fi1.readline().strip()#Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
		if not line:break
		line=re.sub(',','',line)#去除逗号
		'''
		line ="380 0.000000 10000 Voltage v"
		a=re.sub('\S', '@', line)  #[\S]表示，非空白就匹配, 用@替换所有非空白字符串
		b=re.sub('\s', '@', line)  #[\s]表示，只要出现空白就匹配, 用@替换
		c=re.sub('\w', '@', line)  #\w可以匹配一个字母或数字
		d=re.sub('\d', '@', line)  #用\d可以匹配一个数字
		e=re.sub('\.', '@', line)  #用\.可以匹配.
		程序运行结果为： 
		a: @@@ @@@@@@@@ @@@@@ @@@@@@@ @
		b: 380@0.000000@10000@Voltage@v
		c: @@@ @.@@@@@@ @@@@@ @@@@@@@ @
		d: @@@ @.@@@@@@ @@@@@ Voltage v
		e: 380 0@000000 10000 Voltage v
		'''
		ele=line.split()#通过指定分隔符对字符串进行切片,默认为所有的空字符
		ta.append(float(ele[3]))
		tl.append(float(ele[5]))
		#exit()
	while True:
		line=fi2.readline().strip()
		if not line:break
		line=re.sub(',','',line)
		ele=line.split()
		va.append(float(ele[3]))
		vl.append(float(ele[5]))
#ini(f1,f2,ta121,tl121,va121,vl121)
ini(f3,f4,ta101,tl101,va101,vl101)

x=[]
for i in range(200):
	i+=1
	x.append(i)
#fig=plt.figure()
#plt.subplot(2, 1, 1)
fig=plt.figure()
ax = fig.add_subplot(111)
l1=ax.plot(x, ta101, '--',  label="resnet101Training accuracy")
l2=ax.plot(x, va101, '--',  label="resnet101Validation accuracy")
ax2=ax.twinx()
l3=ax2.plot(x, tl101, '--',color='r',  label="resnet101Training loss")
l4=ax2.plot(x, vl101, '--', color='g', label="resnet101Validation loss")
#ax2.set_ylim(0.6,0.8)
plt.title("40 Learning Curve for Resnet101")
ax.set_ylabel("Accuracy")
ax2.set_ylabel("Loss")
ax.set_xlabel("Epoch")
#ax.legend()
#ax2.legend()
la=l1+l2+l3+l4
labs = [l.get_label() for l in la]
ax.legend(la,labs,loc="upper left")
#ax.legend(loc=1)
#ax2.legend(loc=1)
plt.savefig("2019310143040戴国息resnet101.pdf",dpi=300)
exit()

'''
plt.plot(x, ta1615, '--',  label="Training score")
plt.plot(x, va1615, '--',  label="Validation score")
plt.plot(x, tl1615, '--',  label="Training loss")
plt.plot(x, vl1615, '--',  label="Validation loss")
plt.title("Learning Curve for Densenet 1615")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
'''
'''
plt.subplot(2, 1, 2)
plt.plot(x, ta169, '--',  label="Training score")
plt.plot(x, va169, '--',  label="Validation score")
plt.plot(x, tl169, '--',  label="Training loss")
plt.plot(x, vl169, '--',  label="Validation loss")
plt.title("Learning Curve for Densenet 169")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig("LC.png",dpi=300)
'''
